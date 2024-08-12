from utils import get_video_dimensions, get_video_length, create_video_segments
from custom_types import VideoSegment, Frame, Box
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from talknet.main import TalkNetASD
from yolov8_model import YOLOv8

SPEAKER_DETECTION_IN_MEMORY_THRESHOLD = 3000

def run_yolov8(yolov8_instance, file, confidence_threshold, start_frame, end_frame, models, fps, max_num_boxes):
    return yolov8_instance.__predict__(
        file=file,
        confidence_threshold=confidence_threshold,
        models=models,
        start_frame=start_frame,
        end_frame=end_frame,
        fps=fps,
        max_num_boxes=max_num_boxes
    )

def run_talknetasd(talknet_instance, file, start_time, end_time, face_boxes):
    detection_results = talknet_instance.__predict__(
        video=file,
        start_time=start_time,
        end_time=end_time,
        return_visualization=False,
        face_boxes=face_boxes,
        in_memory_threshold=SPEAKER_DETECTION_IN_MEMORY_THRESHOLD,  # Set in_memory_threshold according to your needs
    )

    return detection_results

def push_video_segments_to_object_detection(yolov8_instance, video_segment, file, frame_interval=600, models="yolov8l, yolov8l-face", processing_fps=2):
    # push the video segments to object detection for every frame_interval frames
    total_num_frames = video_segment.end_frame if video_segment.end_frame else int(video_segment.end * video_segment.fps())
    start = video_segment.start_frame if video_segment.start_frame else int(video_segment.start * video_segment.fps())
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(start, total_num_frames, frame_interval):
            start_frame = i
            end_frame = min(i + frame_interval - 1, total_num_frames - 1)
            
            future = executor.submit(
                run_yolov8,
                yolov8_instance,
                file,
                0.25,
                start_frame,
                end_frame,
                models,
                processing_fps,
                30  # max_num_boxes
            )
            futures.append({
                "future": future,
                "start": start_frame,
                "end": end_frame,
            })

    return futures

def get_active_speakers(speaker_frames, alpha=0.5, score_threshold=0):
    # smooth the scores of the boxes from the speaker detection model
    active_speakers = {}
    smoothed_scores = {}

    for frame in speaker_frames:
        frame_number = frame['frame_number']
        active_speakers[frame_number] = []

        for box in frame['boxes']:
            box_id = box['track_id']
            raw_score = box['raw_score']

            if box_id not in smoothed_scores:
                smoothed_scores[box_id] = raw_score
            else:
                smoothed_scores[box_id] = alpha * raw_score + (1 - alpha) * smoothed_scores[box_id]

            if smoothed_scores[box_id] > score_threshold:
                active_speakers[frame_number].append(Box(class_id=-1, confidence=1.0, x1=box['x1'], y1=box['y1'], x2=box['x2'], y2=box['y2'], id=box_id, metadata={'raw_score': smoothed_scores[box_id]}))

    return active_speakers

def process(
    yolov8_instance, 
    talknet_instance,
    file: str,
    speed_boost: bool = False,
    max_num_faces: int = 5,
    return_scene_cuts_only: bool = False,
    return_scene_data: bool = False,
    start_time: float = 0,
    end_time: float = -1,
    processing_fps: float = 2,
    face_size_threshold: float = 0.5,
):
    '''
    :param file: The video file to process
    :param speed_boost: Whether to use the faster but less accurate object detection model when processing the video.
    :param max_num_faces: The maximum number of faces to return per frame. If there are more than this number of faces, only the largest x faces will be returned.
    :param return_scene_cuts_only: Whether to only return the frame data at the start of each scene cut or to return the frame data for every frame in the video.
    :param return_scene_data: Whether to return the scene data along with the frame data. If True, the scene data will be returned in the "related_scene" field of the output.
    :param start_time: The seconds into the video to start processing from. Defaults to 0.
    :param end_time: The seconds into the video to stop processing at. Defaults to -1, which means the end of the video.
    :param processing_fps: The framerate to run object detection at for speaker detection. Defaults to 2.
    :param face_size_threshold: A threshold that determines the minimum size of a face to run speaker detection on. Defaults to 0.5. Lower values allow for smaller faces to be used.
    '''
    # handle webm files by converting them to mp4
    if file.endswith(".webm"):
        print("Converting webm file to mp4...")
        import os
        import subprocess
        new_path = file.replace(".webm", ".mp4")
        subprocess.run(["ffmpeg", "-y", "-i", file, "-c", "copy", new_path])
        file = new_path

    width, height = get_video_dimensions(file)
    original_video_width = width
    original_video_height = height
    original_video_length = get_video_length(file)

    models = "yolov8l-face" if speed_boost else "yolov8n-face"
    if end_time == -1:
        end_time = original_video_length

    if start_time < 0 or start_time > original_video_length:
        raise ValueError(f"start_time must be between 0 and {original_video_length}")
    if end_time < 0 or end_time > original_video_length:
        raise ValueError(f"end_time must be between 0 and {original_video_length}")
    if start_time >= end_time:
        raise ValueError(f"start_time must be less than end_time")
    
    original_video = VideoSegment(
        path=file,
        start=start_time,
        end=end_time,
    )

    original_video_fps = original_video.fps()

    scene_detection_result = queue.Queue()
    object_detection_result = queue.Queue()

    # Define a wrapper function to call scene_detection and put the result in the Queue
    def scene_detection_wrapper(file, result_queue, **kwargs):
        print("Cutting video into scene segments...")
        from scene_detection import scene_detection
        result = list(scene_detection(file, **kwargs))
        result_queue.put(result)
        print("Done cutting video into scene segments")

    # Define a wrapper function to call push_video_segments_to_object_detection and put the result in the Queue
    def object_detection_wrapper(original_video, file, result_queue, frame_interval):
        print("Pushing video to object detection...")
        result = push_video_segments_to_object_detection(yolov8_instance, original_video, file, frame_interval=frame_interval, models=models, processing_fps=processing_fps)
        result_queue.put(result)
        print("Done pushing video to object detection")
    
    scene_detection_thread = threading.Thread(target=scene_detection_wrapper, args=(file, scene_detection_result), kwargs={'adaptive_threshold': True})
    scene_detection_thread.start()

    num_frames_to_process = (end_time - start_time) * original_video_fps
    frame_interval = max(300, int(num_frames_to_process / 100))
    object_detection_thread = threading.Thread(target=object_detection_wrapper, args=(original_video, file, object_detection_result, frame_interval))

    object_detection_thread.start()
    object_detection_thread.join()

    scene_detection_thread.join()
    scene_future = scene_detection_result.get()

    segments = create_video_segments(file, scene_future, start_time=start_time, end_time=end_time, fps=original_video_fps, original_video_length=original_video_length)
    total_num_frames = original_video.end_frame if original_video.end_frame else int(original_video.end * original_video_fps)

    speaker_detection_futures = []
    total_num_frames = original_video.end_frame if original_video.end_frame else int(original_video.end * original_video.fps())
    start = original_video.start_frame if original_video.start_frame else int(original_video.start * original_video.fps())
    for i in range(start, total_num_frames, frame_interval):
        start_frame = i
        end_frame = min(i + frame_interval - 1, total_num_frames - 1)
        start = start_frame / original_video_fps
        end = end_frame / original_video_fps            
        speaker_detection_futures.append({
            "future": None,
            "start": start_frame,
            "end": end_frame,
        })

    object_detection_futures = object_detection_result.get()

    def seconds_to_timecode(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        new_seconds = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{new_seconds:02d}.{milliseconds:03d}"

    def convert_face_detection_outputs_to_string(face_detection_outputs):
        # convert the face detection outputs to a string that can be used as input to the speaker detection model
        out_str = ""
        frame_size = original_video_width * original_video_height
        interpolated_face_detection_outputs = []
        start = face_detection_outputs[0]["frame_number"]
        end = face_detection_outputs[-1]["frame_number"]
        for i in range(start, end + 1):
            frame_segment = None
            for segment in segments:
                segment_start_frame = segment.start_frame if segment.start_frame else int(segment.start * original_video_fps)
                segment_end_frame = segment.end_frame if segment.end_frame else int(segment.end * original_video_fps)
                if i >= segment_start_frame and i <= segment_end_frame:
                    frame_segment = segment
                    break
            if frame_segment is None:
                outputs_to_choose_from = face_detection_outputs
            else:
                outputs_to_choose_from = [output for output in face_detection_outputs if output["frame_number"] >= segment_start_frame and output["frame_number"] <= segment_end_frame]
            if len(outputs_to_choose_from) == 0:
                continue
            closest_frame = min(outputs_to_choose_from, key=lambda x: abs(x["frame_number"] - i))
            # copy the boxes to avoid modifying the original and set the frame number to the current frame
            new_frame = {
                "frame_number": i,
                "boxes": [box.copy() for box in closest_frame["boxes"]],
            }
            interpolated_face_detection_outputs.append(new_frame)
        for frame in interpolated_face_detection_outputs:
            out_str += f"{frame['frame_number']},"
            new_boxes = []
            # onyl keep the face boxes that are greater than 1/50 of the frame size
            for box in frame['boxes']:
                box_area = (box['x2'] - box['x1']) * (box['y2'] - box['y1'])
                if box_area > (face_size_threshold / 100) * frame_size and box['confidence'] > 0.5:
                    new_boxes.append(box)
            for box in new_boxes:
                if box['class_name'] != "face":
                    continue
                out_str += f"{box['x1']:.2f},{box['y1']:.2f},{box['x2']:.2f},{box['y2']:.2f},{box['confidence']:.2f},"
            # remove the last comma
            out_str = out_str[:-1]
            out_str += "\n"
        return out_str
    
    def get_relevant_face_detection_future(i):
        with ThreadPoolExecutor(max_workers=4) as executor:
            for j, future in enumerate(object_detection_futures):
                if "result" not in future and future["future"].done():
                    try:
                        res = future["future"].result()
                        object_detection_futures[j]["result"] = res
                    except:
                        print(f"WARNING: Found failed object detection (frame {future['start']}-{future['end']}), retrying...")
                        object_detection_futures[j]["future"] = executor.submit(
                            run_yolov8,
                            yolov8_instance,
                            file,
                            0.25,
                            future["start"],
                            future["end"],
                            models,
                            processing_fps,
                            30  # max_num_boxes
                        )
                        continue
            if "result" in object_detection_futures[i]:
                return object_detection_futures[i]["result"]
            else:
                for _ in range(10):  # retry up to 10 times
                    try:
                        res = object_detection_futures[i]["future"].result()
                        object_detection_futures[i]["result"] = res
                        return res
                    except:
                        # HERE IS OUR FAILURE POINT
                        print(f"WARNING: Object detection failed, retrying... Attempt {_+1}/10")
                        if "result" in object_detection_futures[i]:
                            del object_detection_futures[i]["result"]
                        object_detection_futures[i]["future"] = executor.submit(
                            run_yolov8,
                            yolov8_instance,
                            file,
                            0.25,
                            object_detection_futures[i]["start"],
                            object_detection_futures[i]["end"],
                            models,
                            processing_fps,
                            30  # max_num_boxes
                        )
                raise Exception("Object detection failed 10 times, please try again later.")
    
    def get_speaker_detection_payload(start, end, fps=30):
        # find all indices that contain the start and end
        def refresh_speakers():
            with ThreadPoolExecutor(max_workers=4) as executor:
                for i, future in enumerate(speaker_detection_futures):
                    in_range = (start >= future["start"] and start <= future["end"]) or (end >= future["start"] and end <= future["end"]) or (start <= future["start"] and end >= future["end"])
                    relevant_future_done = object_detection_futures[i]["future"].done()
                    if not future["future"] and (in_range or relevant_future_done):
                        # this means that we haven't pushed the video to speaker detection yet since face detection is not done
                        # lets find the relevant face detection future, wait for it to be done, and then push the video to speaker detection
                        res = list(get_relevant_face_detection_future(i))
                        face_detection_outputs = convert_face_detection_outputs_to_string(res)
                        speaker_detection_futures[i]["future"] = executor.submit(
                            run_talknetasd,
                            talknet_instance,
                            file,
                            future["start"] / fps,
                            future["end"] / fps,
                            face_detection_outputs
                        )
        refresh_speakers()
        for i, future in enumerate(speaker_detection_futures):
            if (start >= future["start"] and start <= future["end"]) or (end >= future["start"] and end <= future["end"]) or (start <= future["start"] and end >= future["end"]):
                for _ in range(10):  # retry up to 10 times
                    try:
                        if "result" not in speaker_detection_futures[i]:
                            while not speaker_detection_futures[i]["future"].done():
                                import time
                                time.sleep(0.1)
                                refresh_speakers()
                            speaker_detection_futures[i]["result"] = list(speaker_detection_futures[i]["future"].result())
                        break  # if successful, break the retry loop
                    except Exception as e:
                        print(f"WARNING: Speaker detection failed, retrying... Attempt {_+1}/10")
                        if "result" in speaker_detection_futures[i]:
                            del speaker_detection_futures[i]["result"]
                        # recreate the future

                        res = list(get_relevant_face_detection_future(i))
                        face_detection_outputs = convert_face_detection_outputs_to_string(res)
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            speaker_detection_futures[i]["future"] = executor.submit(
                                run_talknetasd,
                                talknet_instance,
                                file,
                                future["start"] / fps,
                                future["end"] / fps,
                                face_detection_outputs
                            )

                if "result" not in speaker_detection_futures[i]:
                    raise Exception("Speaker detection failed 10 times, please try again later.")
                data = speaker_detection_futures[i]["result"]
                for frame in data:
                    if frame["frame_number"] >= start and frame["frame_number"] <= end:
                        yield frame

    def refresh_futures():
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i, future in enumerate(object_detection_futures):
                if "result" in future and speaker_detection_futures[i]["future"] is None:
                    res = list(get_relevant_face_detection_future(i))
                    face_detection_outputs = convert_face_detection_outputs_to_string(res)
                    speaker_detection_futures[i]["future"] = executor.submit(
                        run_talknetasd,
                        talknet_instance,
                        file,
                        future["start"] / fps,
                        future["end"] / fps,
                        face_detection_outputs
                )
            
                # print("bro",len(speaker_detection_futures), len(object_detection_futures))
                if speaker_detection_futures[i]["future"] and "result" not in speaker_detection_futures[i] and speaker_detection_futures[i]["future"].done():
                    try:
                        res = speaker_detection_futures[i]["future"].result()
                        speaker_detection_futures[i]["result"] = res
                    except:
                        print(f"WARNING: Found failed speaker detection (frame {future['start']}-{future['end']}), retrying...")
                        res = list(get_relevant_face_detection_future(i))
                        face_detection_outputs = convert_face_detection_outputs_to_string(res)
                        speaker_detection_futures[i]["future"] = executor.submit(
                            run_talknetasd,
                            talknet_instance,
                            file,
                            future["start"] / fps,
                            future["end"] / fps,
                            face_detection_outputs
                        )
                        continue
            
            for i, future in enumerate(object_detection_futures):
                if "result" not in future and future["future"].done():
                    try:
                        res = future["future"].result()
                        object_detection_futures[i]["result"] = res
                    except:
                        print(f"WARNING: Found failed object detection (frame {future['start']}-{future['end']}), retrying...")
                        object_detection_futures[i]["future"] = executor.submit(
                            run_yolov8,
                            yolov8_instance,
                            file,
                            0.25,
                            future["start"],
                            future["end"],
                            "yolov8n, yolov8n-face" if speed_boost else "yolov8l, yolov8l-face",
                            processing_fps,
                            30  # max_num_boxes
                        )
                        continue
            
            # loop over futures to see if any have errored, if so, recreate the future
            for i, future in enumerate(speaker_detection_futures):
                if "result" not in future and future["future"] and future["future"].done():
                    try:
                        res = future["future"].result()
                        speaker_detection_futures[i]["result"] = res
                    except:
                        print(f"WARNING: Found failed speaker detection (frame {future['start']}-{future['end']}), retrying...")
                        res = list(get_relevant_face_detection_future(i))
                        face_detection_outputs = convert_face_detection_outputs_to_string(res)
                        speaker_detection_futures[i]["future"] = executor.submit(
                            run_talknetasd,
                            talknet_instance,
                            file,
                            future["start"] / fps,
                            future["end"] / fps,
                            face_detection_outputs
                        )
                        continue

    print("------------------")
    print("Video Informaton")
    print("Video Length: {:.2f}s".format(original_video_length))
    print("Video FPS: {:.2f}".format(original_video_fps))
    start_frame = original_video.start_frame if original_video.start_frame else int(original_video.start * original_video_fps)
    end_frame = original_video.end_frame if original_video.end_frame else int(original_video.end * original_video_fps)
    print("Start Time: ", start_time, f"(Frame {start_frame})")
    print("End Time: ", end_time, f"(Frame {end_frame})")
    print("Start Frame: ", start_frame)
    print("End Frame: ", end_frame)
    print("Number of Scenes: ", len(segments))
    print("------------------")
    print("Processing video...")

    frame_count = 0
    for segment_index, segment in enumerate(segments):
        print(f"Processing scene {segment_index} [{segment.start:.2f}s - {segment.end:.2f}s] / {end_time:.2f}s")
        start = segment.start
        end = segment.end
        fps = original_video_fps

        if (segment.start_frame or segment.start_frame == 0) and segment.end_frame:
            start_frame = segment.start_frame
            end_frame = segment.end_frame - 1
        else:
            start_frame = int(start * fps)
            end_frame = round(end * fps)

        def get_frames():
            speaker_payload = get_speaker_detection_payload(start_frame, end_frame, fps=fps)
            frames = []
            last_frame_number = None
            for frame in speaker_payload:
                if last_frame_number == frame["frame_number"]:
                    continue
                boxes = []
                for box in frame["boxes"]:
                    x1 = max(0, box["x1"])
                    y1 = max(0, box["y1"])
                    x2 = min(original_video_width, box["x2"])
                    y2 = min(original_video_height, box["y2"])
                    boxes.append(Box(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=1.0,
                        class_id=0,
                        metadata={'raw_score': box['raw_score']},
                    ))
                frames.append(Frame(
                    boxes=boxes,
                    number=frame["frame_number"],
                    width=original_video_width,
                    height=original_video_height,
                ))
                if len(frames) == 1 and frame["frame_number"] != start_frame:
                    start_frame_boxes = frames[0].boxes
                    num_start_to_add = frame["frame_number"] - start_frame
                    first_frame_number = frame["frame_number"]
                    for i in range(num_start_to_add):
                        frames.insert(0, Frame(
                            boxes=start_frame_boxes,
                            number=first_frame_number - i - 1,
                            width=original_video_width,
                            height=original_video_height,
                        ))
                        for frame1 in frames:
                            yield frame1                 
                else:
                    if last_frame_number and frames[-1].number - last_frame_number > 1:
                        frame_to_add_back = frames[-1]
                        frames = frames[:-1]
                        for i in range(last_frame_number + 1, frame_to_add_back.number):
                            frames.append(Frame(
                                boxes=frame_to_add_back.boxes,
                                number=i,
                                width=original_video_width,
                                height=original_video_height,
                            ))
                            yield frames[-1]
                        frames.append(frame_to_add_back)
                    yield frames[-1]
                last_frame_number = frames[-1].number
            if last_frame_number != end_frame:
                end_frame_boxes = frames[-1].boxes
                last_frame_number = frames[-1].number
                num_end_to_add = end_frame - frame["frame_number"]
                for i in range(num_end_to_add):
                    frames.append(Frame(
                        boxes=end_frame_boxes,
                        number=last_frame_number + i + 1,
                        width=original_video_width,
                        height=original_video_height,
                    ))
                    yield frames[-1]

        scene_out = {}
        scene_out["start_seconds"] = start
        scene_out["end_seconds"] = end
        scene_out["start_frame"] = start_frame
        scene_out["end_frame"] = end_frame
        scene_out["start_timecode"] = seconds_to_timecode(start)
        scene_out["end_timecode"] = seconds_to_timecode(end)
        scene_out["scene_number"] = segment_index
        refresh_futures()
        batch_frames = []
        for i, frame in enumerate(get_frames()):
            if not (start_frame <= frame.number <= end_frame):
                continue
            out_boxes = []
            for box in frame.boxes:
                out_boxes.append({
                    "x1": int(box.x1),
                    "y1": int(box.y1),
                    "x2": int(box.x2),
                    "y2": int(box.y2),
                    "speaking_score": box.metadata["raw_score"],
                    "active": box.metadata["raw_score"] > 0
                })

            # sort by box size
            out_boxes = sorted(out_boxes, key=lambda x: (x['x2'] - x['x1']) * (x['y2'] - x['y1']), reverse=True)
            # keep the max_num_faces largest boxes
            if len(out_boxes) > max_num_faces:
                out_boxes = out_boxes[:max_num_faces]
            if return_scene_data:
                batch_frames.append({
                    "frame_number": frame.number,
                    "faces": out_boxes,
                    "related_scene": scene_out,
                })
            else:
                batch_frames.append({
                    "frame_number": frame.number,
                    "faces": out_boxes,
                })
            if len(batch_frames) == 100:
                yield batch_frames
                batch_frames = []
            if return_scene_cuts_only:
                yield batch_frames
                break
            frame_count += 1
        if not return_scene_cuts_only and batch_frames:
            yield batch_frames
        
if __name__ == "__main__":
    
    yolov8_instance = YOLOv8()
    yolov8_instance.__setup__()
    talknet_instance = TalkNetASD()
    talknet_instance.__setup__()
    # add path for test video
    file = "test-videos/test_vid.mp4"
    for out in process(yolov8_instance, talknet_instance, file):
        print(out)