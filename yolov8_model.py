
class YOLOv8:
    def __setup__(self):
        from ultralytics import YOLO
        import os

        LOCAL_MACHINE = False
        if LOCAL_MACHINE:
            model_dir = os.path.expanduser('~/models/')
            self.model = YOLO(os.path.join(model_dir, 'yolov8l.pt'))
            self.fast_model = YOLO(os.path.join(model_dir, 'yolov8n.pt'))
            self.face_model = YOLO(os.path.join(model_dir, "yolov8l-face.pt"))
            self.face_fast_model = self.face_model
        else:
            self.model = YOLO('/root/.models/yolov8l.pt')
            self.fast_model = YOLO('/root/.models/yolov8n.pt')
            self.face_model = YOLO("/root/.models/yolov8l-face.pt")
            self.face_fast_model = self.face_model

    def __predict__(
            self,
            file: str,
            confidence_threshold: float = 0.05,
            classes: str = "person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush",
            models: str = "yolov8l",
            start_frame: int = 0,
            end_frame: int = -1,
            fps: float = -1,
            max_num_boxes: int = -1,
        ):
        """
        :param file: Image or video file.
        :param confidence_threshold: Confidence threshold for the predictions.
        :param return_visualization: Whether to return the visualization of the results.
        :param classes: The classes to use for inference. The classes are specified as a comma-separated string. Only applicable if the model is yolov8l-world or yolov8s-world which support natural language prompts. The default classes are the COCO classes.
        :param models: The models to use for inference. The models are specified as a comma-separated string. The supported models are yolov8l, yolov8n, yolov8l-face, yolov8n-face, yolov8l-world, and yolov8s-world. The default model is yolov8l. If multiple models are specified, the results from all the models are combined.
        :param start_frame: The frame number to start processing from. Defaults to 0.
        :param end_frame: The frame number to stop processing at. Defaults to -1, which means the end of the video.
        :param speed_boost: Whether to use the faster version of YOLOv8. This is less accurate but faster.
        :param fps: The fps to process the video at. Defaults to -1, which means the original fps of the video. If the specified fps is higher than the original fps, the original fps is used.
        :param max_num_boxes: The maximum number of boxes to return per frame. Defaults to -1, which means all boxes are returned. Otherwise, the boxes are sorted by confidence and the top max_num_boxes are returned.
        :return: A dictionary with the results.
        """
        print("Got request...")
        import cv2
        import os
        import time
        import torch
      
        video_extensions = ["mp4", "avi", "mov", "flv", "mkv", "wmv", "mpg", "mpeg", "m4v"]
        image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]

        # split the models string by comma and remove any whitespace
        models = [model.strip() for model in models.split(",")]
        models_to_use = []

        def process_categories(categories):
            categories = categories.split(",")
            categories = [category.strip() for category in categories]
            return categories + [" "]
        # remove any duplicate models
        models = list(set(models))
        for model in models:
            model = model.strip()
            if model == "yolov8l":
                models_to_use.append(self.model)
            elif model == "yolov8n":
                models_to_use.append(self.fast_model)
            elif model == "yolov8l-face":
                models_to_use.append(self.face_model)
            else:
                raise ValueError(f"Unsupported model: {model}")

        file_extension = os.path.splitext(file)[1][1:].lower()
        print("Starting inference...")

        if file_extension in video_extensions:
            video_path = file
            cap = cv2.VideoCapture(video_path)
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if fps == -1 or fps > original_fps:
                fps = original_fps
                
            if end_frame == -1:
                end_frame = int(num_frames) - 1
            else:
                end_frame = min(end_frame, int(num_frames) - 1)

            start_time = time.time()
            outputs = []

            frames_number_to_read = []
            for i in range(int(end_frame - start_frame) + 1):
                frame_number = int(start_frame + i * (original_fps / fps))
                if start_frame <= frame_number < end_frame:
                    frames_number_to_read.append(frame_number)

            if end_frame not in frames_number_to_read:
                frames_number_to_read.append(end_frame)
            
            t = time.time()
            import imageio
            cap = imageio.get_reader(file)
            current_frame_number = start_frame
            try:
                if start_frame != 0:
                    cap.set_image_index(start_frame)
                current_frame = cap.get_next_data()
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
            except IndexError:
                return [{"frame_number": current_frame_number, "boxes": []}]
            for p in frames_number_to_read:
                frame_to_process = None
                if p == current_frame_number:
                    frame_to_process = current_frame
                    # convert the frame to RGB
                    frame_to_process = cv2.cvtColor(frame_to_process, cv2.COLOR_RGB2BGR)
                else:
                    new_frame_number = p
                    if new_frame_number != current_frame_number + 1:
                        # print(f"Seeking to frame {new_frame_number}")
                        cap.set_image_index(new_frame_number)
                    try:
                        new_frame = cap.get_next_data()
                    except IndexError:
                        break
                    if new_frame is not None:
                        current_frame_number = p
                        current_frame = new_frame
                        frame_to_process = current_frame
                        # frame_to_process = cv2.cvtColor(frame_to_process, cv2.COLOR_RGB2BGR)
                    else:
                        break

                if frame_to_process is not None:
                    combined_boxes = []
                    for x, model_to_use in enumerate(models_to_use):
                        results = model_to_use.predict(frame_to_process, conf=confidence_threshold, verbose=False)
                        results_dict = self.__process_results__(results, model_to_use)
                        # Filter results based on confidence threshold
                        results_dict["boxes"] = [box for box in results_dict["boxes"] if box["confidence"] > confidence_threshold]
                        combined_boxes.extend(results_dict["boxes"])
                    
                    if max_num_boxes != -1 and len(combined_boxes) > max_num_boxes:
                        combined_boxes = sorted(combined_boxes, key=lambda x: x["confidence"], reverse=True)[:max_num_boxes]
                    
                    output_dict = {"frame_number": p, "boxes": combined_boxes}
                    outputs.append(output_dict)

                if len(outputs) % 100 == 0:
                    print(f"Processed {round((len(outputs) / len(frames_number_to_read) * 100), 2)}% of frames ({len(outputs)} / {len(frames_number_to_read)})")
            cap.close()
            end_time = time.time()
            fps = len(outputs) / (end_time - start_time)
            print(f"Processing FPS: {fps}")
            return outputs
        elif file_extension in image_extensions:
            image_path = file
            combined_boxes = []
            for x, model_to_use in enumerate(models_to_use):
                results = model_to_use.predict(image_path, conf=confidence_threshold, verbose=False)
                results_dict = self.__process_results__(results, model_to_use)

                # Filter results based on confidence threshold
                results_dict["boxes"] = [
                    box
                    for box in results_dict["boxes"]
                    if box["confidence"] > confidence_threshold
                ]
                combined_boxes.extend(results_dict["boxes"])

            return {"boxes": combined_boxes}
        else:
            raise ValueError(
                f"Unsupported file extension: {file_extension}. Supported extensions are {video_extensions + image_extensions}"
            )

    def __process_results__(self, results, model_to_use) -> dict:
        results_dict = {
            "boxes": [],
        }
        for result in results:
            # Append boxes information to the dictionary
            for box in result.boxes:
                box_info = {
                    "x1": int(box.xyxy.cpu().numpy().tolist()[0][0]),
                    "y1": int(box.xyxy.cpu().numpy().tolist()[0][1]),
                    "x2": int(box.xyxy.cpu().numpy().tolist()[0][2]),
                    "y2": int(box.xyxy.cpu().numpy().tolist()[0][3]),
                    "width": int(box.xyxy.cpu().numpy().tolist()[0][2]
                    - box.xyxy.cpu().numpy().tolist()[0][0]),
                    "height": int(box.xyxy.cpu().numpy().tolist()[0][3]
                    - box.xyxy.cpu().numpy().tolist()[0][1]),
                    "confidence": float(box.conf.cpu().numpy().tolist()[0]),
                    "class_number": int(box.cls.cpu().numpy().tolist()[0]),
                }
                box_info["class_name"] = model_to_use.names[box_info["class_number"]]
                results_dict["boxes"].append(box_info)

        return results_dict