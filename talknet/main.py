class TalkNetASD:
    def __setup__(self):
        from talknet.demoTalkNet import setup
        self.s, self.DET = setup()

    def __predict__(
        self,
        video: str,
        start_time: float = 0,
        end_time: float = -1,
        return_visualization: bool = False,
        face_boxes: str = "",
        in_memory_threshold: int = 0,
    ):
        """
        :param video: a video to process
        :param start_time: the start time of the video to process (in seconds)
        :param end_time: the end time of the video to process (in seconds). If -1, process until the end of the video.
        :param return_visualization: whether to return the visualization of the video.
        :param face_boxes: a string of face boxes in the format "frame_number,x1,y1,x2,y2,x1,y1,x2,y2,..." separated by new lines per frame. If not provided, the model will detect the faces in the video itself to then detect the active speaker.
        :param in_memory_threshold: the maximum number of frames to load in memory at once. can speed up processing. if 0, this feature is disabled.
        :return: if return_visualization is True, the first element of the tuple is the output of the model, and the second element is the visualization of the video. Otherwise, the first element is the output of the model.
        """
        import gc
        gc.collect()
        from talknet.demoTalkNet import main
        def transform_out(out):
            outputs = []
            for o in out:
                outputs.append({
                    "frame_number": o['frame_number'],
                    "boxes": [b for b in o['faces']]
                })
            return outputs
            
        if return_visualization:
            out, video_path = main(self.s, self.DET, video, start_seconds=start_time, end_seconds=end_time, return_visualization=return_visualization, face_boxes=face_boxes, in_memory_threshold=in_memory_threshold)
            return video_path
        else:
            out = main(self.s, self.DET, video, start_seconds=start_time, end_seconds=end_time, return_visualization=return_visualization, face_boxes=face_boxes, in_memory_threshold=in_memory_threshold)
            return transform_out(out)

if __name__ == "__main__":
    TEST_URL = "https://storage.googleapis.com/sieve-prod-us-central1-public-file-upload-bucket/d979a930-f2a5-4e0d-84fe-a9b233985c4e/dba9cbf3-8374-44bc-8d9d-cc9833d3f502-input-file.mp4"
    test_vid = "test_r2.mp4"
    model = TalkNetASD()
    model.__setup__()  # Make sure to setup the model
    # change "url" to "path" if you want to test with a local file
    out = model.__predict__(test_vid, return_visualization=False)
    print(list(out))