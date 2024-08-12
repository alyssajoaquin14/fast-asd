# Prediction interface for Cog ⚙️
# https://cog.run/python

import json
import tempfile
from cog import BasePredictor, Input, Path
from talknet.main import TalkNetASD
from yolov8_model import YOLOv8
from main import process
from typing import List, Dict

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.yolov8_instance = YOLOv8()
        self.yolov8_instance.__setup__()
        self.talknet_instance = TalkNetASD()
        self.talknet_instance.__setup__()

    def predict(
        self,
        video_file: Path = Input(description="Path to the input video file"),
        speed_boost: bool = Input(description="Use faster but less accurate model", default=False),
        max_num_faces: int = Input(description="Max number of faces per frame", default=5),
        return_scene_cuts_only: bool = Input(description="Return only scene cuts", default=False),
        return_scene_data: bool = Input(description="Return scene data", default=False),
        start_time: float = Input(description="Start time in seconds", default=0),
        end_time: float = Input(description="End time in seconds", default=-1),
        processing_fps: float = Input(description="Processing frame rate", default=2),
        face_size_threshold: float = Input(description="Face size threshold", default=0.5),
    ) -> List[Dict]:
        """Run a single prediction on the models and return a file path"""

        file = str(video_file)
        result = []

        for batch in process(
            self.yolov8_instance,
            self.talknet_instance,
            file,
            speed_boost=speed_boost,
            max_num_faces=max_num_faces,
            return_scene_cuts_only=return_scene_cuts_only,
            return_scene_data=return_scene_data,
            start_time=start_time,
            end_time=end_time,
            processing_fps=processing_fps,
            face_size_threshold=face_size_threshold,
        ):
            print(batch)
            result.append(batch)

        return result