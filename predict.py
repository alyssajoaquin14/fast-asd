# Prediction interface for Cog ⚙️
# https://cog.run/python

import json
import os
import subprocess
import tempfile
import time
from cog import BasePredictor, Input, Path
from talknet.main import TalkNetASD
from yolov8_model import YOLOv8
from main import process
from typing import List, Dict

WEIGHTS_CACHE = "/root/.models"
YOLOV8_FACE_WEIGHTS_URL = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8l-face.pt"
YOLOV8L_WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt"
YOLOV8N_WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"

def download_weights(url, dest):
    if os.path.exists(dest):
        print(f"File {dest} already exists, skipping download.")
        return
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        os.makedirs(WEIGHTS_CACHE, exist_ok=True)
        
        download_weights(YOLOV8_FACE_WEIGHTS_URL, os.path.join(WEIGHTS_CACHE, "yolov8l-face.pt"))
        download_weights(YOLOV8L_WEIGHTS_URL, os.path.join(WEIGHTS_CACHE, "yolov8l.pt"))
        download_weights(YOLOV8N_WEIGHTS_URL, os.path.join(WEIGHTS_CACHE, "yolov8n.pt"))

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
    ) -> Path:
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
            result.append(batch)

        output_path = Path(tempfile.mkdtemp()) / "result.json"
        with open(output_path, 'w') as temp_file:
            json.dump(result, temp_file)

        return Path(output_path)