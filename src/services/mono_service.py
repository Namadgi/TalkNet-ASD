import json
import cv2
import time
import os
from fastapi import UploadFile, HTTPException
import numpy as np
import subprocess
import torch
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from src.models import GCSRequest, ResponseModel
from src.ml_models.mono import Mono
from torchvision.io import read_video

VIDEO_TEMP = 'input.mp4'
VIDEO_OUTPUT = 'output.mp4'
AUDIO_OUTPUT = 'output.mp4'

class MonoService:
    def __init__(self):
        self.initialized = False
        self.device = None
        self.model = None
        # self.processing_lock = asyncio.Lock()
        self.initialize()

    def initialize(self):
        try:
            self.initialized = True
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = Mono()
            self.model.load_state_dict(torch.load('src/model_weights/mono.pth'))
            self.model.to(self.device)
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            self.initialized = False
            raise HTTPException(status_code=500, detail="Failed to initialize model")

    async def process_video(self, data: Any) -> ResponseModel:
        """
        Asynchronously process video with lock to ensure sequential processing
        """
        # async with self.processing_lock:
        try:
            X, v_path, a_path = self.preprocess(data)
            Y = self.inference(X, v_path, a_path)

            if len(Y) == 0:
                return self.form_response(code=2)

            res = self.postprocess(Y)
            return self.form_response(result=res)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def preprocess(self, data: Any) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
        """
        Transform raw input into model input data.
        :param data: Input data (either file upload or GCS request)
        :return: tuple of (video, video_path, audio_path)
        """
        if data is None:
            raise HTTPException(status_code=400, detail="Input data is required")

        cur_time = time.time()

        # Clean up existing files
        for file in [VIDEO_TEMP, VIDEO_OUTPUT, AUDIO_OUTPUT]:
            if os.path.isfile(file): os.remove(file)

        if isinstance(data, UploadFile):
            # Handle file upload
            with open(VIDEO_TEMP, 'wb') as out_file:
                content = data.file.read()
                out_file.write(content)
            result_object_name = VIDEO_TEMP
        else:
            # Handle GCS request
            try:
                gcs_request: GCSRequest = data
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid GCS request format: {str(e)}")

            if not gcs_request.instances or len(gcs_request.instances) == 0:
                raise HTTPException(status_code=400, detail="No instances provided in GCS request")

            instance = gcs_request.instances[0]
            if not instance.token or not instance.bucket_name or not instance.object_name:
                raise HTTPException(status_code=400, detail="Missing required GCS parameters")

            token = instance.token
            bucket_name = instance.bucket_name
            object_name = instance.object_name

            object_encoded_name = object_name.replace('/', '%2F')
            result_object_name = object_name.split('/')[-1]
            curl_cmd = f'curl -X GET -H "Authorization: Bearer {token}" -o {result_object_name} ' + \
                        f'"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o/{object_encoded_name}?alt=media"'
            
            if os.system(curl_cmd) != 0:
                raise HTTPException(status_code=404, detail="Failed to download file from GCS")

        print('DL: ', time.time() - cur_time)

        # Process video with ffmpeg
        cmd = ['ffmpeg', '-y']
        if result_object_name.lower().endswith('.webm'):
            cmd += ['-fflags', '+genpts']

        cmd += ['-i', result_object_name]
        cmd += [
            '-vf', 'scale=-2:640',
            '-qscale:v', '2',
            '-async', '1',
            '-r', '25'
        ]

        if result_object_name.lower().endswith('.webm'):
            cmd += ['-max_muxing_queue_size', '1024']

        cmd += [
            '-qscale:a', '0',
            '-ac', '1',
            '-threads', '10',
            '-ar', '16000'
        ]

        cmd += ['-loglevel', 'panic', VIDEO_OUTPUT]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            raise HTTPException(status_code=500, detail="Failed to process video with FFmpeg")

        print('VP: ', time.time() - cur_time)

        # Read video data
        video_path = os.path.join(os.getcwd(), VIDEO_OUTPUT)
        audio_path = os.path.join(os.getcwd(), AUDIO_OUTPUT)
        
        cur_time = time.time()
        try:
            data_filename = os.path.abspath(video_path)
            video = read_video(data_filename, end_pts=10, pts_unit="sec")[0].numpy()
            print('IP:', time.time() - cur_time)
            return video, video_path, audio_path
        except:
            raise HTTPException(status_code=500, detail="Failed to read video data")

    def inference(self, X: np.ndarray, v_path: str, a_path: str) -> Optional[Any]:
        """
        Internal inference methods
        :param X: Input video data
        :param v_path: Video path
        :param a_path: Audio path
        :return: output
        """
        if not self.initialized:
            raise HTTPException(status_code=503, detail="Model not initialized")

        if X is None or v_path is None or a_path is None:
            raise HTTPException(status_code=400, detail="Invalid input parameters")

        try:
            self.model.eval()
            with torch.no_grad():
                y = self.model(X, v_path, a_path, self.device)
            return y
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    def postprocess(self, y: Any) -> np.ndarray:
        """
        Return inference result.
        :param y: Model output
        :return: result
        """
        try:
            if not isinstance(y, (list, np.ndarray)) or len(y) == 0:
                raise HTTPException(status_code=500, detail="Invalid model output")
            result = np.round((np.mean(np.array(y), axis=0)), 1).astype(float)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Post-processing error: {str(e)}")

    def form_response(self, code: Optional[int] = None, result: float = 0.0) -> ResponseModel:
        descriptions = [
            'Successful check',
            'Person is not speaking',
            'No face present',
        ]
        
        if code != 2:
            result = round((result > 0).mean() * 100, 2)
            code = int(result < 4.0)

        if result < 10:
            score = round((result / 10) * 50, 2)
        else:
            score = round(50 + (result - 10) * 5 / 9, 2)

        return ResponseModel(
            code=code,
            description=descriptions[code],
            result=result,
            score=score
        )