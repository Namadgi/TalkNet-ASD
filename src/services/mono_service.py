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

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.exporter.cloud_monitoring import CloudMonitoringMetricsExporter
from opentelemetry.trace import Status, StatusCode

VIDEO_TEMP = 'input.mp4'
VIDEO_OUTPUT = 'output.mp4'
AUDIO_OUTPUT = 'output.mp4'

class MonoService:
    def __init__(self):
        self.initialized = False
        self.device = None
        self.model = None
        # self.processing_lock = asyncio.Lock()
        self._setup_telemetry()
        self.initialize()

    def _setup_telemetry(self) -> None:
        """Initialize OpenTelemetry tracer and meter for GCP."""
        # Tracing
        trace.set_tracer_provider(TracerProvider())
        tracer_provider = trace.get_tracer_provider()
        tracer_provider.add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter()))
        self.tracer = trace.get_tracer(__name__)

        # Metrics
        reader = PeriodicExportingMetricReader(
            exporter=CloudMonitoringMetricsExporter(),
            export_interval_millis=60000,
        )
        metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
        self.meter = metrics.get_meter(__name__)

        # Metric instruments
        self.request_counter = self.meter.create_counter(
            name="mono_requests_total",
            description="Total number of MONO requests",
            unit="1",
        )
        self.request_duration = self.meter.create_histogram(
            name="mono_request_duration_seconds",
            description="Duration of MONO requests",
            unit="s",
        )
        self.preprocess_duration = self.meter.create_histogram(
            name="mono_preprocess_duration_seconds",
            description="Duration of preprocessing operations",
            unit="s",
        )
        self.inference_duration = self.meter.create_histogram(
            name="mono_inference_duration_seconds",
            description="Duration of inference operations",
            unit="s",
        )
        self.postprocess_duration = self.meter.create_histogram(
            name="mono_postprocess_duration_seconds",
            description="Duration of postprocessing operations",
            unit="s",
        )

    def initialize(self):
        try:
            with self.tracer.start_as_current_span("model_initialization") as span:
                span.set_attribute("device.type", "cuda" if torch.cuda.is_available() else "cpu")
                self.initialized = True
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = Mono()
                self.model.load_state_dict(torch.load('src/model_weights/mono.pth'))
                self.model.to(self.device)
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("model.loaded", True)
        except Exception as e:
            with self.tracer.start_as_current_span("model_initialization") as span:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("model.loaded", False)
            self.initialized = False
            raise HTTPException(status_code=500, detail="Failed to initialize model")

    async def process_video(self, data: Any) -> ResponseModel:
        """
        Asynchronously process video with telemetry instrumentation
        """
        # async with self.processing_lock:
        with self.tracer.start_as_current_span("mono_process_video") as span:
            try:
                span.set_attribute("service.name", "mono_service")
                span.set_attribute("operation", "process_video")
                overall_start_time = time.time()

                # Preprocess
                with self.tracer.start_as_current_span("preprocess") as preprocess_span:
                    step_start_time = time.time()
                    X, v_path, a_path = self.preprocess(data)
                    preprocess_duration = time.time() - step_start_time
                    self.preprocess_duration.record(preprocess_duration)
                    preprocess_span.set_attribute("duration_seconds", preprocess_duration)
                    preprocess_span.set_attribute("video_path", v_path or "")
                    preprocess_span.set_attribute("audio_path", a_path or "")

                # Inference
                with self.tracer.start_as_current_span("inference") as inference_span:
                    step_start_time = time.time()
                    Y = self.inference(X, v_path, a_path)
                    inference_duration = time.time() - step_start_time
                    self.inference_duration.record(inference_duration)
                    inference_span.set_attribute("duration_seconds", inference_duration)

                # Postprocess + form response
                with self.tracer.start_as_current_span("postprocess") as postprocess_span:
                    step_start_time = time.time()
                    if len(Y) == 0:
                        response = self.form_response(code=2)
                    else:
                        res = self.postprocess(Y)
                        response = self.form_response(result=res)
                    postprocess_duration = time.time() - step_start_time
                    self.postprocess_duration.record(postprocess_duration)
                    postprocess_span.set_attribute("duration_seconds", postprocess_duration)
                    postprocess_span.set_attribute("response.code", response.code)
                    postprocess_span.set_attribute("response.score", response.score)

                overall_duration = time.time() - overall_start_time
                self.request_duration.record(overall_duration, {"status": "success", "response_code": response.code})
                self.request_counter.add(1, {"status": "success", "response_code": response.code})

                span.set_attribute("overall_duration_seconds", overall_duration)
                span.set_attribute("response_code", response.code)
                span.set_attribute("response_description", response.description)
                span.set_attribute("score", response.score)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                overall_duration = time.time() - locals().get("overall_start_time", time.time())
                self.request_duration.record(overall_duration, {"status": "error", "response_code": 500})
                self.request_counter.add(1, {"status": "error"})
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("error", True)
                span.set_attribute("error_message", str(e))
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
            with self.tracer.start_as_current_span("upload_file_handling") as span:
                span.set_attribute("input_type", "upload_file")
                span.set_attribute("file_name", getattr(data, 'filename', 'unknown'))
                start_ts = time.time()
                with open(VIDEO_TEMP, 'wb') as out_file:
                    content = data.file.read()
                    out_file.write(content)
                duration = time.time() - start_ts
                span.set_attribute("upload_duration_seconds", duration)
                span.set_attribute("file_size_bytes", len(content))
            result_object_name = VIDEO_TEMP
        else:
            # Handle GCS request
            with self.tracer.start_as_current_span("gcs_request_handling") as span:
                try:
                    gcs_request: GCSRequest = data
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, "Invalid GCS request format"))
                    raise HTTPException(status_code=400, detail=f"Invalid GCS request format: {str(e)}")

                if not gcs_request.instances or len(gcs_request.instances) == 0:
                    span.set_status(Status(StatusCode.ERROR, "No instances provided"))
                    raise HTTPException(status_code=400, detail="No instances provided in GCS request")

                try:
                    instance = gcs_request.instances[0]
                except (IndexError, TypeError) as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, "No usable instances provided"))
                    raise HTTPException(status_code=400, detail="No valid instances provided in GCS request")
                if not instance.token or not instance.bucket_name or not instance.object_name:
                    span.set_status(Status(StatusCode.ERROR, "Missing GCS parameters"))
                    raise HTTPException(status_code=400, detail="Missing required GCS parameters")

                token = instance.token
                bucket_name = instance.bucket_name
                object_name = instance.object_name

                span.set_attribute("gcs.bucket_name", bucket_name)
                span.set_attribute("gcs.object_name", object_name)

                object_encoded_name = object_name.replace('/', '%2F')
                result_object_name = object_name.split('/')[-1]
                curl_cmd = f'curl -X GET -H "Authorization: Bearer {token}" -o {result_object_name} ' + \
                            f'"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o/{object_encoded_name}?alt=media"'

                with self.tracer.start_as_current_span("gcs_download") as download_span:
                    start_ts = time.time()
                    exit_code = os.system(curl_cmd)
                    duration = time.time() - start_ts
                    if exit_code != 0:
                        download_span.set_status(Status(StatusCode.ERROR, "GCS download failed"))
                        download_span.set_attribute("curl_exit_code", exit_code)
                        raise HTTPException(status_code=404, detail="Failed to download file from GCS")
                    download_span.set_attribute("download_duration_seconds", duration)
                    download_span.set_attribute("download_success", True)

        print('DL: ', time.time() - cur_time)

        # Process video with ffmpeg
        with self.tracer.start_as_current_span("ffmpeg_transcoding") as ffmpeg_span:
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
            ffmpeg_span.set_attribute("input_file", result_object_name)
            ffmpeg_span.set_attribute("output_file", VIDEO_OUTPUT)
            try:
                start_ts = time.time()
                subprocess.run(cmd, check=True, capture_output=True)
                duration = time.time() - start_ts
                ffmpeg_span.set_attribute("transcode_duration_seconds", duration)
                ffmpeg_span.set_attribute("transcode_success", True)
            except subprocess.CalledProcessError as e:
                ffmpeg_span.record_exception(e)
                ffmpeg_span.set_status(Status(StatusCode.ERROR, "FFmpeg transcoding failed"))
                raise HTTPException(status_code=500, detail="Failed to process video with FFmpeg")

        print('VP: ', time.time() - cur_time)

        # Read video data
        video_path = os.path.join(os.getcwd(), VIDEO_OUTPUT)
        audio_path = os.path.join(os.getcwd(), AUDIO_OUTPUT)

        cur_time = time.time()
        with self.tracer.start_as_current_span("read_video") as read_span:
            try:
                data_filename = os.path.abspath(video_path)
                start_ts = time.time()
                video_tensors = read_video(data_filename, end_pts=10, pts_unit="sec")
                try:
                    video = video_tensors[0].numpy()
                except (IndexError, TypeError, AttributeError) as ie:
                    read_span.record_exception(ie)
                    read_span.set_status(Status(StatusCode.ERROR, "Video stream missing"))
                    raise HTTPException(status_code=500, detail="Failed to extract video stream from processed file")
                duration = time.time() - start_ts
                read_span.set_attribute("read_duration_seconds", duration)
                read_span.set_status(Status(StatusCode.OK))
                print('IP:', time.time() - cur_time)
                return video, video_path, audio_path
            except Exception as e:
                read_span.record_exception(e)
                read_span.set_status(Status(StatusCode.ERROR, "Failed to read video data"))
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

        with self.tracer.start_as_current_span("model_inference") as span:
            span.set_attribute("video_path", v_path)
            span.set_attribute("audio_path", a_path)
            span.set_attribute("device", str(self.device))
            try:
                self.model.eval()
                with self.tracer.start_as_current_span("torch_inference"):
                    with torch.no_grad():
                        y = self.model(X, v_path, a_path, self.device)
                span.set_attribute("prediction_success", True)
                span.set_status(Status(StatusCode.OK))
                return y
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Inference error: {str(e)}"))
                span.set_attribute("prediction_success", False)
                raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    def postprocess(self, y: Any) -> np.ndarray:
        """
        Return inference result.
        :param y: Model output
        :return: result
        """
        with self.tracer.start_as_current_span("postprocess_outputs") as span:
            try:
                if not isinstance(y, (list, np.ndarray)) or len(y) == 0:
                    span.set_status(Status(StatusCode.ERROR, "Invalid model output"))
                    raise HTTPException(status_code=500, detail="Invalid model output")
                result = np.round((np.mean(np.array(y), axis=0)), 1).astype(float)
                try:
                    # Add basic stats
                    result_np = np.array(result)
                    span.set_attribute("result.mean", float(np.mean(result_np)))
                    span.set_attribute("result.min", float(np.min(result_np)))
                    span.set_attribute("result.max", float(np.max(result_np)))
                except Exception:
                    pass
                span.set_status(Status(StatusCode.OK))
                return result
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Post-processing error: {str(e)}"))
                raise HTTPException(status_code=500, detail=f"Post-processing error: {str(e)}")

    def form_response(self, code: Optional[int] = None, result: float = 0.0) -> ResponseModel:
        descriptions = [
            'Successful check',
            'Person is not speaking',
            'No face present',
        ]

        with self.tracer.start_as_current_span("form_response") as span:
            if code != 2:
                result = round((result > 0).mean() * 100, 2)
                code = int(result < 4.0)

            if result < 10:
                score = round((result / 10) * 50, 2)
            else:
                score = round(50 + (result - 10) * 5 / 9, 2)

            try:
                description = descriptions[code]
            except (IndexError, TypeError):
                description = "Unknown response state"
                span.set_status(Status(StatusCode.ERROR, "Invalid response code generated"))
                span.set_attribute("response.code_invalid", True)

            response = ResponseModel(
                code=code,
                description=description,
                result=result,
                score=score
            )

            span.set_attribute("response.code", response.code)
            span.set_attribute("response.description", response.description)
            span.set_attribute("response.score", response.score)
            span.set_attribute("response.result", float(response.result) if isinstance(response.result, (int, float)) else 0.0)
            span.set_status(Status(StatusCode.OK))
            return response