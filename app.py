from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse

import uvicorn
import logging
import json
import sys
import traceback
import time

class JSONFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Force UTC timezone for formatTime() to justify the "Z" suffix
        self.converter = time.gmtime

    def format(self, record):
        log_record = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
            "service": "ml-asd"
        }
        
        # Safely capture exception text if present
        if record.exc_info:
            log_record["exception"] = "".join(traceback.format_exception(*record.exc_info))

        # Extract structured data from Uvicorn access logs safely
        if record.name == "uvicorn.access" and isinstance(record.args, tuple) and len(record.args) == 5:
            try:
                _, method, path, _, status = record.args
                log_record["msg"] = "Incoming request"
                log_record["method"] = method
                log_record["path"] = path
                log_record["status"] = status
            except Exception:
                # Backup safety step: if unpacking fails for an unexpected reason,
                # we keep the original flat "msg" generated above.
                pass

        return json.dumps(log_record)

# Configure logging at startup
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())

root = logging.getLogger()
root.setLevel(logging.INFO)
root.handlers = [handler]

# Override uvicorn loggers to use our JSON handler
for name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
    l = logging.getLogger(name)
    l.handlers = [handler]
    l.setLevel(logging.INFO)
    l.propagate = False

from src.models import GCSRequest, ResponseModel
from src.services.mono_service import MonoService


app = FastAPI(title="MONO API", description="API for video processing and analysis")
mono_service = MonoService()

# W3C Trace Context propagation
from opentelemetry.propagate import extract

@app.post("/predictions/mono", response_model=ResponseModel)
async def process_video(data: GCSRequest, request: Request):
    """
    Process video from Google Cloud Storage
    """
    parent_context = extract(request.headers)
    with mono_service.tracer.start_as_current_span(
        "mono_request", context=parent_context
    ):
        return await mono_service.process_video(data)

@app.post("/upload", response_model=ResponseModel)
async def upload_video(file: UploadFile = File(...), request: Request = None):
    """
    Process uploaded video file
    """
    parent_context = extract(request.headers) if request else None
    with mono_service.tracer.start_as_current_span(
        "mono_request", context=parent_context
    ):
        return await mono_service.process_video(file)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
