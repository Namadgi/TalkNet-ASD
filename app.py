from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse

import uvicorn
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
    uvicorn.run(app, host="0.0.0.0", port=8000) 