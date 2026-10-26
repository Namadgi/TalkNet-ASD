# TalkNet-ASD
Active Speaker Detection service packaged as a FastAPI application.

## Prerequisites
- Docker with NVIDIA runtime (CUDA 11.x base image) and an NVIDIA GPU (CPU works but is slower).
- Google Cloud service account with permission to write traces/metrics (used by OpenTelemetry) and to read from the target GCS bucket.
- For local runs without Docker: Python 3.8+, `ffmpeg` available on PATH.

## Quickstart (Docker, recommended)
1. Export GCP credentials (any one of the following is enough):
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat path/to/key.json)"
   # or
   export GCP_SA_KEY_B64=$(base64 -w0 path/to/key.json)
   # optional, for telemetry project resolution
   export GOOGLE_CLOUD_PROJECT=<your-project-id>
   ```
2. Build the image:
   ```bash
   docker build -t talknet-asd .
   ```
   You can also run `./build.sh` (expects `gcp.json` in the repo root) for a one-liner build + run.
3. Run the container:
   ```bash
   docker run --rm --gpus all -p 8080:8080 \
     -e GOOGLE_APPLICATION_CREDENTIALS_JSON="$GOOGLE_APPLICATION_CREDENTIALS_JSON" \
     -e GOOGLE_CLOUD_PROJECT="$GOOGLE_CLOUD_PROJECT" \
     talknet-asd
   ```
   The container entrypoint materializes the credential file and starts `uvicorn` on port `8080`.

## Local development (no Docker)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Runtime/telemetry deps installed in the Dockerfile but not pinned in requirements.txt
pip install fastapi uvicorn python-multipart pydantic \
  opentelemetry-api opentelemetry-sdk \
  opentelemetry-exporter-gcp-trace opentelemetry-exporter-gcp-monitoring \
  google-cloud-trace google-cloud-monitoring
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```
Set `GOOGLE_APPLICATION_CREDENTIALS` (or the vars above) and `GOOGLE_CLOUD_PROJECT` before starting so OpenTelemetry exporters can authenticate.

## API
- `POST /upload` — multipart upload of a video file
  ```bash
  curl -X POST http://localhost:8080/upload \
    -F "file=@demo/async.MOV"
  ```
- `POST /predictions/mono` — fetches a video from GCS using the provided OAuth token
  ```bash
  ACCESS_TOKEN=$(gcloud auth print-access-token)
  curl -X POST http://localhost:8080/predictions/mono \
    -H "Content-Type: application/json" \
    -d '{
          "instances": [{
            "token": "'"$ACCESS_TOKEN"'",
            "bucket_name": "your-bucket",
            "object_name": "path/to/video.mp4"
          }]
        }'
  ```

## Response example
```json
{
  "code": 0,
  "description": "Successful check",
  "result": 33.2,
  "score": 65.1
}
```

## Notes
- Model weights for MONO are bundled at `src/model_weights/mono.pth`.
- The Docker build pulls additional face-detection weights from Google Drive via `gdown`.
- Demo videos for quick testing live in the `demo/` directory.
