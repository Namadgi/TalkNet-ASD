#!/usr/bin/env bash
# Deploy the current branch (bm-438-ci baseline, or a candidate) as a GPU
# Cloud Run staging service for golden-set evaluation. Mirrors prod ml-asd
# resources. Auth is required (no --allow-unauthenticated); the runner sends a
# gcloud identity token via --auth.
#
# Run from repo root: tools/asd_eval/deploy_golden.sh [SERVICE_NAME] [BASELINE_IMAGE_DIGEST] [MEDIA_MODE]
set -euo pipefail

SERVICE="${1:-ml-asd-golden}"
PROJECT="biometry-416410"
REGION="us-central1"
BASE_IMAGE="${2:-}"
MEDIA_MODE="${3:-}"
# Do not accidentally inherit a previous experiment's mode on later deploys.
MEDIA_ARGS=(--remove-env-vars ASD_MEDIA_MODE)
if [[ -n "$MEDIA_MODE" ]]; then
  case "$MEDIA_MODE" in
    validated|raw|raw_video|raw_audio) MEDIA_ARGS=(--update-env-vars "ASD_MEDIA_MODE=$MEDIA_MODE") ;;
    *) echo "Unknown media mode: $MEDIA_MODE" >&2; exit 1 ;;
  esac
fi

# For candidates, preserve the baseline's exact installed dependencies/codecs.
if [[ -n "$BASE_IMAGE" ]]; then
  IMAGE="us-central1-docker.pkg.dev/$PROJECT/cloud-run-source-deploy/$SERVICE:bm494-$(date -u +%Y%m%d%H%M%S)"
  BUILD_ID=$(gcloud builds submit --project "$PROJECT" --region "$REGION" \
    --config tools/asd_eval/cloudbuild.candidate.yaml \
    --substitutions "_BASE_IMAGE=$BASE_IMAGE,_IMAGE=$IMAGE" --async --format='value(id)')
  while true; do
    STATUS=$(gcloud builds describe "$BUILD_ID" --project "$PROJECT" --region "$REGION" \
      --format='value(status)')
    case "$STATUS" in
      SUCCESS) break ;;
      QUEUED|PENDING|WORKING) sleep 10 ;;
      *) echo "Build $BUILD_ID failed: $STATUS" >&2; exit 1 ;;
    esac
  done
  DIGEST=$(gcloud builds describe "$BUILD_ID" --project "$PROJECT" --region "$REGION" \
    --format='value(results.images[0].digest)')
  gcloud run deploy "$SERVICE" --image "${IMAGE%:*}@$DIGEST" \
    --project "$PROJECT" --region "$REGION" "${MEDIA_ARGS[@]}"
  exit
fi

echo ">> deploying '$SERVICE' from source (branch: $(git branch --show-current))"

gcloud run deploy "$SERVICE" \
  --source . \
  --project "$PROJECT" \
  --region "$REGION" \
  "${MEDIA_ARGS[@]}" \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --cpu 4 \
  --memory 16Gi \
  --no-cpu-throttling \
  --max-instances 1 \
  --no-allow-unauthenticated \
  --port 8080

echo
echo ">> URL:"
gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --format="value(status.url)"
