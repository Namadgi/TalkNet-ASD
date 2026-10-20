#!/usr/bin/env bash
# Deploy the current branch (bm-438-ci baseline, or a candidate) as a GPU
# Cloud Run staging service for golden-set evaluation. Mirrors prod ml-asd
# resources. Auth is required (no --allow-unauthenticated); the runner sends a
# gcloud identity token via --auth.
#
# Run from repo root:  tools/asd_eval/deploy_golden.sh [SERVICE_NAME]
set -euo pipefail

SERVICE="${1:-ml-asd-golden}"
PROJECT="biometry-416410"
REGION="us-central1"

echo ">> deploying '$SERVICE' from source (branch: $(git branch --show-current))"

gcloud run deploy "$SERVICE" \
  --source . \
  --project "$PROJECT" \
  --region "$REGION" \
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
