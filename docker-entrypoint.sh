#!/bin/sh
set -e

# Materialize GCP service account credentials from environment variables
CRED_PATH="/tmp/gcp.json"

if [ -n "$GCP_SA_KEY_B64" ]; then
  echo "$GCP_SA_KEY_B64" | base64 -d > "$CRED_PATH"
elif [ -n "$GCP_SA_KEY" ]; then
  printf "%s" "$GCP_SA_KEY" > "$CRED_PATH"
elif [ -n "$GOOGLE_APPLICATION_CREDENTIALS_JSON" ]; then
  printf "%s" "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > "$CRED_PATH"
fi

if [ -f "$CRED_PATH" ]; then
  export GOOGLE_APPLICATION_CREDENTIALS="$CRED_PATH"
fi

# Allow setting project via PROJECT_ID if GOOGLE_CLOUD_PROJECT not set
if [ -z "$GOOGLE_CLOUD_PROJECT" ] && [ -n "$PROJECT_ID" ]; then
  export GOOGLE_CLOUD_PROJECT="$PROJECT_ID"
fi

exec "$@"


