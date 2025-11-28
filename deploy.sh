
export PROJECT_ID=project2-477608
export SERVICE_NAME=ai-agent-service
export REGION=us-central1
export APP_SECRET="9fed69bd31317e42ac5c527579e6f0b91b7ae62d"
export GEMINI_API_KEY="AIzaSyDCxk065P89AArcUsZeye2A9D8P2H04MVA"

gcloud run deploy $SERVICE_NAME \
  --source . \
  --project $PROJECT_ID \
  --region $REGION \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 600 \
  --clear-base-image \
  --set-env-vars="APP_SECRET=${APP_SECRET},GEMINI_API_KEY=${GEMINI_API_KEY}"
