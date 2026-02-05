# Deploying LatticeVAD API (Cloud Run / Render / Docker)

This repository contains a FastAPI server at `/api/voice-detection` that accepts a JSON body with the following fields:

- `language` (string) - one of: Tamil, English, Hindi, Malayalam, Telugu
- `audioFormat` (string) - must be `mp3`
- `audioBase64` (string) - base64-encoded audio data

The API expects an `x-api-key` header. Valid keys are loaded from the `VALID_API_KEYS` environment variable (comma-separated). If not set, default test keys are used.

The project includes a `Dockerfile` so you can deploy the API publicly using Cloud Run, Render, Railway, or any container platform.

## Environment variables

- `VALID_API_KEYS` (optional): comma-separated API keys. Example: `sk_test_123,prod_key_abc`
- `CORS_ORIGINS` (optional): comma-separated allowed origins. Default `*` (all origins). To restrict, set e.g. `https://example.com`

## Quick local test (Docker)

Build the image:

```bash
docker build -t latticevad-api:latest .
```

Run locally exposing port 8080:

```bash
docker run -p 8080:8080 -e VALID_API_KEYS="sk_test_123456789" latticevad-api:latest
```

Then test:

```bash
curl -X POST http://localhost:8080/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_123456789" \
  -d '{"language":"English","audioFormat":"mp3","audioBase64":"<BASE64>"}'
```

## Deploy to Google Cloud Run (example)

1. Build and push an image (gcloud):

```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/latticevad-api
```

2. Deploy to Cloud Run:

```bash
gcloud run deploy latticevad-api \
  --image gcr.io/PROJECT-ID/latticevad-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "VALID_API_KEYS=sk_test_123456789,CUSTOM_KEY" \
  --set-env-vars "CORS_ORIGINS=*"
```

Replace `PROJECT-ID` with your Google Cloud project.

## Deploy to Render (example)

1. Create a new Web Service on Render and connect to this repo.
2. Use the Dockerfile-based service. Set the environment variables in the Render dashboard (`VALID_API_KEYS`, `CORS_ORIGINS`).

## Notes on security

- For public deployments, set `VALID_API_KEYS` to strong, secret values.
- Restrict `CORS_ORIGINS` to the domain(s) that will host your endpoint tester UI.
- Consider adding rate limiting and request size limits for production.

## GitHub Actions (automatic deploy)

You can use the provided GitHub Actions workflow at `.github/workflows/cloud-run-deploy.yml` to automatically build and deploy on push to `main`.

Required GitHub repository secrets:
- `GCP_SA_KEY` — JSON contents of a Google Cloud service account key with the roles `Cloud Run Admin`, `Cloud Build Editor`, and `Storage Admin` (for Cloud Build pushes).
- `GCP_PROJECT` — your Google Cloud project ID.
- `VALID_API_KEYS` — comma-separated API keys for the service.
- `CORS_ORIGINS` — optional, comma-separated origins.

Once secrets are set, push to `main` and the workflow will run, building the image with Cloud Build and deploying to Cloud Run.

## GitHub Actions (Render deploy)

An alternative GitHub Actions workflow is provided at `.github/workflows/render-deploy.yml` for deploying to Render instead of Cloud Run.

Required GitHub repository secrets:
- `RENDER_SERVICE_ID` — your Render service ID (found in service dashboard URL).
- `RENDER_API_KEY` — your Render API key (generate in account settings).

Once secrets are set, push to `main` and the workflow will trigger a Render deployment.

