# Retail Shelf Audit MVP

This project is a demo-ready shelf audit app for hackathons and store-ops prototypes.

## What it does

- Accepts a shelf image from any aisle
- Uses Amazon Nova 2 Lite on Bedrock when configured, with a generic demo fallback when it is not
- Identifies as many visible products as the model can infer from the image
- Flags low stock, empty space, messy fronting, mixed products, label gaps, and audit-worthy issues
- Generates restock, rearrange, or audit tasks automatically from the model response
- Returns a voice-ready instruction summary

## Run locally

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000` for the demo UI.

## Main endpoint

`POST /audit-shelf`

Multipart form fields:

- `file`: required shelf image
- `location_hint`: optional aisle or bay name
- `analysis_mode`: `auto`, `demo`, or `bedrock`

## Bedrock setup

Set these values before expecting real product detection:

- `BEDROCK_ENABLED=true`
- `AWS_REGION=us-east-1`
- `NOVA_LITE_MODEL_ID=us.amazon.nova-2-lite-v1:0`

Then make sure:

- your AWS credentials are available to `boto3`
- your IAM user or role can call Bedrock Runtime
- model access for Nova 2 Lite is enabled in the Bedrock console

Use these endpoints while debugging:

- `GET /bedrock-status`
- `GET /bedrock-debug`

If Nova returns broken or truncated JSON, the app now saves the latest raw model output to `debug/latest_bedrock_response.txt` and the matching parse error to `debug/latest_bedrock_error.json`.

## Modes

- `auto`: use Bedrock if enabled, otherwise fall back to the local generic demo audit
- `demo`: always use the local fallback
- `bedrock`: require Bedrock and return a clear error if it is not configured or accessible
