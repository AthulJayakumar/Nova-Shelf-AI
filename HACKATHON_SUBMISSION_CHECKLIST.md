# Amazon Nova Hackathon Submission Checklist

This file is a practical checklist for submitting `Nova Shelf AI` to the Amazon Nova AI Hackathon.

It is based on the official rules on Devpost:

- [Hackathon overview](https://amazon-nova.devpost.com/)
- [Official rules](https://amazon-nova.devpost.com/rules)
- [Schedule](https://amazon-nova.devpost.com/details/dates)

Important dates:

- Submission deadline: March 16, 2026 at 5:00 PM PDT
- Feedback deadline: March 18, 2026 at 5:00 PM PDT
- AWS promotional credit request deadline: March 13, 2026 at 5:00 PM PDT

## Best category for this project

Recommended category:

- `Multimodal Understanding`

Why:

- the core feature is analyzing shelf images with Amazon Nova 2 Lite
- the system identifies products, shelf issues, and operational tasks from images

Do not position this project as `Voice AI` unless real Nova 2 Sonic voice interaction is added.

## Rule fit: current repo status

### Already aligned

- Uses an Amazon Nova foundation model:
  - [services/vision_service.py](C:/CODEX/services/vision_service.py)
  - [backend/config.py](C:/CODEX/backend/config.py)
- Is a generative AI / multimodal application built around Amazon Nova
- Includes a working app and API:
  - [backend/main.py](C:/CODEX/backend/main.py)
- Includes a public code repository structure and English documentation:
  - [README.md](C:/CODEX/README.md)
- Includes a runnable UI demo:
  - [frontend/index.html](C:/CODEX/frontend/index.html)
- Includes original application logic for shelf audit, issue detection, rota-based task assignment, and task generation

### Partially aligned

- Functioning demo access:
  - local demo exists, but you still need to decide what judges will use
  - safest options:
    - deploy a public demo
    - or provide clear local run/test steps and credentials if needed
- Technical implementation story:
  - code is good enough for demo
  - architecture explanation should be tightened in your Devpost description and demo video

### Still required before submission

- Devpost text description
- 3-minute public demo video
- submission category selection
- testing instructions for judges
- private repo access sharing only if the repo becomes private
- optional feedback submission if you want feedback prize eligibility
- optional builder.aws.com blog post if you want the bonus blog prize

## Submission-safe claims

You can safely claim:

- Amazon Nova 2 Lite is used as the core multimodal model
- The app analyzes shelf photos and produces structured shelf audits
- The app generates operational tasks such as restock, rearrange, and audit
- The app assigns tasks to employees on shift using a local rota
- The app includes fallback behavior when Bedrock is unavailable

Avoid claiming unless you implement it first:

- real-time voice AI with Nova 2 Sonic
- full production-grade planogram matching for every aisle
- fully autonomous multi-agent execution
- deployed enterprise-ready platform

## Required submission package

Before you submit on Devpost, make sure you have:

- Project title
- One-paragraph summary
- Category selected: `Multimodal Understanding`
- Clear explanation of how Amazon Nova is used
- Public demo video link
- Working test instructions
- Repo link

## Suggested Devpost positioning

Suggested one-line pitch:

`Nova Shelf AI helps retail teams turn shelf photos into product audits, gap detection, and rota-assigned restocking tasks using Amazon Nova 2 Lite.`

Suggested focus points for judging:

- Technical Implementation:
  - Nova 2 Lite image analysis
  - structured JSON shelf audit
  - FastAPI backend
  - task generation and rota assignment
- Enterprise or Community Impact:
  - reduces manual shelf checks
  - speeds replenishment
  - improves on-shelf availability
- Creativity and Innovation:
  - combines multimodal shelf audit with operational task assignment

## Demo video checklist

- Keep it under 3 minutes
- Show the app actually running
- Show image upload
- Show Bedrock/Nova analysis result
- Show generated tasks
- Show assigned employee from rota
- Mention `Amazon Nova 2 Lite`
- Add `#AmazonNova`
- Do not use copyrighted music unless you have permission

## Testing checklist for judges

- App starts with:

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

- Main app URL:

```text
http://127.0.0.1:8000
```

- Health check:

```text
http://127.0.0.1:8000/health
```

- Bedrock status:

```text
http://127.0.0.1:8000/bedrock-status
```

If you do not deploy publicly, include these exact instructions in the Devpost testing notes.

## Repo hygiene check

Already good:

- `.env` is ignored
- local DB is ignored
- debug artifacts are ignored

Double-check before submission:

- do not commit secrets
- do not include third-party copyrighted assets without permission
- do not include any proprietary client/store data you are not allowed to share

## Final recommendation

This repo is suitable for submission to the hackathon in its current direction, with one important caveat:

- the codebase is aligned enough for `Multimodal Understanding`
- the submission package is not complete until you add the demo video, Devpost description, and judge testing instructions

This is the safest framing:

- `working multimodal retail shelf audit MVP built with Amazon Nova 2 Lite`

This is the unsafe framing:

- `fully production-ready shelf operating system`
