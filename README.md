# Nova Shelf AI

Nova Shelf AI is a FastAPI demo app for retail shelf auditing.

It lets a user:

- upload a shelf image
- send it to Amazon Bedrock Nova 2 Lite
- detect visible products and shelf issues
- generate restock / rearrange / audit tasks
- allocate those tasks to employees on shift from a local rota
- return a text instruction for the store associate

If Bedrock is not available, the app can still run in a local demo fallback mode.

For hackathon submission guidance, see [HACKATHON_SUBMISSION_CHECKLIST.md](C:/CODEX/HACKATHON_SUBMISSION_CHECKLIST.md).

## What your friend needs

Install these first:

- Python 3.11+
- Git
- AWS CLI

Optional but recommended:

- VS Code

## 1. Clone the repo

```bash
git clone https://github.com/AthulJayakumar/Nova-Shelf-AI.git
cd Nova-Shelf-AI
```

## 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure AWS credentials

If they want real Nova 2 Lite product detection, they must configure AWS:

```bash
aws configure
```

Use:

- Region: `us-east-1`
- Access key: your AWS access key
- Secret key: your AWS secret key

Their AWS user or role must also have:

- Bedrock Runtime access
- model access enabled for Nova 2 Lite in Amazon Bedrock

## 5. Create the `.env` file

Create a file named `.env` in the project root with:

```env
AWS_REGION=us-east-1
BEDROCK_ENABLED=true
NOVA_LITE_MODEL_ID=us.amazon.nova-2-lite-v1:0
NOVA_SONIC_ENABLED=false
NOVA_SONIC_MODEL_ID=us.amazon.nova-2-sonic-v1:0
DEFAULT_PLANOGRAM_PRODUCT=Brand X Cereal
```

If they only want to see the app without Bedrock, they can set:

```env
BEDROCK_ENABLED=false
```

## 6. Run the server

```bash
uvicorn backend.main:app --reload
```

Then open:

- App UI: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Bedrock check: [http://127.0.0.1:8000/bedrock-status](http://127.0.0.1:8000/bedrock-status)

## 7. How to use the app

1. Open the web UI.
2. Upload a shelf image.
3. Set a location hint if needed.
4. Choose an analysis mode:

- `auto`: use Bedrock if available, otherwise local fallback
- `demo`: always use local fallback
- `bedrock`: require Bedrock and fail if Bedrock is not configured

5. Run the audit.

The app will return:

- detected products
- shelf issues
- generated tasks
- task assignees chosen from the active rota
- a final instruction message

## Important endpoints

- `GET /health`
- `GET /bedrock-status`
- `GET /bedrock-debug`
- `GET /inventory`
- `GET /rota`
- `GET /active-staff`
- `GET /tasks`
- `POST /audit-shelf`

## Rota-based assignment

The app uses [data/rota.json](C:/CODEX/data/rota.json) to decide who is on shift.

Each employee has:

- an `employee_id`
- a role
- task skills like `RESTOCK`, `REARRANGE`, or `AUDIT`
- optional zone coverage
- weekday shift windows

When a task is created, the app:

1. finds active staff for the current time
2. filters by skill
3. prefers zone matches
4. assigns the task to one active employee

## Troubleshooting

### `/bedrock-status` says `BEDROCK_ENABLED is false`

That means `.env` is missing or the server was started before `.env` was created.

Fix:

1. create `.env`
2. restart `uvicorn`

### Bedrock is enabled but not reachable

Check:

- `aws configure` was run for the same user
- region is `us-east-1`
- model ID is `us.amazon.nova-2-lite-v1:0`
- Bedrock model access is enabled in AWS
- IAM permissions allow Bedrock Runtime

### Shelf audit fails with a Bedrock parsing error

The app saves the latest raw model output here:

- `debug/latest_bedrock_response.txt`
- `debug/latest_bedrock_error.json`

These files help debug malformed or truncated Nova responses.

### The app runs but does not identify real products

That usually means it is running in fallback mode instead of Bedrock mode.

Check:

- `/bedrock-status`
- the selected `analysis_mode`

## Project structure

```text
backend/
agents/
database/
services/
utils/
frontend/
data/
```

## Notes

- `.env` is not committed to GitHub on purpose.
- `store.db` is local and not committed.
- `debug/` is local and not committed.

## Quick start for demo only

If your friend just wants the UI to run without AWS:

1. install Python
2. create venv
3. install requirements
4. set `BEDROCK_ENABLED=false` in `.env`
5. run `uvicorn backend.main:app --reload`

That will use the local fallback instead of Nova.
