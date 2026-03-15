from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from agents.instruction_agent import generate_instruction
from backend.config import get_settings
from database.db import init_db
from database.models import ScanShelfResponse, ShelfAuditSummary
from services.intelligence_hub import build_tasks
from services.inventory_service import list_inventory
from services.task_engine import list_tasks
from services.vision_service import analyze_shelf, get_latest_bedrock_debug, test_bedrock_connection
from services.voice_service import generate_voice


app = FastAPI(
    title="Retail Shelf Restock MVP",
    version="0.5.0",
    summary="Shelf audit, product detection, and task generation for any aisle image.",
)

FRONTEND_PATH = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/", include_in_schema=False)
def landing_page() -> FileResponse:
    return FileResponse(FRONTEND_PATH)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/bedrock-status")
def bedrock_status() -> dict[str, object]:
    settings = get_settings()
    payload: dict[str, object] = {
        "bedrock_enabled": settings.bedrock_enabled,
        "aws_region": settings.aws_region,
        "nova_lite_model_id": settings.nova_lite_model_id,
    }

    if not settings.bedrock_enabled:
        payload["reachable"] = False
        payload["detail"] = "BEDROCK_ENABLED is false."
        return payload

    try:
        check = test_bedrock_connection()
        payload["reachable"] = True
        payload["reply"] = check["reply"]
        return payload
    except RuntimeError as exc:
        payload["reachable"] = False
        payload["detail"] = str(exc)
        return payload


@app.get("/bedrock-debug")
def bedrock_debug() -> dict[str, str | bool | None]:
    return get_latest_bedrock_debug()


@app.get("/inventory")
def inventory() -> list[dict[str, int | str]]:
    return list_inventory()


@app.get("/tasks")
def task_list() -> list[dict]:
    return [task.model_dump() for task in list_tasks()]


@app.post("/scan-shelf", response_model=ScanShelfResponse)
@app.post("/audit-shelf", response_model=ScanShelfResponse)
async def scan_shelf(
    file: UploadFile = File(...),
    location_hint: str | None = Form(default=None),
    analysis_mode: str = Form(default="auto"),
) -> ScanShelfResponse:
    image_bytes = await file.read()

    try:
        audit = analyze_shelf(
            image_bytes=image_bytes,
            location_hint=location_hint,
            analysis_mode=analysis_mode,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    tasks = build_tasks(audit.issues, audit.location)
    instruction = generate_instruction(tasks, audit)
    voice = generate_voice(instruction)

    summary = ShelfAuditSummary(
        detected_product_count=len(audit.detected_products),
        issue_count=len(audit.issues),
        restock_task_count=sum(1 for task in tasks if task.task_type == "RESTOCK"),
        rearrange_task_count=sum(1 for task in tasks if task.task_type == "REARRANGE"),
        audit_task_count=sum(1 for task in tasks if task.task_type == "AUDIT"),
    )

    return ScanShelfResponse(
        provider=audit.provider,
        location=audit.location,
        overview=audit.overview,
        confidence=audit.confidence,
        summary=summary,
        detected_products=audit.detected_products,
        issues=audit.issues,
        tasks=tasks,
        instruction=instruction,
        voice=voice,
    )
