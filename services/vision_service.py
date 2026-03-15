from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from backend.config import get_bedrock_client, get_settings
from database.models import DetectedProduct, ShelfIssue, VisionShelfAudit
from services.planogram_service import DEFAULT_LOCATION
from utils.image_utils import preprocess_image, summarize_image


DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"
LATEST_BEDROCK_RESPONSE_PATH = DEBUG_DIR / "latest_bedrock_response.txt"
LATEST_BEDROCK_ERROR_PATH = DEBUG_DIR / "latest_bedrock_error.json"

VISION_PROMPT = """
Analyze the full retail shelf photo from any aisle in a store.

Your job is to act like a shelf-audit agent. Inspect the image and:
- identify as many visible products as you can by product name or best-effort brand + descriptor
- estimate how well each product block is stocked and front-faced
- detect operational issues such as low stock, empty spaces, messy arrangement, mixed products, blocked access, label gaps, or likely planogram mismatch
- recommend what the store associate should do next

Return JSON only with this exact schema:
{
  "provider": "string",
  "location": "string",
  "overview": "string",
  "confidence": 0.0,
  "detected_products": [
    {
      "product_name": "string",
      "brand": "string or null",
      "category": "string or null",
      "shelf_level": "top shelf | upper shelf | middle shelf | lower shelf | bottom shelf | unknown shelf",
      "estimated_units": 0,
      "facings": 0,
      "condition": "HEALTHY | LOW_STOCK | MESSY | MIXED | UNCERTAIN",
      "confidence": 0.0,
      "notes": "string or null"
    }
  ],
  "issues": [
    {
      "issue_type": "LOW_STOCK | EMPTY_SPACE | MISALIGNED | MIXED_PRODUCTS | UNKNOWN_PRODUCT | PLANOGRAM_MISMATCH | LABEL_GAP | OBSTRUCTION",
      "severity": "LOW | MEDIUM | HIGH",
      "product_name": "string or null",
      "shelf_level": "top shelf | upper shelf | middle shelf | lower shelf | bottom shelf | unknown shelf",
      "details": "string",
      "gap_units": 0,
      "confidence": 0.0,
      "suggested_action": "RESTOCK | REARRANGE | AUDIT"
    }
  ]
}

Rules:
- Return every distinct visible product block you can identify with reasonable confidence.
- If the exact product name is unclear, use the most specific best-effort name you can infer from the package.
- Keep `notes` short.
- Use RESTOCK when the shelf visually needs more units.
- Use REARRANGE for messy fronting, mixed products, or presentation issues.
- Use AUDIT when identity is unclear, an obstruction prevents inspection, or planogram mismatch needs a person to verify.
- Do not wrap the JSON in markdown.
""".strip()

COMPACT_VISION_PROMPT = """
Analyze this retail shelf photo and return STRICT valid minified JSON only.

Use the same schema as before, but keep it compact:
- maximum 12 detected_products
- maximum 8 issues
- product_name should be short
- notes should be null unless essential
- details should be one short sentence
- no markdown
- no explanation outside JSON
""".strip()


def analyze_shelf(
    image_bytes: bytes,
    location_hint: str | None = None,
    analysis_mode: str = "auto",
) -> VisionShelfAudit:
    image, normalized_bytes, image_format = preprocess_image(image_bytes)
    stats = summarize_image(image)

    client = get_bedrock_client()
    requested_bedrock = analysis_mode == "bedrock"
    should_use_bedrock = analysis_mode in {"auto", "bedrock"} and client is not None

    if requested_bedrock and client is None:
        raise RuntimeError(
            "Bedrock mode was requested, but BEDROCK_ENABLED is false or AWS credentials are unavailable. "
            "Switch to auto/demo mode or configure Bedrock first."
        )

    if should_use_bedrock:
        try:
            return _analyze_with_bedrock(
                client=client,
                normalized_bytes=normalized_bytes,
                image_format=image_format,
                location_hint=location_hint or DEFAULT_LOCATION,
            )
        except Exception as exc:
            if requested_bedrock:
                raise RuntimeError(_friendly_bedrock_error(exc)) from exc

    return _build_generic_demo_audit(location_hint=location_hint or DEFAULT_LOCATION, stats=stats)


def test_bedrock_connection() -> dict[str, str]:
    client = get_bedrock_client()
    if client is None:
        raise RuntimeError("BEDROCK_ENABLED is false. Set it to true and provide AWS credentials first.")

    try:
        response = client.converse(
            modelId=get_settings().nova_lite_model_id,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": "Reply with exactly OK."}],
                }
            ],
            inferenceConfig={"temperature": 0, "maxTokens": 10},
        )
    except Exception as exc:
        raise RuntimeError(_friendly_bedrock_error(exc)) from exc

    content = response["output"]["message"]["content"]
    text = next((item["text"] for item in content if "text" in item), "")
    return {
        "model_id": get_settings().nova_lite_model_id,
        "reply": text.strip(),
    }


def get_latest_bedrock_debug() -> dict[str, str | bool | None]:
    return {
        "response_exists": LATEST_BEDROCK_RESPONSE_PATH.exists(),
        "response_path": str(LATEST_BEDROCK_RESPONSE_PATH),
        "error_exists": LATEST_BEDROCK_ERROR_PATH.exists(),
        "error_path": str(LATEST_BEDROCK_ERROR_PATH),
        "response_preview": _safe_preview(LATEST_BEDROCK_RESPONSE_PATH),
        "error_preview": _safe_preview(LATEST_BEDROCK_ERROR_PATH),
    }


def _analyze_with_bedrock(
    *,
    client: Any,
    normalized_bytes: bytes,
    image_format: str,
    location_hint: str,
) -> VisionShelfAudit:
    prompts = [
        (VISION_PROMPT, 2200, "full"),
        (f"{VISION_PROMPT}\n\n{COMPACT_VISION_PROMPT}", 1400, "compact"),
    ]

    last_error: Exception | None = None
    for prompt, max_tokens, mode in prompts:
        response_text = ""
        try:
            response_text = _call_bedrock_audit(
                client=client,
                normalized_bytes=normalized_bytes,
                image_format=image_format,
                location_hint=location_hint,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            _write_latest_response(response_text)
            parsed = _extract_json(response_text)
            return _coerce_audit(parsed, fallback_location=location_hint, fallback_provider="bedrock-nova-2-lite")
        except Exception as exc:
            last_error = exc
            _write_latest_error(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "mode": mode,
                    "model_id": get_settings().nova_lite_model_id,
                    "location_hint": location_hint,
                    "max_tokens": max_tokens,
                    "error": str(exc),
                    "response_preview": response_text[:2000],
                }
            )

    raise ValueError(
        "Unable to parse Nova response after retry. "
        f"Latest debug saved to {LATEST_BEDROCK_RESPONSE_PATH} and {LATEST_BEDROCK_ERROR_PATH}. "
        f"Last error: {last_error}"
    )


def _call_bedrock_audit(
    *,
    client: Any,
    normalized_bytes: bytes,
    image_format: str,
    location_hint: str,
    prompt: str,
    max_tokens: int,
) -> str:
    response = client.converse(
        modelId=get_settings().nova_lite_model_id,
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": f"{prompt}\nLocation hint: {location_hint}"},
                    {
                        "image": {
                            "format": image_format,
                            "source": {"bytes": normalized_bytes},
                        }
                    },
                ],
            }
        ],
        inferenceConfig={
            "temperature": 0,
            "maxTokens": max_tokens,
        },
    )

    content = response["output"]["message"]["content"]
    text_chunks = [item["text"] for item in content if "text" in item]
    return "\n".join(text_chunks)


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fences(text).strip()
    candidates = _extract_balanced_json_candidates(cleaned)

    if not candidates and cleaned.startswith("{") and cleaned.endswith("}"):
        candidates = [cleaned]

    last_error: Exception | None = None
    for candidate in candidates:
        for attempt in (candidate, _repair_json(candidate)):
            if not attempt.strip():
                continue
            try:
                return json.loads(attempt)
            except json.JSONDecodeError as exc:
                last_error = exc

    preview = cleaned[:600].replace("\n", " ")
    if _looks_truncated(cleaned):
        raise ValueError(
            "Model output appears truncated before the JSON completed. "
            f"Response preview: {preview}"
        )

    if last_error is not None:
        raise ValueError(
            f"Model returned malformed JSON after repair attempts: {last_error}. Response preview: {preview}"
        )

    raise ValueError(f"No JSON payload returned by vision model. Response preview: {preview}")


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped


def _extract_balanced_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    depth = 0
    start_index: int | None = None
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
            continue

        if char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start_index is not None:
                candidates.append(text[start_index : index + 1])
                start_index = None

    if candidates:
        candidates.sort(key=len, reverse=True)
    return candidates


def _repair_json(candidate: str) -> str:
    repaired = candidate
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    return repaired


def _looks_truncated(text: str) -> bool:
    return text.count("{") > text.count("}") or text.count("[") > text.count("]")


def _write_latest_response(text: str) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_BEDROCK_RESPONSE_PATH.write_text(text, encoding="utf-8")



def _write_latest_error(payload: dict[str, Any]) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_BEDROCK_ERROR_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")



def _safe_preview(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")[:800]



def _coerce_audit(payload: dict[str, Any], *, fallback_location: str, fallback_provider: str) -> VisionShelfAudit:
    detected_products = [DetectedProduct(**item) for item in payload.get("detected_products", [])]
    issues = [ShelfIssue(**item) for item in payload.get("issues", [])]

    return VisionShelfAudit(
        provider=payload.get("provider", fallback_provider),
        location=payload.get("location", fallback_location),
        overview=payload.get("overview", "Shelf audit completed."),
        confidence=float(payload.get("confidence", 0.6)),
        detected_products=detected_products,
        issues=issues,
    )


def _friendly_bedrock_error(exc: Exception) -> str:
    if isinstance(exc, ClientError):
        error = exc.response.get("Error", {})
        code = error.get("Code", "UnknownError")
        message = error.get("Message", str(exc))

        if code in {"AccessDeniedException", "UnrecognizedClientException"}:
            return (
                "AWS rejected the Bedrock request. Check your AWS credentials, IAM permissions, and model access in the Bedrock console. "
                f"Original error: {message}"
            )

        if code in {"ValidationException", "ResourceNotFoundException"}:
            return (
                f"Bedrock could not use model ID '{get_settings().nova_lite_model_id}'. "
                f"Check the model ID and region. Original error: {message}"
            )

        if code in {"ThrottlingException", "ServiceQuotaExceededException"}:
            return f"Bedrock quota or throttling blocked the request. Original error: {message}"

        return f"Bedrock request failed with {code}: {message}"

    if isinstance(exc, BotoCoreError):
        return f"Bedrock SDK error: {exc}"

    return f"Bedrock request failed: {exc}"


def _build_generic_demo_audit(*, location_hint: str, stats: dict[str, float]) -> VisionShelfAudit:
    brightness = stats["brightness"]
    edge_density = stats["edge_density"]
    confidence = 0.42 if edge_density > 0.04 else 0.35

    overview = (
        "Demo fallback completed. The image appears to show a populated retail bay, but exact product recognition "
        "for any aisle requires Bedrock vision to be enabled. The system is returning generic shelf-health tasks so the workflow remains usable."
    )

    if brightness < 110:
        overview = (
            "Demo fallback completed on a low-light image. Product identification is limited without Bedrock, "
            "but the workflow can still surface likely tidying and restock checks."
        )

    detected_products = [
        DetectedProduct(
            product_name="Unidentified product block A",
            brand=None,
            category="Unknown retail category",
            shelf_level="middle shelf",
            estimated_units=4,
            facings=3,
            condition="UNCERTAIN",
            confidence=0.31,
            notes="Generic fallback result. Enable Bedrock for aisle-specific product names.",
        ),
        DetectedProduct(
            product_name="Unidentified product block B",
            brand=None,
            category="Unknown retail category",
            shelf_level="lower shelf",
            estimated_units=2,
            facings=2,
            condition="MESSY",
            confidence=0.28,
            notes="Front-facing may need attention, but exact SKU detection is unavailable in demo mode.",
        ),
    ]

    issues = [
        ShelfIssue(
            issue_type="LOW_STOCK",
            severity="MEDIUM",
            product_name="Unidentified product block A",
            shelf_level="middle shelf",
            details="This block appears partially depleted and should be checked for restocking.",
            gap_units=2,
            confidence=0.34,
            suggested_action="RESTOCK",
        ),
        ShelfIssue(
            issue_type="MISALIGNED",
            severity="MEDIUM",
            product_name="Unidentified product block B",
            shelf_level="lower shelf",
            details="Visible front-facing looks uneven and should be rearranged.",
            gap_units=0,
            confidence=0.32,
            suggested_action="REARRANGE",
        ),
        ShelfIssue(
            issue_type="PLANOGRAM_MISMATCH",
            severity="LOW",
            product_name=None,
            shelf_level="unknown shelf",
            details="A human check is recommended to confirm the correct assortment for this bay.",
            gap_units=0,
            confidence=0.26,
            suggested_action="AUDIT",
        ),
    ]

    return VisionShelfAudit(
        provider="demo-fallback",
        location=location_hint,
        overview=overview,
        confidence=confidence,
        detected_products=detected_products,
        issues=issues,
    )
