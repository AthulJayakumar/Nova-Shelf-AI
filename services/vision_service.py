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
from utils.image_utils import create_row_crop_images, create_row_crops, detect_gap_segments, preprocess_image, summarize_image


DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"
LATEST_BEDROCK_RESPONSE_PATH = DEBUG_DIR / "latest_bedrock_response.txt"
LATEST_BEDROCK_ERROR_PATH = DEBUG_DIR / "latest_bedrock_error.json"

VISION_PROMPT = """
You are a retail shelf auditing model for supermarket operations.

Goal:
Analyze one real store shelf image and produce a strict JSON shelf audit that can be used to generate replenishment, recovery, and audit tasks.

Core mindset:
Be conservative and operational.
Do not assume shelves are perfect.
If you can see open space, broken front edges, messy alignment, mixed flavors, or a disrupted product block, treat that as a likely issue unless the image clearly proves otherwise.

How to inspect the image:
1. Detect shelf rows from top to bottom.
2. For each shelf row, identify contiguous product zones.
3. Group adjacent identical front facings into one product zone.
4. Count only visible front facings.
5. Estimate expected facings from shelf width, neighboring product widths, and visible shelf-edge labels.
6. Detect empty slots, under-filled blocks, messy fronting, misplaced products, and missing price labels.

Important retail rules:
- Use visible whitespace, exposed backing, or obvious empty span between adjacent product blocks as evidence of a gap.
- If a product block is not fully front-faced to the shelf edge, consider MESSY or LOW_STOCK.
- If facings are broken by a visible empty span larger than about half one product width, count a gap.
- If the same product appears in multiple adjacent facings, combine them into one product entry for that contiguous block.
- Do not repeat the same contiguous product block multiple times.
- Do not mark a shelf as perfectly full unless there are no visible gaps and no obvious fronting problems.
- If uncertain between IN_STOCK and LOW_STOCK, prefer LOW_STOCK when visible spacing suggests missing facings.
- If uncertain between IN_STOCK and MESSY, prefer MESSY when presentation is visibly uneven.
- Only use OUT_OF_STOCK when a product zone is clearly intended but no units are visible.
- Only identify products that are visibly present; do not invent hidden stock behind the front row.

Product identification guidance:
- Read the package front and shelf label together when possible.
- Prefer brand + variant or flavor, not just the parent brand.
- Do not output generic one-word names if a more specific visible name is readable.
- For multipacks, include the multipack type if visible, such as 6 pack, zero sugar, cheese and onion, or ready salted.
- If text is partially visible, return the most specific honest best-effort name instead of a made-up full SKU.
- If confidence is weak, lower confidence instead of forcing an exact SKU.

Output requirements:
- Return ONLY one valid JSON object.
- No markdown.
- No explanation.
- No preamble.
- No code fences.

Use this schema exactly:
{
  "shelf_summary": {
    "total_shelves_detected": <integer>,
    "total_products_identified": <integer>,
    "total_gaps_detected": <integer>,
    "overall_fill_rate_pct": <0-100 float>
  },
  "shelves": [
    {
      "shelf_number": <integer, 1=top>,
      "shelf_label": "<Top Shelf | Upper Shelf | Eye Level | Lower Shelf | Bottom Shelf | Unknown Shelf>",
      "products": [
        {
          "product_name": "<brand + variant if visible, otherwise most specific best-effort visible name>",
          "facings_visible": <integer>,
          "facings_expected": <integer>,
          "gap_count": <integer>,
          "price_label": "<price string or null>",
          "status": "<IN_STOCK | LOW_STOCK | OUT_OF_STOCK | MISPLACED | MESSY | UNKNOWN>",
          "confidence": <0.0-1.0>,
          "notes": "<short factual anomaly note or null>"
        }
      ],
      "shelf_issues": [
        {
          "issue_type": "<GAP | MESSY | MISPLACED | PRICE_TAG_MISSING | UNKNOWN>",
          "location_on_shelf": "<left | center | right | full-shelf>",
          "estimated_missing_units": <integer or null>,
          "confidence": <0.0-1.0>,
          "notes": "<short note or null>"
        }
      ]
    }
  ]
}

Strict output rules:
- Number shelves from top to bottom as 1..N.
- product_name should be specific when visible, but never invented.
- facings_visible = visible front facings only.
- facings_expected = conservative estimate of intended facings for that contiguous block.
- gap_count = max(facings_expected - facings_visible, 0).
- status = LOW_STOCK if there are likely missing facings but some units remain.
- status = MESSY if products are leaning, rotated, buried, misaligned, or not properly front-faced.
- status = MISPLACED if an item appears in the wrong zone/category block.
- price_label = visible shelf-edge price for that block if readable, otherwise null.
- confidence must go down when the image is blurry, far away, occluded, reflective, or partly blocked.
- notes must be short and factual.
- overall_fill_rate_pct must be less than 100 if any product has gap_count > 0 or any shelf issue is GAP.
- total_gaps_detected must reflect all visible missing-facing situations across shelves.

Quality preference:
Prefer useful operational issue detection over optimistic classification.
""".strip()

COMPACT_VISION_PROMPT = """
Use the exact same schema, but keep the response compact and issue-focused:
- maximum 6 shelves
- maximum 4 product entries per shelf
- maximum 3 shelf_issues per shelf
- keep notes short
- never omit visible gaps or messy fronting just to save tokens
- if needed, merge adjacent identical facings into a single product zone entry
- still return strict valid JSON only
""".strip()

ULTRA_COMPACT_AUDIT_PROMPT = """
Return ONLY valid JSON using this smaller task-ready schema:
{
  "provider": "bedrock-nova-2-lite",
  "location": "<short location>",
  "overview": "<1 short sentence>",
  "confidence": <0.0-1.0>,
  "detected_products": [
    {
      "product_name": "<short visible name>",
      "brand": "<brand or null>",
      "category": "<category or null>",
      "shelf_level": "<short shelf label>",
      "estimated_units": <integer>,
      "facings": <integer>,
      "condition": "<HEALTHY | LOW_STOCK | MESSY | MIXED | UNCERTAIN>",
      "confidence": <0.0-1.0>,
      "notes": "<short note or null>"
    }
  ],
  "issues": [
    {
      "issue_type": "<LOW_STOCK | EMPTY_SPACE | MISALIGNED | MIXED_PRODUCTS | UNKNOWN_PRODUCT | PLANOGRAM_MISMATCH | LABEL_GAP | OBSTRUCTION>",
      "severity": "<LOW | MEDIUM | HIGH>",
      "product_name": "<name or null>",
      "shelf_level": "<short shelf label>",
      "details": "<short factual description>",
      "gap_units": <integer>,
      "confidence": <0.0-1.0>,
      "suggested_action": "<RESTOCK | REARRANGE | AUDIT>"
    }
  ]
}

Rules:
- maximum 10 detected_products
- maximum 8 issues
- keep names and notes short
- focus on the most important product blocks and actionable issues only
- do not include shelves array
- do not include markdown or commentary
""".strip()

REVIEW_VISION_PROMPT = """
You are reviewing a first-pass shelf audit that may be too optimistic.

Your job:
- Reinspect the same shelf image carefully.
- Correct missed issues, missed gaps, duplicate product blocks, and unrealistic 100% fill-rate claims.
- Be skeptical of "no issues" outputs on mixed promotional shelves, uneven fronting, broken product runs, or partially empty blocks.

Common failure modes to fix:
- Adjacent identical facings were split into repeated single-product entries instead of one contiguous block.
- Empty spans or broken front edges were missed.
- Messy or rotated packs were marked IN_STOCK instead of MESSY.
- Mixed flavors or misplaced items inside a product run were missed.
- Fill rate was set to 100 even though the shelf is visibly imperfect.

Rules:
- Return the same exact schema as the first-pass audit prompt.
- Prefer factual issue detection over optimism.
- If a visible problem could reasonably create work for store staff, include it.
- If no issue truly exists, you may return zero issues, but only if the image clearly supports that conclusion.
- Return ONLY valid JSON.
""".strip()

GAP_VERIFICATION_PROMPT = """
You are doing a second-pass issue-only review of a retail shelf image.

Focus only on operational problems that store staff would act on:
- visible gaps or broken product runs
- low-stock blocks with missing facings
- messy fronting or badly aligned packs
- misplaced products inside the wrong block
- shelf sections hidden by signage or obstructions that need a manual check

Be skeptical. Promotional shelves are rarely perfect.
If a product run has visible empty span, uneven spacing, or a broken front edge, report it.
A gap can exist between two different adjacent brands if exposed shelf or backing shows a missing product block between them.
If there is a visible open space between products like Oven Baked, Walkers, and M&M's, report that as GAP even when the neighboring products differ.
If a large sign or obstruction hides part of a block, report OBSTRUCTION or GAP if missing facings are still visible.

Return ONLY valid JSON using this schema:
{
  "gap_checks": [
    {
      "shelf_label": "<Top Shelf | Upper Shelf | Eye Level | Lower Shelf | Bottom Shelf | Unknown Shelf>",
      "location_on_shelf": "<left | center | right | full-shelf>",
      "related_product_name": "<best visible product name or null>",
      "issue_type": "<GAP | MESSY | MISPLACED | OBSTRUCTION>",
      "estimated_missing_units": <integer or null>,
      "confidence": <0.0-1.0>,
      "notes": "<short factual note>"
    }
  ]
}

Rules:
- Return at most 8 checks.
- Prefer reporting a visible issue over returning an empty list.
- Do not describe healthy zones.
- Do not include markdown or commentary.
""".strip()


ROW_AUDIT_PROMPT = """
You are auditing a single shelf row crop from a supermarket bay.

Goal:
Inspect this one row thoroughly and return every visible operational problem you can honestly support.
Do not stop at the top 1 or 2 issues.

What to detect on this row:
- contiguous product blocks
- visible front facings per block
- broken runs and open gaps
- low stock compared with neighboring block widths
- messy fronting, leaning, rotated, or buried packs
- mixed or misplaced items inside a block
- unreadable or missing price labels when obvious

Important row rules:
- This crop is only one shelf row, so focus deeply on this row only.
- Adjacent identical facings belong in one product block.
- Visible open space between different adjacent products can still be a GAP if exposed shelf or backing suggests a missing block.
- If a product block is uneven or not pulled forward, mark it MESSY or LOW_STOCK.
- Prefer LOW_STOCK over IN_STOCK when spacing suggests missing facings.
- Prefer MESSY over IN_STOCK when alignment is visibly poor.
- Do not invent exact SKUs when text is unclear.
- Return all actionable issues you can see, not just the largest issue.

Return ONLY valid JSON with this schema:
{
  "shelf_label": "<Top Shelf | Upper Shelf | Eye Level | Lower Shelf | Bottom Shelf | Unknown Shelf>",
  "products": [
    {
      "product_name": "<brand + variant if visible>",
      "facings_visible": <integer>,
      "facings_expected": <integer>,
      "gap_count": <integer>,
      "price_label": "<price string or null>",
      "status": "<IN_STOCK | LOW_STOCK | OUT_OF_STOCK | MISPLACED | MESSY | UNKNOWN>",
      "confidence": <0.0-1.0>,
      "notes": "<short factual note or null>"
    }
  ],
  "shelf_issues": [
    {
      "issue_type": "<GAP | MESSY | MISPLACED | PRICE_TAG_MISSING | OBSTRUCTION | UNKNOWN>",
      "location_on_shelf": "<left | center | right | full-shelf>",
      "estimated_missing_units": <integer or null>,
      "confidence": <0.0-1.0>,
      "notes": "<short factual note or null>"
    }
  ]
}

Rules:
- maximum 8 product blocks
- maximum 6 shelf_issues
- include every visible actionable issue on this row
- no markdown, no explanation, no preamble
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
                image=image,
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
    image: Any,
    normalized_bytes: bytes,
    image_format: str,
    location_hint: str,
) -> VisionShelfAudit:
    prompts = [
        (VISION_PROMPT, 2200, "full"),
        (f"{VISION_PROMPT}\n\n{COMPACT_VISION_PROMPT}", 1100, "compact"),
        (ULTRA_COMPACT_AUDIT_PROMPT, 900, "ultra"),
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
            parsed = _review_if_suspicious(
                client=client,
                normalized_bytes=normalized_bytes,
                image_format=image_format,
                location_hint=location_hint,
                parsed=parsed,
            )
            parsed = _augment_with_row_issue_audit(
                client=client,
                image=image,
                location_hint=location_hint,
                parsed=parsed,
            )
            parsed = _augment_with_gap_verification(
                client=client,
                image=image,
                normalized_bytes=normalized_bytes,
                image_format=image_format,
                location_hint=location_hint,
                parsed=parsed,
            )
            parsed = _post_process_structured_audit(parsed)
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


def _review_if_suspicious(
    *,
    client: Any,
    normalized_bytes: bytes,
    image_format: str,
    location_hint: str,
    parsed: dict[str, Any],
) -> dict[str, Any]:
    if not _looks_suspiciously_perfect(parsed):
        return parsed

    review_prompt = (
        f"{REVIEW_VISION_PROMPT}\n\n"
        "Initial audit to review and correct:\n"
        f"{json.dumps(parsed, ensure_ascii=True)}"
    )

    review_text = _call_bedrock_audit(
        client=client,
        normalized_bytes=normalized_bytes,
        image_format=image_format,
        location_hint=location_hint,
        prompt=review_prompt,
        max_tokens=1800,
    )
    _write_latest_response(review_text)
    reviewed = _extract_json(review_text)
    return _choose_more_actionable_payload(parsed, reviewed)


def _looks_suspiciously_perfect(payload: dict[str, Any]) -> bool:
    if "shelf_summary" in payload and "shelves" in payload:
        summary = payload.get("shelf_summary", {})
        shelves = payload.get("shelves", [])
        structured_issue_count = 0
        duplicate_blocks = 0

        for shelf in shelves:
            shelf_issues = shelf.get("shelf_issues", [])
            products = shelf.get("products", [])
            structured_issue_count += len(shelf_issues)

            seen_names: dict[str, int] = {}
            for product in products:
                status = str(product.get("status", "UNKNOWN"))
                gap_count = max(0, int(product.get("gap_count", 0) or 0))
                if status in {"LOW_STOCK", "OUT_OF_STOCK", "MESSY", "MISPLACED", "UNKNOWN"} or gap_count > 0:
                    structured_issue_count += 1

                name = str(product.get("product_name") or "").strip().lower()
                if name:
                    seen_names[name] = seen_names.get(name, 0) + 1

            duplicate_blocks += sum(1 for count in seen_names.values() if count > 1)

        total_products = int(summary.get("total_products_identified", 0) or 0)
        total_gaps = int(summary.get("total_gaps_detected", 0) or 0)
        fill_rate = float(summary.get("overall_fill_rate_pct", 0.0) or 0.0)

        return (
            structured_issue_count == 0
            and total_products >= 5
            and (fill_rate >= 99.0 or total_gaps == 0 or duplicate_blocks > 0)
        )

    summary = payload.get("summary", {})
    issue_count = int(summary.get("issue_count", len(payload.get("issues", []))) or 0)
    detected_count = int(summary.get("detected_product_count", len(payload.get("detected_products", []))) or 0)
    restock_count = int(summary.get("restock_task_count", 0) or 0)
    rearrange_count = int(summary.get("rearrange_task_count", 0) or 0)

    seen_names: dict[str, int] = {}
    for product in payload.get("detected_products", []):
        name = str(product.get("product_name") or "").strip().lower()
        if name:
            seen_names[name] = seen_names.get(name, 0) + 1
    duplicate_names = sum(1 for count in seen_names.values() if count > 1)

    return (
        issue_count == 0
        and restock_count == 0
        and rearrange_count == 0
        and detected_count >= 5
        and duplicate_names > 0
    )


def _choose_more_actionable_payload(initial: dict[str, Any], reviewed: dict[str, Any]) -> dict[str, Any]:
    initial_score = _payload_actionability_score(initial)
    reviewed_score = _payload_actionability_score(reviewed)
    if reviewed_score >= initial_score:
        return reviewed
    return initial


def _payload_actionability_score(payload: dict[str, Any]) -> tuple[int, float, int]:
    if "shelf_summary" in payload and "shelves" in payload:
        issue_count = 0
        total_gap_units = 0
        non_healthy_products = 0

        for shelf in payload.get("shelves", []):
            issue_count += len(shelf.get("shelf_issues", []))
            for product in shelf.get("products", []):
                gap_count = max(0, int(product.get("gap_count", 0) or 0))
                total_gap_units += gap_count
                if str(product.get("status", "IN_STOCK")) != "IN_STOCK" or gap_count > 0:
                    non_healthy_products += 1

        fill_rate = float(payload.get("shelf_summary", {}).get("overall_fill_rate_pct", 100.0) or 100.0)
        return (issue_count + non_healthy_products, -fill_rate, total_gap_units)

    summary = payload.get("summary", {})
    issue_count = int(summary.get("issue_count", len(payload.get("issues", []))) or 0)
    total_gap_units = sum(max(0, int(issue.get("gap_units", 0) or 0)) for issue in payload.get("issues", []))
    confidence = float(payload.get("confidence", 0.0) or 0.0)
    return (issue_count, total_gap_units + confidence, total_gap_units)


def _augment_with_gap_verification(
    *,
    client: Any,
    image: Any,
    normalized_bytes: bytes,
    image_format: str,
    location_hint: str,
    parsed: dict[str, Any],
) -> dict[str, Any]:
    if not _should_run_gap_verifier(parsed):
        return parsed

    verification_prompt = (
        f"{GAP_VERIFICATION_PROMPT}\n\n"
        "Initial audit summary to verify:\n"
        f"{json.dumps(parsed, ensure_ascii=True)}"
    )

    verification_text = _call_bedrock_audit(
        client=client,
        normalized_bytes=normalized_bytes,
        image_format=image_format,
        location_hint=location_hint,
        prompt=verification_prompt,
        max_tokens=900,
    )
    _write_latest_response(verification_text)
    verification_payload = _extract_json(verification_text)
    parsed = _merge_gap_checks_into_payload(parsed, verification_payload)
    return _augment_with_row_gap_verification(
        client=client,
        image=image,
        location_hint=location_hint,
        parsed=parsed,
    )


def _should_run_gap_verifier(payload: dict[str, Any]) -> bool:
    if "shelf_summary" not in payload or "shelves" not in payload:
        return False

    summary = payload.get("shelf_summary", {})
    shelves = payload.get("shelves", [])
    total_products = int(summary.get("total_products_identified", 0) or 0)
    total_gaps = int(summary.get("total_gaps_detected", 0) or 0)

    issue_count = 0
    for shelf in shelves:
        issue_count += len(shelf.get("shelf_issues", []))
        for product in shelf.get("products", []):
            if max(0, int(product.get("gap_count", 0) or 0)) > 0:
                issue_count += 1
            if str(product.get("status", "IN_STOCK")) in {"LOW_STOCK", "OUT_OF_STOCK", "MESSY", "MISPLACED"}:
                issue_count += 1

    return total_products >= 4 and total_gaps == 0 and issue_count <= 1


def _augment_with_row_issue_audit(
    *,
    client: Any,
    image: Any,
    location_hint: str,
    parsed: dict[str, Any],
) -> dict[str, Any]:
    if "shelf_summary" not in parsed or "shelves" not in parsed:
        return parsed

    shelves = parsed.get("shelves", [])
    row_count = len(shelves)
    if row_count <= 0 or row_count > 8:
        return parsed

    row_bytes_payloads = create_row_crops(image, row_count)

    for row_number, row_bytes, row_format in row_bytes_payloads:
        shelf = shelves[row_number - 1] if row_number - 1 < len(shelves) else None
        if shelf is None:
            continue

        shelf_label = str(shelf.get("shelf_label") or f"Shelf {row_number}")
        current_row_context = {
            "shelf_label": shelf_label,
            "products": shelf.get("products", []),
            "shelf_issues": shelf.get("shelf_issues", []),
        }
        row_prompt = (
            f"{ROW_AUDIT_PROMPT}\n\n"
            f"Focus only on shelf row {row_number}: {shelf_label}.\n"
            "Current first-pass row context is below. Correct it and add any missed issues.\n"
            f"{json.dumps(current_row_context, ensure_ascii=True)}"
        )

        try:
            row_text = _call_bedrock_audit(
                client=client,
                normalized_bytes=row_bytes,
                image_format=row_format,
                location_hint=f"{location_hint} - {shelf_label}",
                prompt=row_prompt,
                max_tokens=900,
            )
            _write_latest_response(row_text)
            row_payload = _extract_json(row_text)
            parsed = _merge_row_audit_into_payload(parsed, row_number=row_number, row_payload=row_payload)
        except Exception:
            continue

    return parsed


def _augment_with_row_gap_verification(
    *,
    client: Any,
    image: Any,
    location_hint: str,
    parsed: dict[str, Any],
) -> dict[str, Any]:
    if "shelf_summary" not in parsed or "shelves" not in parsed:
        return parsed

    shelves = parsed.get("shelves", [])
    row_count = len(shelves)
    if row_count <= 0 or row_count > 8:
        return parsed

    row_images = {row_number: row_image for row_number, row_image in create_row_crop_images(image, row_count)}
    row_bytes_payloads = create_row_crops(image, row_count)

    for row_number, row_bytes, row_format in row_bytes_payloads:
        shelf = shelves[row_number - 1] if row_number - 1 < len(shelves) else None
        row_image = row_images.get(row_number)
        if shelf is None or row_image is None:
            continue

        shelf_label = str(shelf.get("shelf_label") or f"Shelf {row_number}")
        row_prompt = (
            f"{GAP_VERIFICATION_PROMPT}\n\n"
            f"Focus only on shelf row {row_number}: {shelf_label}.\n"
            "This crop shows a single shelf band. Report broken runs, open spaces between adjacent products, messy fronting, or misplacements on this row.\n"
            "Treat visible empty space between different neighboring products as GAP if exposed shelf or backing suggests a missing block."
        )

        try:
            row_text = _call_bedrock_audit(
                client=client,
                normalized_bytes=row_bytes,
                image_format=row_format,
                location_hint=f"{location_hint} - {shelf_label}",
                prompt=row_prompt,
                max_tokens=500,
            )
            _write_latest_response(row_text)
            row_payload = _extract_json(row_text)
            if isinstance(row_payload, dict):
                for check in row_payload.get("gap_checks", []):
                    if isinstance(check, dict) and not check.get("shelf_label"):
                        check["shelf_label"] = shelf_label
                parsed = _merge_gap_checks_into_payload(parsed, row_payload)
        except Exception:
            pass

        heuristic_payload = {
            "gap_checks": _build_heuristic_gap_checks(
                shelf=shelf,
                shelf_label=shelf_label,
                gap_segments=detect_gap_segments(row_image),
            )
        }
        parsed = _merge_gap_checks_into_payload(parsed, heuristic_payload)

    return parsed


def _build_heuristic_gap_checks(
    *,
    shelf: dict[str, Any],
    shelf_label: str,
    gap_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for segment in gap_segments:
        location_on_shelf = str(segment.get("location_on_shelf") or "center")
        related_product_name = _guess_related_product_name_from_location(shelf, location_on_shelf)
        estimated_missing_units = max(1, int(segment.get("estimated_missing_units", 1) or 1))
        confidence = float(segment.get("confidence", 0.55) or 0.55)
        checks.append(
            {
                "shelf_label": shelf_label,
                "location_on_shelf": location_on_shelf,
                "related_product_name": related_product_name,
                "issue_type": "GAP",
                "estimated_missing_units": estimated_missing_units,
                "confidence": confidence,
                "notes": f"Heuristic gap detector found open shelf space on the {location_on_shelf} side of {shelf_label}.",
            }
        )
    return checks


def _guess_related_product_name_from_location(shelf: dict[str, Any], location_on_shelf: str) -> str | None:
    products = [product for product in shelf.get("products", []) if isinstance(product, dict)]
    if not products:
        return None
    if location_on_shelf == "left":
        return products[0].get("product_name")
    if location_on_shelf == "right":
        return products[-1].get("product_name")
    return products[len(products) // 2].get("product_name")


def _merge_row_audit_into_payload(payload: dict[str, Any], *, row_number: int, row_payload: dict[str, Any]) -> dict[str, Any]:
    if "shelf_summary" not in payload or "shelves" not in payload:
        return payload

    shelves = payload.get("shelves", [])
    if not shelves:
        return payload

    normalized_row = _normalize_row_audit_payload(row_payload)
    shelf_label = str(normalized_row.get("shelf_label") or "")
    shelf = shelves[row_number - 1] if row_number - 1 < len(shelves) else _find_matching_shelf(shelves, shelf_label)
    if shelf is None:
        return payload

    incoming_products = [item for item in normalized_row.get("products", []) if isinstance(item, dict)]
    incoming_issues = [item for item in normalized_row.get("shelf_issues", []) if isinstance(item, dict)]

    if incoming_products:
        shelf.setdefault("products", []).extend(incoming_products)
        shelf["products"] = _merge_and_clean_products(shelf.get("products", []))

    if incoming_issues:
        shelf.setdefault("shelf_issues", []).extend(incoming_issues)
        shelf["shelf_issues"] = _dedupe_shelf_issues(shelf.get("shelf_issues", []))

    return payload


def _normalize_row_audit_payload(row_payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row_payload, dict):
        return {"products": [], "shelf_issues": []}

    if isinstance(row_payload.get("shelf"), dict):
        row_payload = row_payload["shelf"]
    elif isinstance(row_payload.get("shelves"), list) and row_payload.get("shelves"):
        first_shelf = row_payload["shelves"][0]
        if isinstance(first_shelf, dict):
            row_payload = first_shelf

    return {
        "shelf_label": row_payload.get("shelf_label") or row_payload.get("label") or "Unknown Shelf",
        "products": row_payload.get("products", []) if isinstance(row_payload.get("products"), list) else [],
        "shelf_issues": row_payload.get("shelf_issues", []) if isinstance(row_payload.get("shelf_issues"), list) else [],
    }


def _merge_gap_checks_into_payload(payload: dict[str, Any], verification_payload: dict[str, Any]) -> dict[str, Any]:
    if "shelf_summary" not in payload or "shelves" not in payload:
        return payload

    checks = verification_payload.get("gap_checks", [])
    if not isinstance(checks, list) or not checks:
        return payload

    shelves = payload.get("shelves", [])
    added_gap_events = 0

    for check in checks:
        if not isinstance(check, dict):
            continue

        shelf = _find_matching_shelf(shelves, str(check.get("shelf_label") or ""))
        if shelf is None:
            continue

        issue_type = str(check.get("issue_type") or "UNKNOWN")
        location_on_shelf = str(check.get("location_on_shelf") or "full-shelf")
        related_product_name = check.get("related_product_name")
        estimated_missing_units = check.get("estimated_missing_units")
        confidence = float(check.get("confidence", 0.0) or 0.0)
        notes = check.get("notes") or None

        if _issue_already_present(shelf, issue_type, location_on_shelf, related_product_name):
            continue

        shelf.setdefault("shelf_issues", []).append(
            {
                "issue_type": issue_type,
                "location_on_shelf": location_on_shelf,
                "estimated_missing_units": estimated_missing_units,
                "confidence": confidence,
                "notes": notes,
            }
        )

        if issue_type == "GAP":
            added_gap_events += 1

        _apply_check_to_products(
            shelf=shelf,
            issue_type=issue_type,
            related_product_name=related_product_name,
            estimated_missing_units=estimated_missing_units,
            notes=notes,
        )

    if added_gap_events == 0:
        return payload

    summary = payload.setdefault("shelf_summary", {})
    summary["total_gaps_detected"] = max(int(summary.get("total_gaps_detected", 0) or 0), added_gap_events)
    current_fill_rate = float(summary.get("overall_fill_rate_pct", 100.0) or 100.0)
    summary["overall_fill_rate_pct"] = min(current_fill_rate, max(70.0, 100.0 - (added_gap_events * 4.0)))
    return payload


def _find_matching_shelf(shelves: list[dict[str, Any]], shelf_label: str) -> dict[str, Any] | None:
    normalized_target = _normalize_text(shelf_label)
    if not normalized_target:
        return shelves[0] if shelves else None

    for shelf in shelves:
        candidate = _normalize_text(str(shelf.get("shelf_label") or ""))
        if candidate == normalized_target:
            return shelf

    for shelf in shelves:
        candidate = _normalize_text(str(shelf.get("shelf_label") or ""))
        if normalized_target in candidate or candidate in normalized_target:
            return shelf

    return shelves[0] if shelves else None


def _issue_already_present(
    shelf: dict[str, Any],
    issue_type: str,
    location_on_shelf: str,
    related_product_name: Any,
) -> bool:
    normalized_name = _normalize_text(related_product_name)
    for existing in shelf.get("shelf_issues", []):
        same_type = str(existing.get("issue_type") or "") == issue_type
        same_location = str(existing.get("location_on_shelf") or "") == location_on_shelf
        notes_text = _normalize_text(existing.get("notes"))
        if same_type and same_location:
            if not normalized_name or normalized_name in notes_text:
                return True
    return False


def _apply_check_to_products(
    *,
    shelf: dict[str, Any],
    issue_type: str,
    related_product_name: Any,
    estimated_missing_units: Any,
    notes: Any,
) -> None:
    if issue_type not in {"GAP", "MESSY", "MISPLACED"}:
        return

    product = _find_matching_product(shelf.get("products", []), related_product_name)
    if product is None:
        return

    missing_units = max(1, int(estimated_missing_units or 1)) if issue_type == "GAP" else 0
    existing_notes = product.get("notes")
    combined_notes = notes or existing_notes
    if existing_notes and notes and notes not in existing_notes:
        combined_notes = f"{existing_notes} {notes}".strip()

    if issue_type == "GAP":
        visible = max(0, int(product.get("facings_visible", 0) or 0))
        product["facings_expected"] = max(int(product.get("facings_expected", visible) or visible), visible + missing_units)
        product["gap_count"] = max(int(product.get("gap_count", 0) or 0), missing_units)
        product["status"] = "LOW_STOCK" if visible > 0 else "OUT_OF_STOCK"
        product["notes"] = combined_notes or "Gap verified in second pass."
    elif issue_type == "MESSY":
        product["status"] = "MESSY"
        product["notes"] = combined_notes or "Messy fronting verified in second pass."
    elif issue_type == "MISPLACED":
        product["status"] = "MISPLACED"
        product["notes"] = combined_notes or "Possible misplacement verified in second pass."


def _find_matching_product(products: list[dict[str, Any]], related_product_name: Any) -> dict[str, Any] | None:
    target = _normalize_text(related_product_name)
    if not target:
        return None

    for product in products:
        name = _normalize_text(product.get("product_name"))
        if name == target:
            return product

    for product in products:
        name = _normalize_text(product.get("product_name"))
        if target and (target in name or name in target):
            return product

    return None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _post_process_structured_audit(payload: dict[str, Any]) -> dict[str, Any]:
    if "shelf_summary" not in payload or "shelves" not in payload:
        return payload

    shelves = payload.get("shelves", [])
    total_gaps = 0
    total_products = 0

    for shelf in shelves:
        merged_products = _merge_and_clean_products(shelf.get("products", []))
        shelf["products"] = merged_products
        shelf["shelf_issues"] = _dedupe_shelf_issues(shelf.get("shelf_issues", []))
        total_products += len(merged_products)
        total_gaps += sum(max(0, int(product.get("gap_count", 0) or 0)) for product in merged_products)

    total_shelf_issues = sum(len(shelf.get("shelf_issues", [])) for shelf in shelves)
    summary = payload.setdefault("shelf_summary", {})
    summary["total_shelves_detected"] = max(int(summary.get("total_shelves_detected", len(shelves)) or 0), len(shelves))
    summary["total_products_identified"] = total_products
    summary["total_gaps_detected"] = max(int(summary.get("total_gaps_detected", 0) or 0), total_gaps)

    if total_products > 0:
        healthy_weight = sum(
            1
            for shelf in shelves
            for product in shelf.get("products", [])
            if str(product.get("status", "UNKNOWN")) == "IN_STOCK" and max(0, int(product.get("gap_count", 0) or 0)) == 0
        )
        estimated_fill = 100.0 * healthy_weight / total_products
        if total_gaps > 0 or total_shelf_issues > 0:
            estimated_fill = min(estimated_fill, 97.0 - min(total_gaps * 2.5 + total_shelf_issues * 1.5, 25.0))
        summary["overall_fill_rate_pct"] = round(max(55.0, min(float(summary.get("overall_fill_rate_pct", estimated_fill) or estimated_fill), estimated_fill)), 1)

    return payload


def _merge_and_clean_products(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []

    for raw_product in products:
        if not isinstance(raw_product, dict):
            continue

        product = dict(raw_product)
        product_name = _clean_product_name(product.get("product_name"))
        if not product_name:
            product_name = "Unknown product"
        product["product_name"] = product_name
        product["status"] = _recalibrate_product_status(product)
        product["notes"] = _clean_note(product.get("notes"))
        product["price_label"] = _clean_price_label(product.get("price_label"))

        existing = _find_merge_candidate(merged, product)
        if existing is None:
            merged.append(product)
            continue

        existing["facings_visible"] = max(0, int(existing.get("facings_visible", 0) or 0)) + max(0, int(product.get("facings_visible", 0) or 0))
        existing["facings_expected"] = max(
            max(0, int(existing.get("facings_expected", 0) or 0)),
            max(0, int(product.get("facings_expected", 0) or 0)),
            existing["facings_visible"],
        )
        existing["gap_count"] = max(0, int(existing.get("gap_count", 0) or 0)) + max(0, int(product.get("gap_count", 0) or 0))
        existing["confidence"] = round(max(float(existing.get("confidence", 0.0) or 0.0), float(product.get("confidence", 0.0) or 0.0)), 3)
        existing["status"] = _more_severe_status(str(existing.get("status", "UNKNOWN")), str(product.get("status", "UNKNOWN")))
        existing["notes"] = _merge_text(existing.get("notes"), product.get("notes"))
        if not existing.get("price_label") and product.get("price_label"):
            existing["price_label"] = product["price_label"]

    return merged


def _find_merge_candidate(products: list[dict[str, Any]], candidate: dict[str, Any]) -> dict[str, Any] | None:
    candidate_name = _normalize_text(candidate.get("product_name"))
    candidate_price = _clean_price_label(candidate.get("price_label"))

    for product in products:
        product_name = _normalize_text(product.get("product_name"))
        same_name = product_name == candidate_name and candidate_name != ""
        close_name = candidate_name and (candidate_name in product_name or product_name in candidate_name)
        same_price = candidate_price and _clean_price_label(product.get("price_label")) == candidate_price
        if same_name or (close_name and same_price):
            return product

    return None


def _recalibrate_product_status(product: dict[str, Any]) -> str:
    current = str(product.get("status", "UNKNOWN") or "UNKNOWN")
    visible = max(0, int(product.get("facings_visible", 0) or 0))
    expected = max(visible, int(product.get("facings_expected", visible) or visible))
    gap_count = max(0, int(product.get("gap_count", 0) or 0))
    confidence = float(product.get("confidence", 0.0) or 0.0)
    notes = _normalize_text(product.get("notes"))
    name = _normalize_text(product.get("product_name"))

    if any(token in notes for token in ["messy", "lean", "rotated", "uneven", "misaligned", "not forward", "buried"]):
        return "MESSY"
    if any(token in notes for token in ["misplaced", "wrong block", "wrong shelf", "mixed"]):
        return "MISPLACED"
    if gap_count >= expected and expected > 0:
        return "OUT_OF_STOCK"
    if gap_count > 0 or expected > visible:
        return "LOW_STOCK"
    if confidence < 0.45 and current == "IN_STOCK":
        return "UNKNOWN"
    if len(name.split()) == 1 and confidence < 0.65 and current == "IN_STOCK":
        return "UNKNOWN"
    return current


def _dedupe_shelf_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for issue in issues:
        if not isinstance(issue, dict):
            continue
        key = (
            str(issue.get("issue_type") or "UNKNOWN"),
            str(issue.get("location_on_shelf") or "full-shelf"),
            _normalize_text(issue.get("notes")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)

    return deduped


def _clean_product_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[^A-Za-z0-9]+", "", text)
    text = re.sub(r"(regular|original)$", lambda match: match.group(1).title(), text, flags=re.IGNORECASE)
    return text[:80].strip()


def _clean_note(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _clean_price_label(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _merge_text(left: Any, right: Any) -> str | None:
    left_text = _clean_note(left)
    right_text = _clean_note(right)
    if left_text and right_text:
        if right_text in left_text:
            return left_text
        if left_text in right_text:
            return right_text
        return f"{left_text} {right_text}".strip()
    return left_text or right_text


def _more_severe_status(left: str, right: str) -> str:
    ranking = {
        "OUT_OF_STOCK": 5,
        "LOW_STOCK": 4,
        "MESSY": 3,
        "MISPLACED": 3,
        "UNKNOWN": 2,
        "IN_STOCK": 1,
    }
    return left if ranking.get(left, 0) >= ranking.get(right, 0) else right


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
    if "shelf_summary" in payload and "shelves" in payload:
        return _convert_structured_shelf_audit(payload, fallback_location=fallback_location, fallback_provider=fallback_provider)

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


def _convert_structured_shelf_audit(payload: dict[str, Any], *, fallback_location: str, fallback_provider: str) -> VisionShelfAudit:
    summary = payload.get("shelf_summary", {})
    shelves = payload.get("shelves", [])

    detected_products: list[DetectedProduct] = []
    issues: list[ShelfIssue] = []
    confidence_values: list[float] = []

    for shelf in shelves:
        shelf_label = shelf.get("shelf_label") or f"Shelf {shelf.get('shelf_number', '?')}"
        products = shelf.get("products", [])
        shelf_issues = shelf.get("shelf_issues", [])

        for product in products:
            status = str(product.get("status", "UNKNOWN"))
            product_confidence = float(product.get("confidence", 0.0) or 0.0)
            confidence_values.append(product_confidence)
            detected_products.append(
                DetectedProduct(
                    product_name=product.get("product_name") or "Unknown product",
                    brand=_infer_brand(product.get("product_name")),
                    category=_infer_category(product.get("product_name")),
                    shelf_level=shelf_label,
                    estimated_units=max(0, int(product.get("facings_visible", 0) or 0)),
                    facings=max(0, int(product.get("facings_visible", 0) or 0)),
                    condition=_map_product_status(status),
                    confidence=product_confidence,
                    notes=product.get("notes"),
                )
            )

            product_issue = _product_to_issue(product, shelf_label)
            if product_issue is not None:
                issues.append(product_issue)

        for shelf_issue in shelf_issues:
            issue_confidence = float(shelf_issue.get("confidence", 0.0) or 0.0)
            confidence_values.append(issue_confidence)
            issues.append(
                ShelfIssue(
                    issue_type=_map_shelf_issue_type(shelf_issue.get("issue_type")),
                    severity=_derive_severity(shelf_issue.get("estimated_missing_units"), shelf_issue.get("issue_type")),
                    product_name=None,
                    shelf_level=f"{shelf_label} ({shelf_issue.get('location_on_shelf', 'unknown')})",
                    details=shelf_issue.get("notes") or f"{shelf_issue.get('issue_type', 'UNKNOWN')} detected on {shelf_label}.",
                    gap_units=max(0, int(shelf_issue.get("estimated_missing_units", 0) or 0)),
                    confidence=issue_confidence,
                    suggested_action=_map_suggested_action(shelf_issue.get("issue_type")),
                )
            )

    avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    overview = (
        f"Detected {int(summary.get('total_products_identified', len(detected_products)))} product zones across "
        f"{int(summary.get('total_shelves_detected', len(shelves)))} shelves, with "
        f"{int(summary.get('total_gaps_detected', 0))} gaps and an estimated fill rate of "
        f"{float(summary.get('overall_fill_rate_pct', 0.0)):.1f}%."
    )

    return VisionShelfAudit(
        provider=fallback_provider,
        location=fallback_location,
        overview=overview,
        confidence=avg_confidence,
        detected_products=detected_products,
        issues=issues,
    )


def _product_to_issue(product: dict[str, Any], shelf_label: str) -> ShelfIssue | None:
    status = str(product.get("status", "UNKNOWN"))
    gap_count = max(0, int(product.get("gap_count", 0) or 0))
    product_name = product.get("product_name") or "Unknown product"
    confidence = float(product.get("confidence", 0.0) or 0.0)
    notes = product.get("notes") or None

    if status in {"LOW_STOCK", "OUT_OF_STOCK"} or gap_count > 0:
        issue_type = "EMPTY_SPACE" if status == "OUT_OF_STOCK" else "LOW_STOCK"
        return ShelfIssue(
            issue_type=issue_type,
            severity=_derive_severity(gap_count, status),
            product_name=product_name,
            shelf_level=shelf_label,
            details=notes or f"{product_name} has {gap_count} estimated gap facings.",
            gap_units=gap_count,
            confidence=confidence,
            suggested_action="RESTOCK",
        )

    if status == "MESSY":
        return ShelfIssue(
            issue_type="MISALIGNED",
            severity="MEDIUM",
            product_name=product_name,
            shelf_level=shelf_label,
            details=notes or f"{product_name} looks untidy or not forward-faced.",
            gap_units=0,
            confidence=confidence,
            suggested_action="REARRANGE",
        )

    if status == "MISPLACED":
        return ShelfIssue(
            issue_type="MIXED_PRODUCTS",
            severity="MEDIUM",
            product_name=product_name,
            shelf_level=shelf_label,
            details=notes or f"{product_name} appears to be in the wrong shelf zone.",
            gap_units=0,
            confidence=confidence,
            suggested_action="REARRANGE",
        )

    if status == "UNKNOWN":
        return ShelfIssue(
            issue_type="UNKNOWN_PRODUCT",
            severity="LOW",
            product_name=product_name,
            shelf_level=shelf_label,
            details=notes or f"{product_name} could not be clearly identified.",
            gap_units=0,
            confidence=confidence,
            suggested_action="AUDIT",
        )

    return None


def _map_product_status(status: str) -> str:
    mapping = {
        "IN_STOCK": "HEALTHY",
        "LOW_STOCK": "LOW_STOCK",
        "OUT_OF_STOCK": "LOW_STOCK",
        "MISPLACED": "MIXED",
        "MESSY": "MESSY",
        "UNKNOWN": "UNCERTAIN",
    }
    return mapping.get(status, "UNCERTAIN")


def _map_shelf_issue_type(issue_type: Any) -> str:
    mapping = {
        "GAP": "EMPTY_SPACE",
        "MESSY": "MISALIGNED",
        "MISPLACED": "MIXED_PRODUCTS",
        "PRICE_TAG_MISSING": "LABEL_GAP",
        "OBSTRUCTION": "OBSTRUCTION",
        "UNKNOWN": "UNKNOWN_PRODUCT",
    }
    return mapping.get(str(issue_type), "UNKNOWN_PRODUCT")


def _map_suggested_action(issue_type: Any) -> str:
    mapping = {
        "GAP": "RESTOCK",
        "MESSY": "REARRANGE",
        "MISPLACED": "REARRANGE",
        "PRICE_TAG_MISSING": "AUDIT",
        "OBSTRUCTION": "AUDIT",
        "UNKNOWN": "AUDIT",
    }
    return mapping.get(str(issue_type), "AUDIT")


def _derive_severity(missing_units: Any, marker: Any) -> str:
    units = max(0, int(missing_units or 0))
    marker_text = str(marker)
    if marker_text in {"OUT_OF_STOCK", "GAP"} or units >= 4:
        return "HIGH"
    if marker_text in {"LOW_STOCK", "MESSY", "MISPLACED"} or units >= 1:
        return "MEDIUM"
    return "LOW"


def _infer_brand(product_name: Any) -> str | None:
    if not product_name:
        return None
    tokens = [token for token in re.split(r"\s+", str(product_name).strip()) if token]
    if not tokens:
        return None

    if len(tokens) >= 2 and tokens[0].lower() in {"mr", "dr", "tesco", "coca-cola", "coca", "rice", "fruit"}:
        if tokens[0].lower() == "coca" and tokens[1].lower().startswith("cola"):
            return "Coca-Cola"
        if tokens[0].lower() == "rice" and len(tokens) >= 2:
            return "Kellogg's"
        if tokens[0].lower() == "fruit" and len(tokens) >= 2 and tokens[1].lower() == "shoot":
            return "Fruit Shoot"
        return " ".join(tokens[:2])

    return tokens[0]


def _infer_category(product_name: Any) -> str | None:
    if not product_name:
        return None
    name = _normalize_text(product_name)
    if any(token in name for token in ["cola", "coke", "lemonade", "fanta", "sprite", "pepsi", "vimto"]):
        return "Soft Drinks"
    if any(token in name for token in ["juice", "cranberry", "orange", "apple", "smoothie"]):
        return "Juice"
    if any(token in name for token in ["cereal", "krispies", "corn flakes", "muesli"]):
        return "Breakfast Cereal"
    if any(token in name for token in ["walkers", "sensations", "hula hoops", "chips", "crisps"]):
        return "Crisps"
    if any(token in name for token in ["mars", "snickers", "bounty", "maltesers", "m m", "m&m", "counters"]):
        return "Confectionery"
    if any(token in name for token in ["cordial", "squash", "ribena", "robinsons"]):
        return "Squash and Cordial"
    return None


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

