from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PlanogramItem(BaseModel):
    expected_units: int = Field(ge=0)
    expected_facings: int = Field(ge=0)
    location: str
    shelf_level: str


class DetectedProduct(BaseModel):
    product_name: str
    brand: str | None = None
    category: str | None = None
    shelf_level: str
    estimated_units: int = Field(ge=0)
    facings: int = Field(ge=0)
    condition: Literal["HEALTHY", "LOW_STOCK", "MESSY", "MIXED", "UNCERTAIN"]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str | None = None


class ShelfIssue(BaseModel):
    issue_type: Literal[
        "LOW_STOCK",
        "EMPTY_SPACE",
        "MISALIGNED",
        "MIXED_PRODUCTS",
        "UNKNOWN_PRODUCT",
        "PLANOGRAM_MISMATCH",
        "LABEL_GAP",
        "OBSTRUCTION",
    ]
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    product_name: str | None = None
    shelf_level: str
    details: str
    gap_units: int = Field(default=0, ge=0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    suggested_action: Literal["RESTOCK", "REARRANGE", "AUDIT"]


class VisionShelfAudit(BaseModel):
    provider: str
    location: str
    overview: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    detected_products: list[DetectedProduct] = Field(default_factory=list)
    issues: list[ShelfIssue] = Field(default_factory=list)


class TaskRecord(BaseModel):
    id: int
    task_type: Literal["RESTOCK", "REARRANGE", "AUDIT"]
    priority: Literal["LOW", "MEDIUM", "HIGH"]
    product_name: str | None = None
    quantity: int = Field(default=0, ge=0)
    location: str
    shelf_level: str
    reason: str
    status: Literal["PENDING", "COMPLETED"] = "PENDING"


class ShelfAuditSummary(BaseModel):
    detected_product_count: int = Field(ge=0)
    issue_count: int = Field(ge=0)
    restock_task_count: int = Field(ge=0)
    rearrange_task_count: int = Field(ge=0)
    audit_task_count: int = Field(ge=0)


class VoiceInstruction(BaseModel):
    provider: str
    text: str
    audio_base64: str | None = None


class ScanShelfResponse(BaseModel):
    provider: str
    location: str
    overview: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: ShelfAuditSummary
    detected_products: list[DetectedProduct]
    issues: list[ShelfIssue]
    tasks: list[TaskRecord]
    instruction: str
    voice: VoiceInstruction
