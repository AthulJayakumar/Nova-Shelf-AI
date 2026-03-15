from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import boto3
from botocore.config import Config
from dotenv import load_dotenv


load_dotenv()


def _normalize_nova_model_id(model_id: str, region: str) -> str:
    value = model_id.strip()
    if not value:
        return value

    if value.startswith(("us.", "global.", "eu.", "jp.")):
        return value

    if value.startswith("amazon.nova-2-"):
        prefix = "us" if region.startswith("us-") else "global"
        return f"{prefix}.{value}"

    return value


@dataclass(frozen=True)
class Settings:
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    bedrock_enabled: bool = os.getenv("BEDROCK_ENABLED", "false").lower() == "true"
    nova_lite_model_id_raw: str = os.getenv("NOVA_LITE_MODEL_ID", "us.amazon.nova-2-lite-v1:0")
    nova_sonic_enabled: bool = os.getenv("NOVA_SONIC_ENABLED", "false").lower() == "true"
    nova_sonic_model_id_raw: str = os.getenv("NOVA_SONIC_MODEL_ID", "us.amazon.nova-2-sonic-v1:0")
    default_planogram_product: str = os.getenv("DEFAULT_PLANOGRAM_PRODUCT", "Brand X Cereal")

    @property
    def nova_lite_model_id(self) -> str:
        return _normalize_nova_model_id(self.nova_lite_model_id_raw, self.aws_region)

    @property
    def nova_sonic_model_id(self) -> str:
        return _normalize_nova_model_id(self.nova_sonic_model_id_raw, self.aws_region)


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_bedrock_client() -> Any | None:
    settings = get_settings()
    if not settings.bedrock_enabled:
        return None

    return boto3.client(
        "bedrock-runtime",
        region_name=settings.aws_region,
        config=Config(
            read_timeout=3600,
            connect_timeout=30,
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )
