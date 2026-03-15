from __future__ import annotations

from database.models import VoiceInstruction


def generate_voice(text: str) -> VoiceInstruction:
    return VoiceInstruction(
        provider="text-fallback",
        text=text,
        audio_base64=None,
    )
