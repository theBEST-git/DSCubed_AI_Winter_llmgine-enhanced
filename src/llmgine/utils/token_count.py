"""
Tiny helper that counts how many tokens a list of OpenAI‑style messages will
consume, using the same encoding as the main LLM model (gpt‑4o‑mini).
"""
from __future__ import annotations

from typing import List, Dict

import tiktoken

_ENCODER = tiktoken.encoding_for_model("gpt-4o-mini")


def count_tokens(messages: List[Dict]) -> int:
    """Return the total number of tokens across message contents."""
    return sum(
        len(_ENCODER.encode(msg.get("content") or ""))
        for msg in messages
    )
