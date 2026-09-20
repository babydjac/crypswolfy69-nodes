"""Split a pasted prompt dump into one prompt per generation."""

from __future__ import annotations

import re

_NUMBERED = re.compile(r"(?m)^\s*\d+[.)]\s+")
_RULE = re.compile(r"(?m)^\s*(?:-{3,}|={3,})\s*$")
_BLANK = re.compile(r"\n\s*\n")


def parse_prompts(text: str, split_mode: str = "auto") -> list[str]:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return []

    mode = (split_mode or "auto").strip().lower()
    if mode == "forced":
        return [raw] if raw else []
    if mode == "numbered":
        return _clean(_NUMBERED.split(raw))
    if mode == "blank lines":
        return _clean(_BLANK.split(raw))
    if mode == "one per line":
        return _clean(raw.split("\n"))

    numbered = _clean(_NUMBERED.split(raw))
    if len(numbered) >= 2:
        return numbered

    ruled = _clean(_RULE.split(raw))
    if len(ruled) >= 2:
        return ruled

    blanks = _clean(_BLANK.split(raw))
    if len(blanks) >= 2:
        return blanks

    lines = _clean(raw.split("\n"))
    if len(lines) >= 2 and all(len(line) > 24 for line in lines):
        return lines

    return [raw]


def _clean(parts: list[str]) -> list[str]:
    out: list[str] = []
    for part in parts:
        item = (part or "").strip()
        if item:
            out.append(item)
    return out


def pick_prompt(
    text: str,
    prompt_index: int,
    split_mode: str = "auto",
    wrap: bool = True,
    forced_prompt: str = "",
) -> tuple[str, int, int]:
    forced = (forced_prompt or "").strip()
    items = parse_prompts(text, split_mode)
    if forced:
        if not items:
            items = [forced]
        idx = max(1, int(prompt_index or 1))
        if wrap and items:
            idx = ((idx - 1) % len(items)) + 1
        return forced, idx, len(items)

    if not items:
        return "", 0, 0

    idx = int(prompt_index or 1)
    if wrap:
        idx = ((idx - 1) % len(items)) + 1
    else:
        idx = max(1, min(idx, len(items)))
    return items[idx - 1], idx, len(items)
