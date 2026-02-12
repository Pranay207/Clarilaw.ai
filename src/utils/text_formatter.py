from __future__ import annotations

import re
from typing import Iterable, List, Optional


_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """
    Normalize spaces and newlines:
    - collapse tabs/spaces to a single space
    - normalize newlines to \\n
    - remove trailing spaces on each line
    - collapse 3+ newlines to 2
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WHITESPACE_RE.sub(" ", line).rstrip() for line in text.split("\n")]
    text = "\n".join(lines).strip()
    return _MULTI_NEWLINE_RE.sub("\n\n", text)


def sentence_case(text: str) -> str:
    """
    Convert text to sentence case while preserving existing line breaks.
    """
    text = normalize_whitespace(text)
    if not text:
        return ""

    def _fix_sentence(s: str) -> str:
        s = s.strip()
        if not s:
            return s
        return s[0].upper() + s[1:].lower()

    parts: List[str] = []
    for line in text.split("\n"):
        if not line.strip():
            parts.append("")
            continue
        sentences = re.split(r"([.!?])\s+", line)
        rebuilt: List[str] = []
        i = 0
        while i < len(sentences):
            chunk = sentences[i]
            if i + 1 < len(sentences) and sentences[i + 1] in ".!?":
                rebuilt.append(_fix_sentence(chunk) + sentences[i + 1])
                i += 2
            else:
                rebuilt.append(_fix_sentence(chunk))
                i += 1
        parts.append(" ".join(s for s in rebuilt if s))
    return "\n".join(parts).strip()


def to_title_case(text: str) -> str:
    text = normalize_whitespace(text)
    return text.title() if text else ""


def to_lower(text: str) -> str:
    text = normalize_whitespace(text)
    return text.lower() if text else ""


def to_upper(text: str) -> str:
    text = normalize_whitespace(text)
    return text.upper() if text else ""


def truncate(text: str, max_len: int, suffix: str = "...") -> str:
    text = normalize_whitespace(text)
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if len(suffix) >= max_len:
        return text[:max_len]
    return text[: max_len - len(suffix)].rstrip() + suffix


def split_sentences(text: str) -> List[str]:
    """
    Basic sentence splitter. Not language-aware, but good enough for UI display.
    """
    text = normalize_whitespace(text)
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def join_bullets(items: Iterable[str], bullet: str = "- ") -> str:
    cleaned = [normalize_whitespace(x) for x in items]
    cleaned = [x for x in cleaned if x]
    return "\n".join(f"{bullet}{x}" for x in cleaned)


def format_text(
    text: str,
    *,
    case: Optional[str] = None,
    max_len: Optional[int] = None,
    collapse_whitespace: bool = True
) -> str:
    """
    General formatter with optional case normalization and truncation.
    case: one of {"lower", "upper", "title", "sentence"}.
    """
    if collapse_whitespace:
        text = normalize_whitespace(text)

    if case:
        case = case.lower()
        if case == "lower":
            text = to_lower(text)
        elif case == "upper":
            text = to_upper(text)
        elif case == "title":
            text = to_title_case(text)
        elif case == "sentence":
            text = sentence_case(text)

    if max_len is not None:
        text = truncate(text, max_len=max_len)

    return text
