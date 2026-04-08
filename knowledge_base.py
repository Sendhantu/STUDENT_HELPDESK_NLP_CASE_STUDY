"""
knowledge_base.py — JSON-backed SRM helpdesk knowledge base loader.

This module keeps the app-facing API simple (`QA_PAIRS` stays importable)
while adding stronger normalisation, richer metadata, and a few lookup helpers
that make the expanded knowledge base easier to search and inspect.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from copy import deepcopy
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

# ── Path to the JSON knowledge base ───────────────────────────────────────
KB_PATH = Path(__file__).parent / "knowledge_base.json"

# ── Required fields every QA pair must have ───────────────────────────────
REQUIRED_FIELDS = {"question", "answer", "intent", "keywords"}

# ── Optional fields with their default values ─────────────────────────────
OPTIONAL_DEFAULTS = {
    "tags": [],
    "aliases": [],
    "related_questions": [],
    "source": "Unknown",
    "department": "",
    "last_updated": "2024-01-01",
    "answer_steps": [],
    "priority": 1,
}

DEFAULT_META = {
    "version": "0.0",
    "institution": "SRM Institute of Science and Technology",
    "last_updated": "2024-01-01",
    "total_pairs": 0,
}


def _normalise_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalise_string_list(value: object, *, lowercase: bool = False) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = []

    seen: set[str] = set()
    normalised: list[str] = []
    for item in items:
        text = _normalise_text(item)
        if not text:
            continue
        if lowercase:
            text = text.lower()
        if text not in seen:
            seen.add(text)
            normalised.append(text)
    return normalised


def _safe_date(value: object, fallback: str = "2024-01-01") -> str:
    text = _normalise_text(value)
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return fallback


def _safe_int(value: object, fallback: int = 1) -> int:
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return fallback


def _domain_from_intent(intent: str) -> str:
    return intent.split("_", 1)[0] if "_" in intent else intent or "general"


def _keywords_from_intent(intent: str) -> list[str]:
    if not intent:
        return []
    parts = [part for part in intent.split("_") if part]
    if not parts:
        return []
    joined = " ".join(parts)
    return parts + [joined]


def _build_search_terms(pair: dict) -> list[str]:
    extra_terms = []
    extra_terms.extend(pair.get("keywords", []))
    extra_terms.extend(pair.get("tags", []))
    extra_terms.extend(pair.get("aliases", []))
    extra_terms.extend(pair.get("related_questions", []))
    extra_terms.extend(_keywords_from_intent(pair.get("intent", "")))
    return _normalise_string_list(extra_terms, lowercase=True)


def _validate_pair(pair: dict, index: int) -> dict:
    """
    Validate and normalise a single QA pair.
    Missing required fields are logged but do not crash the app.
    """
    pair = dict(pair or {})

    for field in REQUIRED_FIELDS:
        if field not in pair or not _normalise_text(pair[field]):
            log.warning(
                "QA pair #%d is missing required field '%s'. "
                "It will be included but may not match correctly.",
                index,
                field,
            )

    for field, default in OPTIONAL_DEFAULTS.items():
        if field not in pair:
            pair[field] = deepcopy(default)

    pair["question"] = _normalise_text(pair.get("question"))
    pair["answer"] = _normalise_text(pair.get("answer"))
    pair["intent"] = _normalise_text(pair.get("intent")).lower()
    pair["source"] = _normalise_text(pair.get("source")) or "Unknown"
    pair["department"] = _normalise_text(pair.get("department"))
    pair["last_updated"] = _safe_date(pair.get("last_updated"))
    pair["priority"] = _safe_int(pair.get("priority"), fallback=1)

    pair["answer_steps"] = _normalise_string_list(pair.get("answer_steps"))
    pair["tags"] = _normalise_string_list(pair.get("tags"), lowercase=True)
    pair["aliases"] = _normalise_string_list(pair.get("aliases"))
    pair["related_questions"] = _normalise_string_list(pair.get("related_questions"))
    pair["keywords"] = _normalise_string_list(pair.get("keywords"), lowercase=True)

    # Enrich keywords using aliases, related prompts, tags, and intent words.
    pair["keywords"] = _build_search_terms(pair)
    pair["domain"] = _domain_from_intent(pair["intent"])
    pair["search_text"] = " | ".join(
        filter(
            None,
            [
                pair["question"].lower(),
                pair["answer"].lower(),
                " ".join(pair["keywords"]),
                " ".join(pair["aliases"]).lower(),
                " ".join(pair["related_questions"]).lower(),
            ],
        )
    )
    return pair


def _normalise_meta(meta: dict | None, pair_count: int) -> dict:
    meta = {**DEFAULT_META, **(meta or {})}
    meta["version"] = _normalise_text(meta.get("version")) or DEFAULT_META["version"]
    meta["institution"] = (
        _normalise_text(meta.get("institution")) or DEFAULT_META["institution"]
    )
    meta["last_updated"] = _safe_date(meta.get("last_updated"))
    meta["total_pairs"] = pair_count
    return meta


def load_knowledge_base(path: Path = KB_PATH) -> tuple[list[dict], dict]:
    """
    Load, validate, and return the knowledge base plus its metadata.
    Falls back to an empty list and default metadata if the file is missing or
    malformed so the server can keep running.
    """
    if not path.exists():
        log.error(
            "Knowledge base file not found at '%s'. "
            "The chatbot will not answer any questions.",
            path,
        )
        return [], _normalise_meta({}, 0)

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        log.error("Failed to parse '%s': %s", path, exc)
        return [], _normalise_meta({}, 0)

    raw_pairs = data.get("qa_pairs", [])
    if not raw_pairs:
        log.warning("No qa_pairs found in '%s'.", path)
        return [], _normalise_meta(data.get("meta"), 0)

    validated = [_validate_pair(pair, i) for i, pair in enumerate(raw_pairs)]
    meta = _normalise_meta(data.get("meta"), len(validated))

    declared_total = data.get("meta", {}).get("total_pairs")
    if declared_total != len(validated):
        log.warning(
            "Knowledge base meta total_pairs=%s does not match actual pair count=%d. "
            "Using the actual count.",
            declared_total,
            len(validated),
        )

    log.info(
        "Knowledge base loaded: %d pairs | version=%s | last_updated=%s",
        len(validated),
        meta.get("version", "?"),
        meta.get("last_updated", "?"),
    )
    return validated, meta


def load_qa_pairs(path: Path = KB_PATH) -> list[dict]:
    """Compatibility wrapper that returns only the validated QA pairs."""
    return load_knowledge_base(path)[0]


# ── Public globals (imported by app.py and nlp_engine.py) ─────────────────
QA_PAIRS, KB_META = load_knowledge_base()


# ── Helper utilities ───────────────────────────────────────────────────────
def find_by_intent(intent: str) -> dict | None:
    """Return the first QA pair that matches the given intent."""
    target = _normalise_text(intent).lower()
    return next((pair for pair in QA_PAIRS if pair.get("intent") == target), None)


def filter_by_tag(tag: str) -> list[dict]:
    """Return all QA pairs that include the given tag."""
    target = _normalise_text(tag).lower()
    return [pair for pair in QA_PAIRS if target in pair.get("tags", [])]


def filter_by_intent_prefix(prefix: str) -> list[dict]:
    """Return all QA pairs whose intent starts with the given prefix."""
    target = _normalise_text(prefix).lower()
    return [pair for pair in QA_PAIRS if pair.get("intent", "").startswith(target)]


def get_all_tags() -> list[str]:
    """Return a sorted list of all unique tags in the knowledge base."""
    tags: set[str] = set()
    for pair in QA_PAIRS:
        tags.update(pair.get("tags", []))
    return sorted(tags)


def get_all_sources() -> list[str]:
    """Return a sorted list of all unique source offices."""
    return sorted({pair.get("source", "Unknown") for pair in QA_PAIRS})


def get_domain_counts() -> dict[str, int]:
    """Return counts grouped by the intent prefix/domain."""
    return dict(Counter(pair.get("domain", "general") for pair in QA_PAIRS))


def get_featured_questions(limit: int = 8) -> list[dict]:
    """
    Return the highest-priority, student-facing questions for UI surfaces.
    Pairs with more guided steps are nudged upward because they are usually
    better quick-start workflows for users.
    """
    ranked = sorted(
        QA_PAIRS,
        key=lambda pair: (
            -pair.get("priority", 1),
            -len(pair.get("answer_steps", [])),
            pair.get("question", ""),
        ),
    )
    return ranked[: max(limit, 1)]


def get_featured_items(limit: int = 8) -> list[dict]:
    """Return compact featured question objects for UI surfaces."""
    return [
        {
            "question": pair.get("question", ""),
            "intent": pair.get("intent", ""),
            "source": pair.get("source", "Unknown"),
            "domain": pair.get("domain", "general"),
            "tags": pair.get("tags", [])[:3],
            "has_steps": bool(pair.get("answer_steps")),
        }
        for pair in get_featured_questions(limit)
    ]


def search_pairs(query: str, limit: int = 5) -> list[dict]:
    """
    Lightweight string search over the pre-built search text.
    This is useful for admin/debug workflows and future UI enhancements.
    """
    tokens = _normalise_string_list(re.findall(r"\w+", _normalise_text(query)), lowercase=True)
    if not tokens:
        return []

    ranked: list[tuple[int, dict]] = []
    for pair in QA_PAIRS:
        haystack = pair.get("search_text", "")
        score = sum(token in haystack for token in tokens)
        if score:
            ranked.append((score, pair))

    ranked.sort(
        key=lambda item: (
            -item[0],
            -item[1].get("priority", 1),
            item[1].get("question", ""),
        )
    )
    return [pair for _, pair in ranked[: max(limit, 1)]]


def get_recently_updated(limit: int = 10) -> list[dict]:
    """Return the most recently updated QA pairs."""
    ranked = sorted(
        QA_PAIRS,
        key=lambda pair: (
            pair.get("last_updated", "0000-00-00"),
            pair.get("priority", 1),
            pair.get("question", ""),
        ),
        reverse=True,
    )
    return ranked[: max(limit, 1)]


def get_stale_entries(cutoff_date: str = "2024-06-01") -> list[dict]:
    """
    Return QA pairs whose last_updated date is before cutoff_date.
    Malformed dates are treated as stale.
    """
    cutoff = date.fromisoformat(cutoff_date)
    stale: list[dict] = []
    for pair in QA_PAIRS:
        try:
            updated = date.fromisoformat(pair.get("last_updated", "2000-01-01"))
            if updated < cutoff:
                stale.append(pair)
        except ValueError:
            stale.append(pair)
    return stale


def get_step_answer(intent: str) -> list[str] | None:
    """Return guided steps for a given intent, if available."""
    pair = find_by_intent(intent)
    if pair and pair.get("answer_steps"):
        return pair["answer_steps"]
    return None


def reload_from_disk() -> list[dict]:
    """
    Re-read the JSON file from disk and update the module globals in place.
    Returns the fresh list so the caller can hand it to dependent components.
    """
    global QA_PAIRS, KB_META
    fresh_pairs, fresh_meta = load_knowledge_base()
    QA_PAIRS.clear()
    QA_PAIRS.extend(fresh_pairs)
    KB_META = fresh_meta
    log.info("QA_PAIRS reloaded in-place: %d pairs.", len(QA_PAIRS))
    return QA_PAIRS


def kb_summary() -> dict:
    """Return a compact summary payload for API/UI surfaces."""
    with_steps = sum(1 for pair in QA_PAIRS if pair.get("answer_steps"))
    return {
        "meta": dict(KB_META),
        "total_pairs": len(QA_PAIRS),
        "domains": get_domain_counts(),
        "with_steps": with_steps,
        "unique_tags": get_all_tags(),
        "unique_sources": get_all_sources(),
        "featured_questions": [pair["question"] for pair in get_featured_questions()],
        "featured_items": get_featured_items(),
        "stale_count": len(get_stale_entries()),
    }


def get_debug_snapshot(limit: int = 10) -> dict:
    """Return a debug-friendly snapshot of the current knowledge base."""
    return {
        "summary": kb_summary(),
        "recent_updates": [
            {
                "question": pair.get("question", ""),
                "intent": pair.get("intent", ""),
                "source": pair.get("source", "Unknown"),
                "last_updated": pair.get("last_updated", ""),
            }
            for pair in get_recently_updated(limit)
        ],
        "stale_entries": [
            {
                "question": pair.get("question", ""),
                "intent": pair.get("intent", ""),
                "source": pair.get("source", "Unknown"),
                "last_updated": pair.get("last_updated", ""),
            }
            for pair in get_stale_entries()[: max(limit, 1)]
        ],
    }
