from __future__ import annotations

import re
from typing import Dict, Iterable, Tuple


EMOTIONS = ["happy", "sad", "dark", "excited", "calm", "hopeful"]


_EMOTION_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "happy": (
        "happy",
        "joy",
        "fun",
        "funny",
        "smile",
        "uplifting",
        "cheerful",
        "delight",
        "warm",
        "playful",
    ),
    "sad": (
        "sad",
        "cry",
        "tears",
        "lonely",
        "grief",
        "heartbreak",
        "tragic",
        "pain",
        "sorrow",
        "melancholy",
    ),
    "dark": (
        "dark",
        "bleak",
        "grim",
        "violent",
        "haunting",
        "murder",
        "death",
        "fear",
        "evil",
        "twisted",
    ),
    "excited": (
        "excited",
        "thrilling",
        "intense",
        "fast",
        "action",
        "adrenaline",
        "energetic",
        "wild",
        "epic",
        "powerful",
    ),
    "calm": (
        "calm",
        "peaceful",
        "gentle",
        "quiet",
        "soft",
        "relaxing",
        "slow",
        "soothing",
        "still",
        "tender",
    ),
    "hopeful": (
        "hope",
        "hopeful",
        "inspiring",
        "believe",
        "faith",
        "healing",
        "uplift",
        "bright",
        "redeeming",
        "optimistic",
    ),
}


def _normalize_text(text: str) -> str:
    """Lowercase and simplify punctuation for keyword matching."""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


def _count_keyword_hits(text: str, keywords: Iterable[str]) -> int:
    """Count keyword appearances using whole-word matching where possible."""
    total = 0
    for keyword in keywords:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        total += len(re.findall(pattern, text))
    return total


def extract_emotion_scores(review_text: str) -> Dict[str, float]:
    """
    Extract normalized emotion scores from user movie reviews using keyword matching.

    Args:
        review_text: Raw input string containing one or more movie reviews.

    Returns:
        Dictionary like {"happy": float, "sad": float, ...} with values summing to 1.0
        when any keywords are found, otherwise all values are 0.0.
    """
    normalized_text = _normalize_text(review_text if isinstance(review_text, str) else "")

    raw_scores: Dict[str, int] = {
        emotion: _count_keyword_hits(normalized_text, _EMOTION_KEYWORDS[emotion])
        for emotion in EMOTIONS
    }

    total_hits = sum(raw_scores.values())
    if total_hits == 0:
        return {emotion: 0.0 for emotion in EMOTIONS}

    return {
        emotion: raw_scores[emotion] / total_hits
        for emotion in EMOTIONS
    }
