from __future__ import annotations
import random
from typing import Dict, Iterable, Tuple


EMOTIONS = ["happy", "sad", "dark", "excited", "calm", "hopeful"]


_EMOTION_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "happy": (
        "happy",
        "joy",
        "smile",
        "sunshine",
        "fun",
        "party",
        "dance",
        "good",
        "bright",
        "laugh",
    ),
    "sad": (
        "sad",
        "cry",
        "tears",
        "lonely",
        "alone",
        "heartbreak",
        "broken",
        "blue",
        "pain",
        "goodbye",
    ),
    "dark": (
        "dark",
        "shadow",
        "night",
        "black",
        "devil",
        "hell",
        "grave",
        "blood",
        "fear",
        "evil",
    ),
    "excited": (
        "excited",
        "fire",
        "wild",
        "run",
        "fast",
        "electric",
        "boom",
        "rock",
        "rush",
        "energy",
    ),
    "calm": (
        "calm",
        "peace",
        "quiet",
        "soft",
        "dream",
        "breeze",
        "ocean",
        "moon",
        "gentle",
        "still",
    ),
    "hopeful": (
        "hope",
        "light",
        "rise",
        "tomorrow",
        "believe",
        "faith",
        "home",
        "heaven",
        "golden",
        "forever",
    ),
}


def _count_matches(title: str, keywords: Iterable[str]) -> int:
    words = title.split()
    return sum(1 for keyword in keywords if keyword in words)


def _infer_emotion_from_title(title: str) -> str:
    title = title.lower()
    
    scores = {emotion: 0 for emotion in EMOTIONS}

    keyword_map = {
        "sad": ["love", "heart", "cry", "alone", "broken", "goodbye", "years"],
        "excited": ["rock", "roll", "fire", "dance", "energy", "music"],
        "dark": ["dark", "night", "shadow", "black"],
        "calm": ["dream", "peace", "calm", "soft", "slow", "jazz", "live"],
        "hopeful": ["hope", "rise", "light", "promise", "land"],
        "happy": ["fun", "joy", "smile", "good", "bright"]
    }

    for emotion, keywords in keyword_map.items():
        for word in keywords:
            if word in title:
                scores[emotion] += 1

    best_emotion = max(scores, key=scores.get)

    # fallback
    if scores[best_emotion] == 0:
        return "happy"

    return best_emotion


def build_song_emotion_map(metadata_map: Dict[str, str]) -> Dict[str, str]:
    """
    Build an ASIN -> emotion mapping from an ASIN -> song title mapping.

    Args:
        metadata_map: Dictionary where key is the ASIN and value is the song title.

    Returns:
        Dictionary of {asin: emotion}.
    """
    emotion_map: Dict[str, str] = {}

    for asin, song_name in metadata_map.items():
        title = song_name if isinstance(song_name, str) else ""
        emotion_map[asin] = _infer_emotion_from_title(title)

    return emotion_map
