from __future__ import annotations

from typing import Dict, List


def _safe_model_score(recommendation: Dict[str, str]) -> float:
    """Read the backend rating field as a float, defaulting to 0.0."""
    try:
        return float(recommendation.get("rating", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _top_user_emotion(emotion_scores: Dict[str, float]) -> str | None:
    """Return the highest-scoring user emotion, or None if unavailable."""
    if not emotion_scores:
        return None
    return max(emotion_scores, key=emotion_scores.get)


def rerank_recommendations(
    recommendations: List[Dict[str, str]],
    emotion_scores: Dict[str, float],
    song_emotion_map: Dict[str, str],
) -> List[Dict[str, str]]:

    scored_recommendations = []

    for index, recommendation in enumerate(recommendations):
        asin = recommendation.get("asin")
        model_score = _safe_model_score(recommendation)

        # safer default
        song_emotion = song_emotion_map.get(asin, "happy")

        # 🔥 USE FULL EMOTION DISTRIBUTION
        emotion_match = emotion_scores.get(song_emotion, 0.0)

        # 🔥 STRONG BOOST
        final_score = model_score + (3.0 * emotion_match)

        scored_recommendations.append((final_score, index, recommendation))

    scored_recommendations.sort(key=lambda item: (-item[0], item[1]))

    return [recommendation for _, _, recommendation in scored_recommendations]