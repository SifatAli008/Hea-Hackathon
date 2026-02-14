"""
Empathetic follow-up question generator. No diagnosis, no treatment; supportive tone.
Rules: consider life events (job loss, retirement, divorce, stress).
"""
import random
from typing import List, Optional


# Templates by dominant signal type (psycho-emotional, metabolic, general). Rules: life events.
TEMPLATES = {
    "mood": [
        "We noticed some changes in your mood ratings recently. Would you like to share if anything stressful has been happening?",
        "Your recent responses suggest some shifts in how you've been feeling. Is there anything going on that you'd like to talk about?",
    ],
    "stress": [
        "We noticed some changes in your stress levels lately. Would you like to share if anything has been weighing on you?",
        "It looks like stress might have increased recently. Would you like to share what's been on your mind?",
    ],
    "life_events": [
        "We noticed some changes that sometimes go with life transitions (work, relationships, or stress). Would you like to share if anything has shifted recently?",
        "Your responses suggest things may have been different lately. Would you like to share if there have been any big changes or stresses we should know about?",
    ],
    "sleep": [
        "We noticed some recent changes in your sleep pattern. Has anything been affecting your rest lately?",
        "Your sleep seems to have changed recently. Would you like to share if your routine or environment has changed?",
    ],
    "activity": [
        "Your activity level has changed recently. Would you like to share if your routine has changed?",
        "We noticed some changes in your activity. Is there anything that has made it harder to stay active lately?",
    ],
    "health_rating": [
        "We noticed some changes in how you've been rating your health recently. Would you like to share if anything has been different?",
        "Your recent health ratings have shifted a bit. Is there anything you'd like to share about how you've been feeling?",
    ],
    "general": [
        "We noticed some changes in your recent health responses. Have there been any new stresses or lifestyle changes you'd like to share?",
        "Your responses have shown some changes lately. Would you like to share if anything has been going on?",
    ],
}


def pick_follow_up(
    top_contributors: List[str],
    category: str,
    risk_score: float,
) -> str:
    """
    Choose one empathetic follow-up question based on top contributing features and risk category.
    Rules: life events (job loss, retirement, divorce, stress). Varies by template choice.
    """
    key = "general"
    for name in top_contributors:
        name_lower = name.lower()
        if "mood" in name_lower:
            key = "mood"
            break
        if "stress" in name_lower:
            key = "stress"
            break
        if "life_event" in name_lower:
            key = "life_events"
            break
        if "sleep" in name_lower:
            key = "sleep"
            break
        if "activity" in name_lower:
            key = "activity"
            break
        if "health" in name_lower or "rating" in name_lower:
            key = "health_rating"
            break
    # Psycho-emotional category: prefer life-events template (Rules: job loss, retirement, divorce, stress)
    if category and "psycho" in category.lower() and key == "general":
        key = "life_events"
    templates = TEMPLATES.get(key, TEMPLATES["general"])
    # Reproducible variety: seed from risk_score so same person gets same question
    rng = random.Random(int(risk_score * 10) % (2 ** 32))
    return rng.choice(templates)
