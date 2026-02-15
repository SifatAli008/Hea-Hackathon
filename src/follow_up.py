"""
Empathetic follow-up question generator. No diagnosis, no treatment; supportive tone.
Rules: consider life events (job loss, retirement, divorce, stress).
"""
import random
from typing import List, Optional


# Templates by dominant signal type. Use {main_change} for personalized insertion (e.g. "Activity Level").
# More variety per topic; mix of short and warm, no diagnosis.
TEMPLATES = {
    "mood": [
        "We noticed some changes in your mood ratings recently. Would you like to share if anything stressful has been happening?",
        "Your recent responses suggest some shifts in how you've been feeling. Is there anything going on that you'd like to talk about?",
        "It looks like your mood has been shifting a bit lately. Would it help to talk about what's been on your mind?",
        "We noticed changes in how you've been feeling. Sometimes it helps to put it into words — would you like to share?",
        "Your {main_change} has shown some changes recently. Would you like to share if anything has been affecting how you feel?",
    ],
    "stress": [
        "We noticed some changes in your stress levels lately. Would you like to share if anything has been weighing on you?",
        "It looks like stress might have increased recently. Would you like to share what's been on your mind?",
        "Your stress levels seem to have shifted. Is there anything going on that you'd find helpful to talk about?",
        "We noticed your {main_change} has changed recently. Would you like to share what might be behind that?",
        "Sometimes stress shows up in small changes first. Would you like to share if anything has felt heavier lately?",
    ],
    "life_events": [
        "We noticed some changes that sometimes go with life transitions (work, relationships, or stress). Would you like to share if anything has shifted recently?",
        "Your responses suggest things may have been different lately. Would you like to share if there have been any big changes or stresses we should know about?",
        "Life changes — like work, relationships, or moving — can show up in how we feel. Would you like to share if anything has changed for you recently?",
        "We noticed patterns that often go with life transitions. Would it help to talk about what's been different lately?",
        "Big or small life changes can affect how we rate our health. Would you like to share if anything has shifted for you?",
    ],
    "sleep": [
        "We noticed some recent changes in your sleep pattern. Has anything been affecting your rest lately?",
        "Your sleep seems to have changed recently. Would you like to share if your routine or environment has changed?",
        "Sleep can shift for many reasons. Would you like to share if anything has been different with your rest or routine?",
        "We noticed your {main_change} has shifted. Would you like to share if anything has been affecting your sleep?",
        "Rest and routine often go together. Is there anything you'd like to share about how your sleep has been lately?",
    ],
    "activity": [
        "Your activity level has changed recently. Would you like to share if your routine has changed?",
        "We noticed some changes in your activity. Is there anything that has made it harder to stay active lately?",
        "Your {main_change} has shifted a bit lately. Would you like to share what might have changed in your routine?",
        "Activity can go up or down for many reasons. Would you like to share if something has made it easier or harder to stay active?",
        "We noticed changes in your activity pattern. Is there anything you'd like to share about what's been different?",
    ],
    "health_rating": [
        "We noticed some changes in how you've been rating your health recently. Would you like to share if anything has been different?",
        "Your recent health ratings have shifted a bit. Is there anything you'd like to share about how you've been feeling?",
        "Your {main_change} has shown some change lately. Would you like to share what might be behind that?",
        "How we rate our health can reflect a lot of things. Would you like to share if anything has felt different recently?",
        "We noticed a shift in how you've been rating your health. Would it help to talk about what's been on your mind?",
    ],
    "general": [
        "We noticed some changes in your recent health responses. Have there been any new stresses or lifestyle changes you'd like to share?",
        "Your responses have shown some changes lately. Would you like to share if anything has been going on?",
        "We noticed a few things have shifted in your responses. Would you like to share what might be behind that?",
        "Sometimes small changes add up. Would you like to share if anything has been different for you lately?",
        "Your {main_change} and other areas have shown some changes. Would you like to share if anything has been on your mind?",
    ],
}


def _fill_main_change(template: str, main_change_name: Optional[str]) -> str:
    """Replace {main_change} with the main change name or a friendly fallback."""
    if "{main_change}" not in template:
        return template
    label = (main_change_name or "health responses").replace("_", " ").strip()
    if label and not label[0].isupper():
        label = label.title()
    return template.replace("{main_change}", label)


def pick_follow_up(
    top_contributors: List[str],
    category: str,
    risk_score: float,
    main_change_names: Optional[List[str]] = None,
    main_change_name: Optional[str] = None,
) -> str:
    """
    Choose one empathetic follow-up question. Prefer main_change_names (order of change in explanation)
    so the question matches "why we flagged"; else use top_contributors.
    When main_change_name is provided, templates with {main_change} are personalized (e.g. "Activity Level").
    """
    key = "general"
    # Prefer order of change in explanation so follow-up aligns with "Main changes we observed"
    order = (main_change_names or []) + [n for n in top_contributors if n not in (main_change_names or [])]
    for name in order:
        name_lower = str(name).lower()
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
    chosen = rng.choice(templates)
    # Personalize: fill {main_change} with first main change name if available
    fill_name = main_change_name or (main_change_names[0] if main_change_names else None)
    return _fill_main_change(chosen, fill_name)
