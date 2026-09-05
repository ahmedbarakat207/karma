"""Bilingual movement voice intents (EN + Egyptian Arabic).

Returns (action, param) where action is one of:
stop, forward, backward, left, right, follow, unfollow,
explore, stop_explore, goto, dock. Param carries beacon/duration.
Mirrors _check_kiosk_intent style: substring triggers, no LLM needed.
"""

import re
from typing import Optional, Tuple

from src.navigation.explorer import BEACONS

_BEACON_ALIASES = {
    "desk": ("desk", "مكتب", "المكتب", "الديسك"),
    "kitchen": ("kitchen", "مطبخ", "المطبخ"),
    "door": ("door", "باب", "الباب", "entrance"),
    "charger": ("charger", "charge", "dock", "شاحن", "الشاحن", "اشحن"),
    "window": ("window", "شباك", "الشباك", "نافذة"),
}


def _match_beacon(t: str) -> Optional[str]:
    for beacon, aliases in _BEACON_ALIASES.items():
        if any(a in t for a in aliases):
            return beacon
    return None


def check_move_intent(text: str) -> Optional[Tuple[str, str]]:
    t = (text or "").lower().strip()
    if not t:
        return None

    # Stop first: safety-critical, matches many phrasings.
    if any(p in t for p in [
        "stop moving", "stop right there", "halt", "freeze", "don't move",
        "stay here", "stay put", "stop the robot", "emergency stop",
        "اقف", "قف", "اوقف", "أوقف", "استنى", "اثبت", "ما تتحركش", "بطل حركة",
    ]):
        return ("stop", "")

    if any(p in t for p in [
        "stop exploring", "stop wandering", "come back", "that's enough exploring",
        "بطل استكشاف", "كفاية لف", "ارجع مكانك",
    ]):
        return ("stop_explore", "")

    if any(p in t for p in [
        "stop following", "don't follow", "unfollow", "stop that",
        "بطل تتبع", "ما تتبعنيش", "كفاية متابعة",
    ]):
        return ("unfollow", "")

    if any(p in t for p in [
        "follow me", "come with me", "walk with me", "come here", "come to me",
        "اتبعني", "تعال هنا", "تعالي هنا", "امشي معايا", "الحقني",
    ]):
        return ("follow", "")

    if any(p in t for p in [
        "explore", "wander", "look around", "go explore", "roam",
        "استكشف", "اتجول", "لف في الأوضة", "لف في الاوضة", "بص حواليك",
    ]):
        return ("explore", "")

    beacon = _match_beacon(t)
    if beacon and any(p in t for p in [
        "go to", "drive to", "move to", "roll to", "take me to", "navigate",
        "روح", "اتحرك", "امشي لـ", "امشي ل", "وديني",
    ]):
        return ("goto", beacon)

    if any(p in t for p in [
        "go forward", "move forward", "come forward", "drive forward", "go straight",
        "امشي قدام", "امشي لقدام", "قدام", "اتقدم",
    ]):
        return ("forward", "")

    if any(p in t for p in [
        "go back", "move back", "back up", "reverse", "drive back",
        "ارجع ورا", "امشي ورا", "ورا", "ارجع",
    ]):
        # "ارجع مكانك" already handled as stop_explore above.
        return ("backward", "")

    # Bare "left"/"I left my keys" guard: single-word directions only count
    # in short utterances or next to a movement verb.
    words = t.split()
    has_move_verb = any(v in t for v in [
        "turn", "go", "move", "drive", "roll", "veer", "لف", "امشي", "اتحرك", "روح",
    ])
    if any(p in t for p in ["turn left", "go left", "move left", "لف شمال", "امشي شمال"]) \
            or (("شمال" in t or re.search(r"\bleft\b", t)) and (len(words) <= 2 or has_move_verb)):
        # "I left my keys on the table" must not steer the robot.
        if "i left" in t and not has_move_verb:
            pass
        else:
            return ("left", "")

    if any(p in t for p in ["turn right", "go right", "move right", "لف يمين", "امشي يمين"]) \
            or (("يمين" in t or re.search(r"\bright\b", t)) and (len(words) <= 2 or has_move_verb)):
        return ("right", "")

    if any(p in t for p in ["dock", "go charge", "charge yourself"]):
        return ("goto", "charger")

    return None


def beacon_list() -> Tuple[str, ...]:
    return BEACONS
