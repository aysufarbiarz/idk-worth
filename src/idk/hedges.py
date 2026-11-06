"""Rule-based detection of "I don't know"-style abstentions, safety refusals and hedging.

This is a deliberately transparent first pass. Every pattern is listed here so a
reader can audit it, and the pipeline writes a random sample of flagged
responses to results/audit_sample.csv for hand-labelling (precision check).
A later step uses an LLM classifier to double-check these labels.
"""
from __future__ import annotations

import re

# --- Knowledge-based abstention ("I don't know / can't verify") -------------
# Patterns describe a lack of knowledge, information or access -- NOT a refusal
# on safety or policy grounds.
_ABSTAIN = [
    r"\bI (?:don'?t|do not) know\b",
    r"\bI(?:'m| am) not (?:sure|certain|aware)\b",
    r"\bI(?:'m| am) (?:unsure|uncertain)\b",
    r"\bI (?:don'?t|do not) have (?:any |enough |sufficient |specific |reliable |accurate |"
    r"up-to-date |real-time |current |detailed |personal |direct )*"
    r"(?:information|data|knowledge|details|access|records)\b",
    r"\bI (?:can'?t|cannot|am unable to|'m unable to|am not able to|'m not able to) "
    r"(?:verify|confirm|determine|find|access|browse|know|look up|check)\b",
    r"\bI (?:couldn'?t|could not|was unable to|wasn'?t able to) find\b",
    r"\b(?:my|the) (?:knowledge|training data|training) (?:cutoff|cut-off|only goes|is limited|extends only)\b",
    r"\bas of my (?:last |latest )?(?:knowledge|training)(?: data)? (?:update|cutoff|cut-off)\b",
    r"\b(?:there is|there's|there are) no (?:reliable|available|publicly available|public|known|widely known) "
    r"(?:information|data|records|sources)\b",
    r"\bI(?:'m| am) not able to (?:provide|give) (?:real-time|current|up-to-date|live)\b",
    r"\bI (?:do not|don'?t) have (?:the ability|the capability) to (?:browse|access)\b",
]

# --- Safety / policy refusals (a different construct; excluded) --------------
_SAFETY = [
    r"\b(?:harmful|unethical|inappropriate|offensive)\b",
    r"\bagainst (?:my|our|the|OpenAI'?s?) (?:guidelines|policies|policy|principles|programming)\b",
    r"\bnot (?:appropriate|ethical)\b",
    r"\bviolat\w* (?:\w+ )?(?:policy|policies|guidelines)\b",
    r"\bI (?:can'?t|cannot|won'?t|will not) (?:help|assist) with (?:that|this)\b",
    r"\bI(?:'m| am) (?:sorry|afraid),? (?:but )?I (?:can'?t|cannot|won'?t) "
    r"(?:help|assist|fulfill|comply|create|generate|write|engage)\b",
]

# --- Clarification requests (the model didn't understand the prompt; excluded) ---
_CLARIFY = [
    r"\bnot sure (?:what|I understand|if I understand|I follow)\b",
    r"\b(?:don'?t|do not) understand\b",
    r"\b(?:provide|give|share) (?:me )?(?:more|some|additional) (?:context|details|information)\b",
    r"\bcould you (?:please )?(?:clarify|explain|elaborate|rephrase)\b",
]

# --- Hedge markers (density per 100 words) -----------------------------------
# Case-sensitive "may"/"might" avoid the month name and proper nouns.
_HEDGES = [
    r"\bprobably\b", r"\bpossibly\b", r"\bperhaps\b", r"\blikely\b", r"\bunlikely\b",
    r"\bmight\b", r"\bmay\b", r"\bcould be\b", r"\bit seems\b", r"\bseems to\b",
    r"\bappears to\b", r"\bI think\b", r"\bI believe\b", r"\bI suspect\b",
    r"\bnot (?:entirely |completely |100% )?(?:sure|certain)\b", r"\buncertain\b",
    r"\bto (?:the best of )?my knowledge\b", r"\bif I recall\b", r"\bI'?m guessing\b",
]

ABSTAIN_RE = re.compile("|".join(_ABSTAIN), re.IGNORECASE)
SAFETY_RE = re.compile("|".join(_SAFETY), re.IGNORECASE)
CLARIFY_RE = re.compile("|".join(_CLARIFY), re.IGNORECASE)
HEDGE_RE = re.compile("|".join(_HEDGES))  # case-sensitive on purpose

# An abstention only counts if it appears early: the response *leads* with it,
# rather than adding a knowledge-cutoff footnote to a confident answer.
LEAD_CHARS = 300


def word_count(text: str) -> int:
    return len(text.split())


def is_safety_refusal(text: str) -> bool:
    return bool(SAFETY_RE.search(text[:LEAD_CHARS]))


def leads_with_abstention(text: str) -> bool:
    return bool(ABSTAIN_RE.search(text[:LEAD_CHARS]))


def hedge_density(text: str) -> float:
    """Hedge markers per 100 words."""
    n = word_count(text)
    if n == 0:
        return 0.0
    return 100.0 * len(HEDGE_RE.findall(text)) / n


def classify(text: str) -> str:
    """Return 'safety_refusal', 'clarify', 'abstain' or 'answer'.

    Safety refusals win ties: a response that says "I can't help with that,
    it's harmful" is a policy refusal, not an admission of ignorance.
    """
    if not text or not text.strip():
        return "empty"
    if is_safety_refusal(text):
        return "safety_refusal"
    if CLARIFY_RE.search(text[:LEAD_CHARS]):
        return "clarify"
    if leads_with_abstention(text):
        return "abstain"
    return "answer"
