import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from idk.hedges import classify, hedge_density  # noqa: E402


def test_abstentions():
    for t in [
        "I don't know who won that game.",
        "I'm not sure about the exact date, but you could check the city website.",
        "I do not have real-time information about stock prices.",
        "As of my last knowledge update in 2021, I cannot verify that.",
        "I couldn't find any reliable information about that person.",
    ]:
        assert classify(t) == "abstain", t


def test_safety_refusals_are_not_abstentions():
    for t in [
        "I'm sorry, but I can't help with that request.",
        "I cannot assist with that as it would be harmful.",
        "That request goes against my guidelines.",
    ]:
        assert classify(t) == "safety_refusal", t


def test_answers():
    for t in [
        "The capital of Australia is Canberra.",
        "Here is a Python function that reverses a list: ...",
        "Paris has been the capital of France since the 10th century. " * 10
        + "I don't know of any earlier capital.",  # late caveat, not a lead abstention
    ]:
        assert classify(t) == "answer", t


def test_hedge_density():
    assert hedge_density("It is probably Canberra, I think.") > 0
    assert hedge_density("It is Canberra.") == 0
    assert hedge_density("") == 0
    assert hedge_density("Born in May 1990.") == 0  # month name is not a hedge


def test_clarification_requests():
    for t in [
        "I'm sorry, I'm not sure what you are referring to. Could you provide more context?",
        "Sorry, but I'm not sure I understand what you are saying. Could you explain?",
    ]:
        assert classify(t) == "clarify", t
