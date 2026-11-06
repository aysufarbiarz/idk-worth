"""Load the Chatbot Arena human-preference data into a tidy, one-row-per-battle frame."""
from __future__ import annotations

import json

import pandas as pd

DEFAULT_DATASET = "lmarena-ai/arena-human-preference-55k"


def _parse_turns(value) -> list[str]:
    """Arena stores conversations as JSON-encoded lists of strings (one per turn)."""
    if isinstance(value, list):
        return [v if isinstance(v, str) else "" for v in value]
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return [value]
    if isinstance(parsed, list):
        return [v if isinstance(v, str) else "" for v in parsed]
    return [str(parsed)]


def load_raw(dataset: str = DEFAULT_DATASET, csv_path: str | None = None) -> pd.DataFrame:
    if csv_path:
        return pd.read_csv(csv_path)
    from datasets import load_dataset  # imported lazily so tests don't need it

    return load_dataset(dataset, split="train").to_pandas()


def tidy(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep single-turn battles with a clear outcome label.

    Returns columns: id, model_a, model_b, prompt, response_a, response_b,
    outcome in {'a', 'b', 'tie'}.
    """
    expected = {"id", "model_a", "model_b", "prompt", "response_a", "response_b",
                "winner_model_a", "winner_model_b", "winner_tie"}
    missing = expected - set(raw.columns)
    if missing:
        raise ValueError(f"Unexpected schema; missing columns {sorted(missing)}. "
                         f"Found: {list(raw.columns)}")

    df = raw.copy()
    for col in ("prompt", "response_a", "response_b"):
        df[col + "_turns"] = df[col].map(_parse_turns)
    n_turns = df["prompt_turns"].map(len)
    single = (n_turns == 1) & (df["response_a_turns"].map(len) == 1) & (df["response_b_turns"].map(len) == 1)
    df = df[single].copy()

    df["prompt"] = df["prompt_turns"].str[0]
    df["response_a"] = df["response_a_turns"].str[0]
    df["response_b"] = df["response_b_turns"].str[0]

    def outcome(r):
        if r.winner_model_a == 1:
            return "a"
        if r.winner_model_b == 1:
            return "b"
        return "tie"

    df["outcome"] = df.apply(outcome, axis=1)
    return df[["id", "model_a", "model_b", "prompt", "response_a", "response_b", "outcome"]].reset_index(drop=True)
