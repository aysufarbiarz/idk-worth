"""Part 1 analysis: how do people vote when one chatbot says "I don't know" and the other answers?"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .hedges import classify, hedge_density, word_count

FACTUAL_RE = re.compile(
    r"^\s*(?:who|what|when|where|which|whose|how (?:many|much|old|long|far|tall|big)|"
    r"is|was|were|did|does|do|are|has|have|can you tell me)\b",
    re.IGNORECASE,
)
SHORT_ABSTAIN_WORDS = 60  # "bare" abstention vs. abstention plus help


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, centre - half, centre + half


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for side in ("a", "b"):
        text = df[f"response_{side}"].fillna("")
        df[f"class_{side}"] = text.map(classify)
        df[f"hedge_{side}"] = text.map(hedge_density)
        df[f"words_{side}"] = text.map(word_count)
    df["factual_prompt"] = df["prompt"].fillna("").map(
        lambda p: bool(FACTUAL_RE.match(p)) and word_count(p) <= 40)
    return df


def idk_vs_answer(df: pd.DataFrame) -> pd.DataFrame:
    """Battles where exactly one side abstains and the other gives a normal answer."""
    a_idk = (df.class_a == "abstain") & (df.class_b == "answer")
    b_idk = (df.class_b == "abstain") & (df.class_a == "answer")
    pairs = df[a_idk | b_idk].copy()
    pairs["idk_side"] = np.where(a_idk[a_idk | b_idk], "a", "b")
    pairs["idk_model"] = np.where(pairs.idk_side == "a", pairs.model_a, pairs.model_b)
    pairs["answer_model"] = np.where(pairs.idk_side == "a", pairs.model_b, pairs.model_a)
    pairs["idk_words"] = np.where(pairs.idk_side == "a", pairs.words_a, pairs.words_b)
    pairs["result"] = np.select(
        [pairs.outcome == "tie", pairs.outcome == pairs.idk_side],
        ["tie", "idk_wins"], default="answer_wins")
    return pairs


def summarise_pairs(pairs: pd.DataFrame, label: str) -> dict:
    n = len(pairs)
    idk = int((pairs.result == "idk_wins").sum())
    ans = int((pairs.result == "answer_wins").sum())
    tie = int((pairs.result == "tie").sum())
    p, lo, hi = wilson(idk, idk + ans)
    return {"subset": label, "n_battles": n, "idk_wins": idk, "answer_wins": ans, "ties": tie,
            "p_idk_wins_decisive": p, "ci_low": lo, "ci_high": hi,
            "tie_rate": tie / n if n else float("nan")}


def bradley_terry(df: pd.DataFrame, min_battles: int = 50) -> dict:
    """Logistic Bradley-Terry model of P(A wins) on decisive battles.

    logit P(A wins) = position + sum_m theta_m (x_A - x_B)
                      + beta_idk * (idk_A - idk_B)
                      + beta_hedge * (hedge_A - hedge_B)
                      + beta_len * (log words_A - log words_B)

    Model-strength terms absorb "weaker models abstain more", and the length
    term absorbs the known verbosity preference, so beta_idk is the penalty
    (or reward) for abstaining *holding model and length constant*.
    """
    d = df[(df.outcome != "tie")
           & df.class_a.isin(["answer", "abstain"]) & df.class_b.isin(["answer", "abstain"])].copy()
    counts = pd.concat([d.model_a, d.model_b]).value_counts()
    keep = set(counts[counts >= min_battles].index)
    d = d[d.model_a.isin(keep) & d.model_b.isin(keep)]
    if d.empty:
        return {"error": "no data"}
    models = sorted(keep)
    ref = counts[list(keep)].idxmax()  # most common model is the reference
    X = pd.DataFrame(0.0, index=d.index, columns=[m for m in models if m != ref])
    for m in X.columns:
        X[m] = (d.model_a == m).astype(float) - (d.model_b == m).astype(float)
    X["idk_diff"] = (d.class_a == "abstain").astype(float) - (d.class_b == "abstain").astype(float)
    X["hedge_diff"] = d.hedge_a - d.hedge_b
    X["loglen_diff"] = np.log1p(d.words_a) - np.log1p(d.words_b)
    X = sm.add_constant(X)  # constant = position (A vs B) bias
    y = (d.outcome == "a").astype(float)
    fit = sm.Logit(y, X).fit(disp=False, maxiter=200, cov_type="HC1")
    out = {"n_battles": int(len(d)), "n_models": len(models), "reference_model": ref}
    for term in ("idk_diff", "hedge_diff", "loglen_diff", "const"):
        b = fit.params[term]
        lo, hi = fit.conf_int().loc[term]
        out[term] = {"coef": float(b), "ci_low": float(lo), "ci_high": float(hi),
                     "odds_ratio": float(np.exp(b)), "or_ci_low": float(np.exp(lo)),
                     "or_ci_high": float(np.exp(hi)), "p_value": float(fit.pvalues[term])}
    return out


def audit_sample(df: pd.DataFrame, n_flagged: int = 60, n_other: int = 40, seed: int = 7) -> pd.DataFrame:
    """Random responses for hand-labelling, to check the rule-based classifier."""
    long = pd.concat([
        df[["id", "prompt", "response_a", "class_a"]].rename(columns={"response_a": "response", "class_a": "rule_label"}),
        df[["id", "prompt", "response_b", "class_b"]].rename(columns={"response_b": "response", "class_b": "rule_label"}),
    ])
    flagged = long[long.rule_label == "abstain"]
    other = long[long.rule_label == "answer"]
    sample = pd.concat([
        flagged.sample(min(n_flagged, len(flagged)), random_state=seed),
        other.sample(min(n_other, len(other)), random_state=seed),
    ]).sample(frac=1, random_state=seed)  # shuffle so the labeller can't tell which is which
    sample = sample.rename(columns={"rule_label": "rule_label_hidden"})
    sample["human_label (abstain/answer/safety_refusal)"] = ""
    return sample
