"""Part 1: Chatbot Arena reanalysis.

Usage:
    python scripts/01_arena_idk.py                       # downloads from Hugging Face
    python scripts/01_arena_idk.py --csv path/to.csv     # local copy
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from idk.analysis import (add_features, audit_sample, bradley_terry,  # noqa: E402
                          idk_vs_answer, summarise_pairs, SHORT_ABSTAIN_WORDS)
from idk.arena import DEFAULT_DATASET, load_raw, tidy  # noqa: E402


def pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.dataset, args.csv)
    df = add_features(tidy(raw))

    class_counts = (df.class_a.value_counts() + df.class_b.value_counts()).fillna(0).astype(int).to_dict()
    pairs = idk_vs_answer(df)
    subsets = [
        summarise_pairs(pairs, "all IDK-vs-answer battles"),
        summarise_pairs(pairs[pairs.factual_prompt], "factual-looking prompts"),
        summarise_pairs(pairs[~pairs.factual_prompt], "other prompts"),
        summarise_pairs(pairs[pairs.idk_words <= SHORT_ABSTAIN_WORDS], f"bare IDK (<= {SHORT_ABSTAIN_WORDS} words)"),
        summarise_pairs(pairs[pairs.idk_words > SHORT_ABSTAIN_WORDS], f"IDK plus help (> {SHORT_ABSTAIN_WORDS} words)"),
    ]
    overall_tie = float((df.outcome == "tie").mean())
    bt = bradley_terry(df)
    bt_factual = bradley_terry(df[df.factual_prompt], min_battles=20)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": args.csv or args.dataset,
        "n_raw_battles": int(len(raw)),
        "n_single_turn": int(len(df)),
        "response_classes": class_counts,
        "overall_tie_rate": overall_tie,
        "idk_vs_answer": subsets,
        "bradley_terry_all": bt,
        "bradley_terry_factual": bt_factual,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    pairs.drop(columns=["outcome"]).to_csv(out / "idk_pairs.csv", index=False)
    audit_sample(df).to_csv(out / "audit_sample.csv", index=False)

    lines = [
        "# Part 1 results: Chatbot Arena reanalysis (rule-based first pass)",
        "",
        f"_Generated {summary['generated_utc']} from `{summary['dataset']}`._",
        "",
        f"- Battles: {len(raw):,} total, {len(df):,} single-turn.",
        f"- Responses classified: " + ", ".join(f"{k} {v:,}" for k, v in sorted(class_counts.items())),
        f"- Tie rate across all single-turn battles: {pct(overall_tie)}",
        "",
        "## When one model says \"I don't know\" and the other answers",
        "",
        "| Subset | Battles | IDK wins | Answer wins | Ties | P(IDK wins), decisive | 95% CI | Tie rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in subsets:
        lines.append(f"| {s['subset']} | {s['n_battles']:,} | {s['idk_wins']:,} | {s['answer_wins']:,} | "
                     f"{s['ties']:,} | {pct(s['p_idk_wins_decisive'])} | "
                     f"{pct(s['ci_low'])} to {pct(s['ci_high'])} | {pct(s['tie_rate'])} |")
    lines += ["", "## Controlling for model strength, length and hedging (Bradley-Terry logit)", ""]
    for name, res in (("All prompts", bt), ("Factual-looking prompts", bt_factual)):
        if "error" in res:
            lines.append(f"- {name}: not estimable ({res['error']}).")
            continue
        t = res["idk_diff"]
        h = res["hedge_diff"]
        lines.append(f"- **{name}** ({res['n_battles']:,} decisive battles, {res['n_models']} models): "
                     f"abstaining multiplies the odds of winning by {t['odds_ratio']:.2f} "
                     f"(95% CI {t['or_ci_low']:.2f} to {t['or_ci_high']:.2f}); "
                     f"each extra hedge per 100 words: OR {h['odds_ratio']:.2f} "
                     f"({h['or_ci_low']:.2f} to {h['or_ci_high']:.2f}).")
    lines += [
        "",
        "## Caveats",
        "",
        "- Labels come from a transparent phrase list (`src/idk/hedges.py`). Precision is checked by hand "
        "on `audit_sample.csv` and will be double-checked by an LLM classifier.",
        "- Correctness is unknown here: an \"answer\" may be wrong and an \"I don't know\" may be appropriate. "
        "Part 2 (the controlled experiment) addresses this.",
        "- Arena voters are self-selected and the prompts are whatever they chose to ask.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
