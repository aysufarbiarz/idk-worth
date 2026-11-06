"""Build a small synthetic Arena-format CSV so the pipeline can be tested offline.

The synthetic data has a built-in IDK penalty, so the analysis should recover
an odds ratio below 1. It contains no real conversations.
"""
import json
import random
import sys

import pandas as pd

random.seed(0)
MODELS = [f"model_{i}" for i in range(8)]
STRENGTH = {m: i * 0.15 for i, m in enumerate(MODELS)}
ANSWERS = ["The answer is {x}. It was established in {y}.", "Sure! {x} is correct; see {y} for details."]
IDKS = ["I don't know the answer to that, but you could check {y}.",
        "I'm not sure about {x}; I do not have reliable information on it."]
REFUSALS = ["I'm sorry, but I can't help with that request."]

rows = []
for i in range(4000):
    a, b = random.sample(MODELS, 2)
    def resp(m):
        r = random.random()
        if r < 0.08:
            return random.choice(IDKS), 1
        if r < 0.11:
            return random.choice(REFUSALS), 0
        return random.choice(ANSWERS) + " Extra detail." * random.randint(0, 20), 0
    ra, ia = resp(a)
    rb, ib = resp(b)
    import math
    logit = STRENGTH[a] - STRENGTH[b] - 0.8 * (ia - ib) + 0.05
    pa = 1 / (1 + math.exp(-logit))
    u = random.random()
    tie = u < 0.25
    win_a = (not tie) and random.random() < pa
    q = random.choice(["Who founded {x}?", "Write a poem about {x}", "When was {x} built?"])
    rows.append({
        "id": i, "model_a": a, "model_b": b,
        "prompt": json.dumps([q.format(x="X")]),
        "response_a": json.dumps([ra.format(x="X", y="Y")]),
        "response_b": json.dumps([rb.format(x="X", y="Y")]),
        "winner_model_a": int(win_a), "winner_model_b": int((not tie) and not win_a), "winner_tie": int(tie),
    })
pd.DataFrame(rows).to_csv(sys.argv[1] if len(sys.argv) > 1 else "synthetic.csv", index=False)
