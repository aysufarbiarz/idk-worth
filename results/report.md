# Part 1 results: Chatbot Arena reanalysis (rule-based first pass)

_Generated 2026-11-06T23:34:15+00:00 from `lmarena-ai/arena-human-preference-55k`._

- Battles: 57,477 total, 49,938 single-turn.
- Responses classified: abstain 1,514, answer 95,347, clarify 817, empty 63, safety_refusal 2,135
- Tie rate across all single-turn battles: 31.3%

## When one model says "I don't know" and the other answers

| Subset | Battles | IDK wins | Answer wins | Ties | P(IDK wins), decisive | 95% CI | Tie rate |
|---|---|---|---|---|---|---|---|
| all IDK-vs-answer battles | 1,145 | 276 | 501 | 368 | 35.5% | 32.2% to 39.0% | 32.1% |
| factual-looking prompts | 541 | 135 | 229 | 177 | 37.1% | 32.3% to 42.2% | 32.7% |
| other prompts | 604 | 141 | 272 | 191 | 34.1% | 29.7% to 38.8% | 31.6% |
| bare IDK (<= 60 words) | 377 | 76 | 169 | 132 | 31.0% | 25.6% to 37.1% | 35.0% |
| IDK plus help (> 60 words) | 768 | 200 | 332 | 236 | 37.6% | 33.6% to 41.8% | 30.7% |

## Controlling for model strength, length and hedging (Bradley-Terry logit)

- **All prompts** (32,597 decisive battles, 64 models): abstaining multiplies the odds of winning by 0.52 (95% CI 0.44 to 0.61); each extra hedge per 100 words: OR 0.95 (0.93 to 0.98).
- **Factual-looking prompts** (7,692 decisive battles, 62 models): abstaining multiplies the odds of winning by 0.53 (95% CI 0.42 to 0.68); each extra hedge per 100 words: OR 0.92 (0.87 to 0.97).

## Caveats

- Labels come from a transparent phrase list (`src/idk/hedges.py`). Precision is checked by hand on `audit_sample.csv` and will be double-checked by an LLM classifier.
- Correctness is unknown here: an "answer" may be wrong and an "I don't know" may be appropriate. Part 2 (the controlled experiment) addresses this.
- Arena voters are self-selected and the prompts are whatever they chose to ask.
