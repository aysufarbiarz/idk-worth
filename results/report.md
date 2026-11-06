# Part 1 results: Chatbot Arena reanalysis (rule-based first pass)

_Generated 2026-11-06T23:32:50+00:00 from `lmarena-ai/arena-human-preference-55k`._

- Battles: 57,477 total, 49,938 single-turn.
- Responses classified: abstain 1,813, answer 95,865, empty 63, safety_refusal 2,135
- Tie rate across all single-turn battles: 31.3%

## When one model says "I don't know" and the other answers

| Subset | Battles | IDK wins | Answer wins | Ties | P(IDK wins), decisive | 95% CI | Tie rate |
|---|---|---|---|---|---|---|---|
| all IDK-vs-answer battles | 1,402 | 327 | 618 | 457 | 34.6% | 31.6% to 37.7% | 32.6% |
| factual-looking prompts | 645 | 163 | 265 | 217 | 38.1% | 33.6% to 42.8% | 33.6% |
| other prompts | 757 | 164 | 353 | 240 | 31.7% | 27.9% to 35.9% | 31.7% |
| bare IDK (<= 60 words) | 582 | 108 | 270 | 204 | 28.6% | 24.3% to 33.3% | 35.1% |
| IDK plus help (> 60 words) | 820 | 219 | 348 | 253 | 38.6% | 34.7% to 42.7% | 30.9% |

## Controlling for model strength, length and hedging (Bradley-Terry logit)

- **All prompts** (33,073 decisive battles, 64 models): abstaining multiplies the odds of winning by 0.54 (95% CI 0.46 to 0.63); each extra hedge per 100 words: OR 0.95 (0.93 to 0.98).
- **Factual-looking prompts** (7,844 decisive battles, 62 models): abstaining multiplies the odds of winning by 0.61 (95% CI 0.49 to 0.77); each extra hedge per 100 words: OR 0.93 (0.88 to 0.98).

## Caveats

- Labels come from a transparent phrase list (`src/idk/hedges.py`). Precision is checked by hand on `audit_sample.csv` and will be double-checked by an LLM classifier.
- Correctness is unknown here: an "answer" may be wrong and an "I don't know" may be appropriate. Part 2 (the controlled experiment) addresses this.
- Arena voters are self-selected and the prompts are whatever they chose to ask.
