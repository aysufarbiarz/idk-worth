# What is "I don't know" worth?

**How much do people, and the AI judges used to evaluate chatbots, value an assistant that admits it doesn't know?**

Evaluations that reward confident answers over honest abstentions push models toward guessing
([Kalai et al., 2025](https://arxiv.org/abs/2509.04664)). Earlier work shows that people often prefer a confident
falsehood to "I don't know" ([Bar-Or Nirman et al., 2024](https://arxiv.org/abs/2412.11625)) and that LLM judges
favour assertive answers ([Lee et al., 2025](https://aclanthology.org/2025.naacl-long.452);
[Datta & Baitalik, 2026](https://aclanthology.org/2026.acl-srw.33/)), while classic advice research finds people
don't always penalise uncertainty ([Gaertig & Simmons, 2018](https://doi.org/10.1177/0956797617739369)).
This project measures *where* the preference flips, and whether AI judges agree with people.

## Plan

| Part | Question | Data | Status |
|---|---|---|---|
| 1. Arena reanalysis | In real head-to-head votes, how often does the model that says "I don't know" win, holding model strength and length constant? Do LLM judges agree with the human votes? | [Chatbot Arena human preferences](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k) (Apache-2.0) | In progress |
| 2. Controlled experiment | At what stated confidence do people switch from preferring an answer to preferring "I don't know", how far is that from the rational threshold, and where do LLM judges switch? | New psychophysics-style study, preregistered on OSF | Design |

## Part 1 method

1. Keep single-turn Arena battles.
2. Label each response as **abstain** (leads with a lack of knowledge or access), **safety refusal** (a policy
   refusal, a different construct, excluded) or **answer**, using a transparent phrase list in
   [`src/idk/hedges.py`](src/idk/hedges.py). A random sample is hand-checked
   ([`results/audit_sample.csv`](results/audit_sample.csv)), then an LLM classifier double-checks the labels.
3. For battles where one side abstains and the other answers, report the abstainer's win rate with Wilson
   confidence intervals, split by prompt type and by bare vs. helpful abstentions.
4. Fit a Bradley-Terry logistic model on all decisive battles, so the abstention effect is estimated
   **holding model strength, response length and hedging constant** (weaker models may abstain more, and voters
   prefer longer answers).

Results: [`results/report.md`](results/report.md).

**Limits.** Arena data is observational. Correctness is unknown, so an "answer" may be wrong and an abstention may be
appropriate. Voters and prompts are self-selected. Part 2 is designed to address these.

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests
python scripts/01_arena_idk.py                    # downloads the data from Hugging Face
python scripts/make_synthetic.py syn.csv && \
python scripts/01_arena_idk.py --csv syn.csv --out /tmp/out   # offline check on synthetic data
```

The GitHub Action in `.github/workflows/arena.yml` runs the analysis and commits `results/`.

## Author

Aysu Farbiarz, Ph.D. (cognitive psychology).
