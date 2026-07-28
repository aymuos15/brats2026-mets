# Our own scored submissions

This directory intentionally contains **no full-field leaderboard dumps**.

The snapshot files we work from (`lb_snapshot.csv`, `lb.json`, `lb_1400.csv`,
`brats2026final.html`) hold the scores of every submission by every team. Under the challenge
rules, participants may publish their own method's results and their own team ranking, but not
overall challenge results or any analysis of them, until the organisers release the overview
paper. Those files are therefore gitignored and kept locally only.

To regenerate them for private use, with a Synapse token:

```bash
python ../../src/eval/lbquery2.py   # -> lb_snapshot.csv
python ../../src/eval/rank12.py     # 12-metric standing
```
