# Acquirer Engine

Finds the 10 most likely acquirers for a target company from a dataset of past
M&A transactions, and writes a one-page rationale for each.

## Quick start

```
python -m venv .venv
.venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]"
acquirer-engine run --mode replay
```
