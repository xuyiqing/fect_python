# fect (Python)

Fixed Effects Counterfactuals (FEct) for Python.

This is a Python port of the R package `fect`, providing counterfactual estimation in panel data using fixed effects, along with event-study summaries and plots.

## Install

```bash
pip install .
```

or in editable mode during development:

```bash
pip install -e .
```

## Quick start

```python
import pandas as pd
from fect import fect, effect, esplot

# df must include columns: unit id, time, outcome Y, treatment D (0/1), optional covariates X...
res = fect(
    data=df,
    Y="Y",
    D="D",
    X=["x1", "x2"],
    index=("unit", "time"),
    method="fe"  # two-way fixed effects (r=0)
)

# Event-study style effect summary
res2 = effect(res, cumu=False, plot=True)
```

## Status

- Implements a two-way fixed-effects counterfactual estimator (equivalent to `method="fe"` in R, internally treated as `ife` with `r=0`).
- Includes `effect` and `esplot` utilities to summarize and visualize ATT.
- Interactive fixed effects with r>0 and binary outcome models will be added in future iterations.
