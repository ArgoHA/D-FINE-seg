# Lab notebook — D-FINE-seg autoresearch

This file is the **memory across agents/sessions**. A fresh agent reads the *Current state* block
first to know exactly where to resume, then scans entries to avoid repeating dead ends. The
structured numbers live in `ledger.csv`; this file is the reasoning.

## Current state  (keep this block updated every iteration)
- **Baseline established:** no — run the control once (see EXPERIMENT_GUIDE §4).
- **Current best (`main_exp`):** — (none yet)
- **Best metrics (test):** mAP_50_95 = — , f1 = — (from `baseline.json`)
- **In progress:** — (branch / idea being run right now, or "idle")
- **Next idea:** — (top of `ideas.md`)
- **Notes for the next agent:** —

---

Chronological log, newest first. One entry per candidate (promoted **or** rejected). Record the
*why*, not just the number — especially for failures.

Entry template:
```
## <date> — <name>   [PROMOTED | rejected | failed]
- Paper / source:
- Hypothesis:
- Change (files):
- Result (test, mean±std/seeds): mAP_50_95 <m>±<s> (best <b>), f1 <m>±<s>, lat ratio <r>, params <M>
- Read: why it worked / didn't. What it implies for the next idea.
```

---

<!-- entries below -->
