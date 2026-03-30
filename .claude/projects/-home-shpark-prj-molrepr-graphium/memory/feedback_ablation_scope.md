---
name: feedback_ablation_scope
description: User prefers small representative ablation runs (5 tasks, one per ADMET category) over full 22-task sweeps for initial validation
type: feedback
---

When writing ablation/pilot finetuning scripts, default to 5 representative ADMET tasks (one per A/D/M/E/T category) rather than all 22.

**Why:** Full 22-task sweeps are expensive and slow; the user wants quick signal before committing to a full run.

**How to apply:** Use representative tasks like caco2_wang (A), bbb_martins (D), cyp3a4_veith (M), half_life_obach (E), herg (T) for initial ablation scripts. Only expand to all 22 when explicitly asked.
