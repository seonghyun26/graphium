<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-06 | Updated: 2026-04-06 -->

# expts/hydra-configs/experiment

## Purpose
Named experiment bundles — self-contained Hydra overrides that pin a specific combination of model/tasks/training for reproducibility. Rarely used in day-to-day work; most combinations are composed on-the-fly from the other groups.

## Key Files

| File | Description |
|------|-------------|
| `toymix_mpnn.yaml` | Pinned MPNN-on-ToyMix experiment bundle |

## For AI Agents

### Working In This Directory
- Prefer composing configs from the other groups (`model=`, `tasks=`, `training=`) over creating new experiment bundles. Bundle files are most useful when you want a shareable one-liner (`+experiment=<name>`).
- The Hydra override is `+experiment=<stem>` (leading `+`).

## Dependencies

### Internal
- Composes the other groups. No new consumers.

<!-- MANUAL: -->
