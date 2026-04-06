<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-04-05 | Updated: 2026-04-05 -->

# docs

## Purpose
MkDocs-based documentation site. Content is authored in Markdown here and rendered via `mkdocs serve` using the config at `../mkdocs.yml`. The site documents the library, the CLI, the Hydra config system, and a handful of tutorials.

## Key Files

| File | Description |
|------|-------------|
| `index.md` | Landing page for the docs site |
| (other top-level `.md` files) | Guides for install, datasets, design, and features |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `api/` | Auto-referenced API docs; `api/graphium.nn/` groups the neural-network module pages |
| `cli/` | CLI reference pages for `graphium-train` and its subcommands |
| `tutorials/` | Walkthrough notebooks exported as Markdown |
| `tutorials/feature_processing/` | SMILES featurization walkthroughs |
| `tutorials/gnn/` | GNN architecture tutorials |
| `tutorials/model_training/` | Training + finetuning walkthroughs |
| `images/` | Logo, architecture diagrams, plots |
| `_assets/` | CSS and JS overrides (`_assets/css/`, `_assets/js/`) |

## For AI Agents

### Working In This Directory
- Build locally with `mkdocs serve` (from the project root). Config: `../mkdocs.yml`.
- API docs are generated from docstrings via `mkdocstrings` — if you add a new public class, adding a one-line entry under `docs/api/` is enough.
- Tutorials are plain Markdown; if you need to embed code outputs, keep them small and static — we do not execute notebooks during the build.
- Keep image assets small — the repo already has enough binary weight.

### Testing Requirements
- `mkdocs build --strict` should succeed — this is the closest thing to a doc lint.

### Common Patterns
- Cross-links use MkDocs path syntax (`[text](../api/graphium.nn/base_layers.md)`).
- The CLI subcommand docs live under `cli/` and mirror the Typer app structure in `graphium/cli/`.

## Dependencies

### Internal
- Docstrings in `../graphium/` (via `mkdocstrings`)

### External
- `mkdocs`, `mkdocs-material`, `mkdocstrings[python]`

<!-- MANUAL: -->
