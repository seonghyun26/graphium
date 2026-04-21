"""Downstream-task evaluation suite.

Packages:
    model/   Molecule-encoder adapters (pairmixer/minimol/mole) with a uniform
             ``extract(smiles) -> (features, mask)`` interface. Checkpoint files
             themselves stay under ``models_checkpoints/`` — this package only
             wraps loading + feature extraction.
    tasks/   Per-task eval drivers. Each task owns its data loader + a single
             per-task CSV output under ``results/downstream/<task>.csv``. The
             graphium-train / Hydra pipeline is reserved for ADMET only; every
             other downstream task runs through modules in ``tasks/``.
"""
