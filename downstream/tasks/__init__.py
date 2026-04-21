"""Per-task downstream eval drivers.

Each task subpackage owns:
    config.py   constants (subsets, methods, CV folds, feature dims)
    data.py     loaders for prepped parquets + split index files
    eval.py     main CLI driver (``python -m downstream.tasks.<task>.eval``)

Results land in ``results/downstream/<task>.csv`` (single CSV per task so the
dashboard notebook doesn't have to union per-encoder files).
"""
