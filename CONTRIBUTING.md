# Contributing to CreditGuard

This is a reference implementation rather than an actively-developed library — but if you spot a bug, a stale doc, or want to extend a module, the following keeps changes consistent with the rest of the project.

## Setup

```bash
git clone https://github.com/josephhzy/creditguard.git
cd creditguard
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install   # ruff fix + ruff format + standard hygiene hooks
```

Requires Python 3.10 or 3.11 (CI runs against 3.11).

## Quality gates

Run before opening a pull request:

```bash
make lint     # ruff check . + ruff format --check .
make type     # mypy on the 7 source directories CI also checks
make test     # pytest tests/  (86 tests, ~20s)
make coverage # optional, gated at 55% on the same source set
```

Pre-commit will also fire `ruff --fix` and `ruff format` on every commit; the same checks run in `.github/workflows/ci.yml`.

## Conventions

- **Config-driven over hardcoded.** New thresholds, paths, and hyperparameters belong in `config/train.yaml`, not in module-level constants.
- **Two score scales.** The decisioning bands (`config/train.yaml::decisioning.bands`) live on the raw class-balanced model score; the API surfaces the Platt-calibrated PD as `risk_score`. See `decisioning/THRESHOLD_JUSTIFICATION.md` and the *Score scales* callout in the README. New code that references "the score" should be explicit about which scale.
- **mypy strict scope.** Full type strictness is enforced on `serving/*` (the web surface). The rest of the codebase is checked but not required to be fully annotated — see the per-package overrides in `pyproject.toml`.
- **Tests are evidence, not decoration.** Prefer assertions that would fail on a wrong answer over assertions that only check shape. The earlier audit removed several tautological assertions that passed regardless of correctness.
- **Honest commits.** If a change affects measured numbers (any value cited in README, model card, or governance docs), regenerate the relevant artefact (`make evaluate` chain) and update the docs in the same PR.

## Pull-request checklist

- [ ] `make lint`, `make type`, `make test` all pass locally.
- [ ] Any changed measured number is reflected in README + model card + governance docs.
- [ ] No new tracked files under `data/raw/`, `data/processed/`, `mlruns/`, or `*.pkl` — `.gitignore` and `.dockerignore` should already prevent this.
- [ ] If touching the serving stack: the real-bundle integration test in `tests/test_serving.py::TestRealBundleIntegration` still passes when the shipped LightGBM bundle is present.

## Reporting issues

Use GitHub Issues. For security-relevant findings, see `SECURITY.md`.
