.PHONY: install dev lint type test coverage train evaluate serve monitor clean docker docker-up verify-artifacts join

# ── Setup ──────────────────────────────────────────────
install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install || echo "pre-commit hook install skipped (not in a git repo or pre-commit unavailable)"

# ── Quality ────────────────────────────────────────────
lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

type:
	mypy data/ features/ training/ evaluation/ decisioning/ monitoring/ serving/

test:
	pytest tests/ -v --tb=short

# Coverage floor reflects what tests actually exercise today (~56% across
# the full code surface). Streamlit, training/experiments, retrain orchestration,
# and one-off scripts are excluded in pyproject.toml because they are not unit
# tested. Raising this floor requires adding tests, not relaxing exclusions.
coverage:
	pytest tests/ --cov=. --cov-report=html --cov-fail-under=55
	@echo "Coverage report: htmlcov/index.html"

# ── Pipeline ───────────────────────────────────────────
download:
	python -m data.download

validate:
	python -m data.schema_validator
	python -m data.leakage_scanner

# Joins the 7 Home Credit tables into a single one-row-per-applicant parquet.
# Run after `make validate`, before `make features`.
join:
	python -m data.join_tables

features:
	python -m features.build_all

train:
	python -m training.train --config config/train.yaml

# Re-runs OOF predictions, recomputes evaluation metrics (champion + challenger),
# refits the Platt calibrator on OOF, and reconciles fairness slices into the
# governance JSON. This is the canonical sequence after a fresh train.
evaluate:
	python -m scripts.evaluate_and_compare
	python -m scripts.fit_calibrator
	python -m scripts.reconcile_fairness

# Re-runs the five experiment tracks (baseline, imbalance treatment,
# feature selection, validation strategy, calibration). Each track logs
# its runs to MLflow under the configured tracking_uri (default
# `mlruns/`); no artefact files are written outside the tracking store.
# Wall time: ~30-45 min total on a laptop CPU.
# Note: `experiments/` is a CLI driver package, distinct from the
# `training/experiments.py` library module — same word, different roles.
experiments:
	python -m experiments.track_1_baseline
	python -m experiments.track_2_imbalance
	python -m experiments.track_3_feature_selection
	python -m experiments.track_4_validation
	python -m experiments.track_5_calibration

# ── Serving ────────────────────────────────────────────
# `make serve` and `make docker` rely on the trained bundle being present;
# fail loudly with a one-liner pointing at `make train` rather than letting
# the API start in degraded mode.
verify-artifacts:
	@test -f artifacts/lightgbm_model.pkl || (echo "ERROR: artifacts/lightgbm_model.pkl is missing. Run 'make train' first (see Reproducibility note in README.md)." && exit 1)
	@echo "OK: artifacts/lightgbm_model.pkl present."

serve: verify-artifacts
	uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload

# ── Monitoring ─────────────────────────────────────────
# Full-feature mode (15 features) when data/processed/features.parquet exists;
# falls back to score-only demo (1 feature, demo:true) otherwise. Run `make features`
# first for the full report. The API's /drift/report endpoint reads the same file.
monitor:
	python -m monitoring.run_checks

# ── Docker ─────────────────────────────────────────────
docker: verify-artifacts
	docker build -t creditguard:latest -f infra/Dockerfile .

docker-up: verify-artifacts
	docker compose -f infra/docker-compose.yml up --build

docker-down:
	docker compose -f infra/docker-compose.yml down

# ── Cleanup ────────────────────────────────────────────
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
