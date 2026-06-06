"""Download the Home Credit Default Risk dataset from Kaggle.

This module handles dataset acquisition from the Kaggle competition
``home-credit-default-risk``. It supports both API-based download
(requires ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` in the environment or
``~/.kaggle/kaggle.json``) and provides clear instructions for manual
fallback when API access is unavailable.

Typical usage::

    python -m data.download              # download all tables
    python -m data.download --check      # verify files exist with expected sizes
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Final

import structlog
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMPETITION: Final[str] = "home-credit-default-risk"

EXPECTED_FILES: Final[dict[str, int]] = {
    # filename -> approximate minimum size in bytes (sanity check)
    "application_train.csv": 150_000_000,
    "application_test.csv": 40_000_000,
    "bureau.csv": 90_000_000,
    "bureau_balance.csv": 250_000_000,
    "previous_application.csv": 170_000_000,
    "POS_CASH_balance.csv": 130_000_000,
    "installments_payments.csv": 230_000_000,
    "credit_card_balance.csv": 100_000_000,
}

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def _load_config(config_path: Path | None = None) -> dict:
    """Load project configuration from ``config/train.yaml``."""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "train.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _raw_dir(cfg: dict) -> Path:
    """Resolve the raw data directory from the config."""
    project_root = Path(__file__).resolve().parents[1]
    return project_root / cfg["data"]["raw_dir"]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _kaggle_api_available() -> bool:
    """Check whether the ``kaggle`` CLI is installed and credentials exist."""
    if shutil.which("kaggle") is None:
        return False
    # Check env vars first, then JSON credential file
    has_env = bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY"))
    has_json = (Path.home() / ".kaggle" / "kaggle.json").exists()
    return has_env or has_json


def download_via_api(raw_dir: Path) -> None:
    """Download all competition files using the Kaggle CLI.

    Parameters
    ----------
    raw_dir:
        Directory to store the downloaded CSV files.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info("kaggle_download_start", competition=COMPETITION, dest=str(raw_dir))

    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-p",
        str(raw_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("kaggle_download_failed", stderr=result.stderr.strip())
        raise RuntimeError(
            f"Kaggle download failed (exit {result.returncode}): {result.stderr.strip()}"
        )

    logger.info("kaggle_download_complete", dest=str(raw_dir))

    # The competition download comes as a single ZIP — extract it
    zip_path = raw_dir / f"{COMPETITION}.zip"
    if zip_path.exists():
        _extract_zip(zip_path, raw_dir)
    else:
        # Sometimes individual zips are downloaded per file
        for zf in raw_dir.glob("*.zip"):
            _extract_zip(zf, raw_dir)


def _extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract a ZIP archive and remove the archive file."""
    logger.info("extracting_zip", path=str(zip_path))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    zip_path.unlink()
    logger.info("zip_extracted_and_removed", path=str(zip_path))


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_download(raw_dir: Path) -> bool:
    """Verify that all expected CSV files exist and are above minimum size.

    Parameters
    ----------
    raw_dir:
        Directory where CSV files should be located.

    Returns
    -------
    bool
        ``True`` if all files pass checks; ``False`` otherwise.
    """
    all_ok = True
    for filename, min_size in EXPECTED_FILES.items():
        fpath = raw_dir / filename
        if not fpath.exists():
            logger.error("missing_file", file=filename)
            all_ok = False
            continue
        actual_size = fpath.stat().st_size
        if actual_size < min_size:
            logger.warning(
                "small_file",
                file=filename,
                actual=actual_size,
                expected_min=min_size,
            )
            all_ok = False
        else:
            logger.info(
                "file_ok",
                file=filename,
                size_mb=round(actual_size / 1_048_576, 1),
            )
    return all_ok


# ---------------------------------------------------------------------------
# Manual download instructions
# ---------------------------------------------------------------------------

_MANUAL_INSTRUCTIONS: Final[str] = """
╔══════════════════════════════════════════════════════════════════╗
║  MANUAL DOWNLOAD INSTRUCTIONS                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Go to https://www.kaggle.com/c/home-credit-default-risk/data ║
║  2. Log in to your Kaggle account                                ║
║  3. Click "Download All" to get the ZIP archive                  ║
║  4. Extract the ZIP into:                                        ║
║         {raw_dir}
║                                                                  ║
║  Required files:                                                 ║
║    - application_train.csv                                       ║
║    - application_test.csv                                        ║
║    - bureau.csv                                                  ║
║    - bureau_balance.csv                                          ║
║    - previous_application.csv                                    ║
║    - POS_CASH_balance.csv                                        ║
║    - installments_payments.csv                                   ║
║    - credit_card_balance.csv                                     ║
║                                                                  ║
║  Alternatively, install the Kaggle CLI:                          ║
║    pip install kaggle                                             ║
║  Set credentials via env vars or ~/.kaggle/kaggle.json           ║
║  Then rerun: python -m data.download                             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


def print_manual_instructions(raw_dir: Path) -> None:
    """Print human-readable download instructions for manual fallback."""
    print(_MANUAL_INSTRUCTIONS.format(raw_dir=raw_dir))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def download_dataset(config_path: Path | None = None, *, check_only: bool = False) -> bool:
    """Download (or verify) the Home Credit dataset.

    Parameters
    ----------
    config_path:
        Optional path to ``config/train.yaml``. Uses the default location
        when ``None``.
    check_only:
        When ``True``, skip the download and only run verification.

    Returns
    -------
    bool
        ``True`` when all files are present and valid.
    """
    cfg = _load_config(config_path)
    raw_dir = _raw_dir(cfg)

    if check_only:
        logger.info("verification_only_mode", raw_dir=str(raw_dir))
        return verify_download(raw_dir)

    # Attempt API download first; fall back to manual instructions
    if _kaggle_api_available():
        try:
            download_via_api(raw_dir)
        except RuntimeError:
            logger.warning("api_download_failed_falling_back_to_manual")
            print_manual_instructions(raw_dir)
            return False
    else:
        logger.warning("kaggle_cli_not_available")
        print_manual_instructions(raw_dir)
        return False

    return verify_download(raw_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Home Credit dataset from Kaggle")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify that files exist (skip download)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (default: config/train.yaml)",
    )
    args = parser.parse_args()

    ok = download_dataset(config_path=args.config, check_only=args.check)
    sys.exit(0 if ok else 1)
