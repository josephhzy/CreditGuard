"""CreditGuard data layer — download, validate, join, scan, and report.

Modules
-------
download          Download raw CSVs from Kaggle.
schema_validator  Validate table schemas, nulls, cardinality, duplicates.
join_tables       Join all 7 tables into one modelling-ready DataFrame.
leakage_scanner   Detect target leakage before training.
quality_report    Generate comprehensive data quality report (JSON artifact).
"""

from data.download import download_dataset, verify_download
from data.join_tables import build_joined_dataset
from data.leakage_scanner import LeakageReport, get_clean_columns, scan_for_leakage
from data.quality_report import QualityReport, build_quality_report
from data.schema_validator import SchemaReport, validate_all, validate_table

__all__ = [
    "download_dataset",
    "verify_download",
    "build_joined_dataset",
    "scan_for_leakage",
    "get_clean_columns",
    "LeakageReport",
    "build_quality_report",
    "QualityReport",
    "validate_all",
    "validate_table",
    "SchemaReport",
]
