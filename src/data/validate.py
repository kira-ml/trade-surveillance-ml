"""Raw data validation module for synthetic limit order book events.

Performs essential data quality checks on the raw dataset before downstream processing.
Validates schema compliance, value constraints, and data completeness without
modifying the data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional
from pathlib import Path

import pandas as pd
import numpy as np


@dataclass
class ValidationReport:
    """Immutable validation results container."""
    passed: bool
    checks: Dict[str, bool]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class DataValidator:
    """Validates raw LOB event data against schema and quality requirements.

    Performs configurable validation checks on the input DataFrame, ensuring data
    meets minimum quality standards before entering the preprocessing pipeline.
    """

    # Expected schema definition
    REQUIRED_COLUMNS: Set[str] = {
        "trader_id", "event_type", "order_size", "timestamp",
        "price", "mid_price", "label"
    }

    ALLOWED_EVENT_TYPES: Set[str] = {"placement", "cancellation", "execution"}
    ALLOWED_LABELS: Set[int] = {0, 1}

    def __init__(self, config: Dict[str, Any]):
        """Initialize validator with configuration.

        Args:
            config: Validated configuration dictionary containing validation
                parameters including null_threshold.

        Raises:
            ValueError: If required config keys are missing or invalid.
        """
        self._validate_config(config)
        self.null_threshold = config["validation"]["null_threshold"]
        self.checks: Dict[str, bool] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Ensure configuration contains required validation parameters."""
        if "validation" not in config:
            raise ValueError("Missing 'validation' section in config")

        validation_config = config["validation"]
        if "null_threshold" not in validation_config:
            raise ValueError("Missing 'null_threshold' in validation config")

        threshold = validation_config["null_threshold"]
        if not 0 <= threshold <= 1:
            raise ValueError(f"null_threshold must be between 0 and 1, got {threshold}")

    def _check_required_columns(self, df: pd.DataFrame) -> bool:
        """Verify all required columns exist in DataFrame."""
        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)

        if missing_cols:
            self.errors.append(f"Missing required columns: {sorted(missing_cols)}")
            return False

        return True

    def _check_null_values(self, df: pd.DataFrame) -> bool:
        """Verify null ratio doesn't exceed threshold for any column."""
        null_ratios = df.isnull().mean()

        for col, ratio in null_ratios.items():
            if ratio > self.null_threshold:
                self.errors.append(
                    f"Column '{col}' has {ratio:.1%} nulls, "
                    f"exceeds threshold of {self.null_threshold:.1%}"
                )
                return False
            elif ratio > 0:
                self.warnings.append(
                    f"Column '{col}' has {ratio:.1%} nulls "
                    f"(within threshold {self.null_threshold:.1%})"
                )

        return True

    def _check_event_types(self, df: pd.DataFrame) -> bool:
        """Validate event_type contains only allowed values."""
        if "event_type" not in df.columns:
            return False

        invalid_types = set(df["event_type"].dropna().unique()) - self.ALLOWED_EVENT_TYPES

        if invalid_types:
            self.errors.append(
                f"Invalid event_type values found: {sorted(invalid_types)}. "
                f"Allowed: {sorted(self.ALLOWED_EVENT_TYPES)}"
            )
            return False

        return True

    def _check_labels(self, df: pd.DataFrame) -> bool:
        """Validate label contains only binary values 0 or 1."""
        if "label" not in df.columns:
            return False

        unique_labels = set(df["label"].dropna().unique())
        invalid_labels = unique_labels - self.ALLOWED_LABELS

        if invalid_labels:
            self.errors.append(
                f"Invalid label values found: {sorted(invalid_labels)}. "
                f"Labels must be binary (0 or 1)"
            )
            return False

        return True

    def _check_positive_numeric(self, df: pd.DataFrame, col: str) -> bool:
        """Validate column contains only positive numeric values."""
        if col not in df.columns:
            return False

        # Drop nulls for this check - nulls handled separately
        non_null = df[col].dropna()

        if not pd.api.types.is_numeric_dtype(non_null):
            self.errors.append(f"Column '{col}' must be numeric")
            return False

        if (non_null <= 0).any():
            self.errors.append(f"Column '{col}' contains non-positive values")
            return False

        return True

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Execute all configured validation checks.

        Args:
            df: Raw DataFrame to validate.

        Returns:
            ValidationReport containing pass/fail status, detailed check results,
            and any errors or warnings encountered.

        Raises:
            ValueError: If critical validation failures occur.
        """
        # Reset state for this validation run
        self.checks = {}
        self.errors = []
        self.warnings = []

        # Execute validation checks in order
        checks = [
            ("required_columns", self._check_required_columns(df)),
            ("null_values", self._check_null_values(df)),
            ("event_types", self._check_event_types(df)),
            ("labels", self._check_labels(df)),
            ("order_size_positive", self._check_positive_numeric(df, "order_size")),
            ("price_positive", self._check_positive_numeric(df, "price")),
            ("mid_price_positive", self._check_positive_numeric(df, "mid_price")),
        ]

        # Record check results
        for check_name, result in checks:
            self.checks[check_name] = result

        # Determine overall status
        all_passed = all(self.checks.values())
        has_critical_errors = len(self.errors) > 0

        # Log warnings for informational purposes
        for warning in self.warnings:
            print(f"Validation warning: {warning}")

        # Raise on critical failures
        if has_critical_errors:
            error_msg = "Validation failed with critical errors:\n" + "\n".join(
                f"  - {error}" for error in self.errors
            )
            raise ValueError(error_msg)

        return ValidationReport(
            passed=all_passed,
            checks=self.checks.copy(),
            errors=self.errors.copy(),
            warnings=self.warnings.copy()
        )


def validate_raw_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load raw data from configured path and validate it.

    Args:
        config: Validated configuration dictionary containing:
            - paths.raw_data: Path to raw CSV file
            - validation.null_threshold: Maximum allowed null ratio

    Returns:
        Dictionary containing validation results with keys:
            - passed: bool indicating if validation succeeded
            - checks: dict mapping check names to pass/fail
            - errors: list of error messages
            - warnings: list of warning messages
            - num_rows: number of rows in validated dataset
            - num_traders: number of unique traders

    Raises:
        FileNotFoundError: If raw data file doesn't exist
        pd.errors.EmptyDataError: If data file is empty
        ValueError: If validation fails critically
    """
    raw_data_path = Path(config["paths"]["raw_data"])

    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data not found at: {raw_data_path}")

    # Load raw data
    df = pd.read_csv(raw_data_path)

    if df.empty:
        raise pd.errors.EmptyDataError("Raw data file is empty")

    # Validate data
    validator = DataValidator(config)
    report = validator.validate(df)

    # Build return dictionary
    result = {
        "passed": report.passed,
        "checks": report.checks,
        "errors": report.errors,
        "warnings": report.warnings,
        "num_rows": len(df),
        "num_traders": df["trader_id"].nunique() if "trader_id" in df.columns else 0,
    }

    # Log summary
    print(f"Validation {'PASSED' if report.passed else 'FAILED'}")
    print(f"Checks passed: {sum(report.checks.values())}/{len(report.checks)}")
    print(f"Total rows: {result['num_rows']:,}")
    print(f"Unique traders: {result['num_traders']:,}")

    return result