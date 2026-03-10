"""Unit tests for the data validation module.

Tests focus on core validation logic with minimal mocking and realistic data scenarios.
Avoids over-engineering by testing behavior, not implementation details.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.data.validate import validate_raw_data, DataValidator, ValidationReport


@pytest.fixture
def valid_config():
    """Return a minimal valid configuration."""
    return {
        "paths": {
            "raw_data": "dummy/path.csv"
        },
        "validation": {
            "null_threshold": 0.05  # 5% max nulls
        }
    }


@pytest.fixture
def valid_dataframe():
    """Create a DataFrame with valid data matching expected schema."""
    np.random.seed(42)
    n_rows = 100

    return pd.DataFrame({
        "trader_id": np.repeat(range(10), 10),
        "event_type": np.random.choice(
            ["placement", "cancellation", "execution"], 
            size=n_rows,
            p=[0.5, 0.25, 0.25]
        ),
        "order_size": np.random.lognormal(4, 0.8, n_rows).astype(int),
        "timestamp": np.cumsum(np.random.exponential(0.2, n_rows)),
        "price": np.random.normal(100, 2, n_rows),
        "mid_price": 100.0,
        "label": np.random.choice([0, 1], size=n_rows, p=[0.9, 0.1])
    })


@pytest.fixture
def temp_csv_file(valid_dataframe):
    """Create a temporary CSV file with valid data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        valid_dataframe.to_csv(f.name, index=False)
        yield Path(f.name)
    # Cleanup after test
    Path(f.name).unlink(missing_ok=True)


class TestDataValidator:
    """Test suite for DataValidator class."""

    def test_initialization_with_valid_config(self, valid_config):
        """Validator should initialize correctly with valid config."""
        validator = DataValidator(valid_config)
        assert validator.null_threshold == 0.05
        assert validator.checks == {}
        assert validator.errors == []
        assert validator.warnings == []

    def test_initialization_with_invalid_threshold(self, valid_config):
        """Should reject null_threshold outside [0,1] range."""
        for invalid_threshold in [-0.1, 1.5]:
            valid_config["validation"]["null_threshold"] = invalid_threshold
            with pytest.raises(ValueError, match="null_threshold must be between 0 and 1"):
                DataValidator(valid_config)

    def test_initialization_missing_validation_section(self, valid_config):
        """Should reject config missing validation section."""
        del valid_config["validation"]
        with pytest.raises(ValueError, match="Missing 'validation' section"):
            DataValidator(valid_config)

    def test_check_required_columns_all_present(self, valid_config, valid_dataframe):
        """Should pass when all required columns exist."""
        validator = DataValidator(valid_config)
        assert validator._check_required_columns(valid_dataframe) is True

    def test_check_required_columns_missing(self, valid_config, valid_dataframe):
        """Should fail when required columns are missing."""
        validator = DataValidator(valid_config)
        df_missing = valid_dataframe.drop(columns=["price"])
        assert validator._check_required_columns(df_missing) is False
        assert any("Missing required columns" in e for e in validator.errors)

    def test_check_null_values_within_threshold(self, valid_config, valid_dataframe):
        """Should pass when null ratios are below threshold."""
        validator = DataValidator(valid_config)
        # Introduce some nulls within threshold (2% < 5%)
        valid_dataframe.loc[0:1, "price"] = np.nan
        assert validator._check_null_values(valid_dataframe) is True
        assert len(validator.warnings) > 0  # Should warn about nulls

    def test_check_null_values_exceeds_threshold(self, valid_config, valid_dataframe):
        """Should fail when null ratios exceed threshold."""
        validator = DataValidator(valid_config)
        # Introduce 10% nulls (> 5% threshold)
        valid_dataframe.loc[0:9, "price"] = np.nan
        assert validator._check_null_values(valid_dataframe) is False
        assert any("exceeds threshold" in e for e in validator.errors)

    def test_check_event_types_all_valid(self, valid_config, valid_dataframe):
        """Should pass when all event types are allowed."""
        validator = DataValidator(valid_config)
        assert validator._check_event_types(valid_dataframe) is True

    def test_check_event_types_invalid(self, valid_config, valid_dataframe):
        """Should fail when invalid event types exist."""
        validator = DataValidator(valid_config)
        valid_dataframe.loc[0, "event_type"] = "invalid_type"
        assert validator._check_event_types(valid_dataframe) is False
        assert any("Invalid event_type" in e for e in validator.errors)

    def test_check_labels_all_valid(self, valid_config, valid_dataframe):
        """Should pass when labels are only 0 or 1."""
        validator = DataValidator(valid_config)
        assert validator._check_labels(valid_dataframe) is True

    def test_check_labels_invalid(self, valid_config, valid_dataframe):
        """Should fail when labels outside {0,1} exist."""
        validator = DataValidator(valid_config)
        valid_dataframe.loc[0, "label"] = 2
        assert validator._check_labels(valid_dataframe) is False
        assert any("Invalid label values" in e for e in validator.errors)

    def test_check_positive_numeric_valid(self, valid_config, valid_dataframe):
        """Should pass for columns with all positive values."""
        validator = DataValidator(valid_config)
        assert validator._check_positive_numeric(valid_dataframe, "order_size") is True
        assert validator._check_positive_numeric(valid_dataframe, "price") is True

    def test_check_positive_numeric_non_positive(self, valid_config, valid_dataframe):
        """Should fail when column contains non-positive values."""
        validator = DataValidator(valid_config)
        valid_dataframe.loc[0, "order_size"] = 0
        assert validator._check_positive_numeric(valid_dataframe, "order_size") is False
        assert any("non-positive values" in e for e in validator.errors)

    def test_check_positive_numeric_non_numeric(self, valid_config, valid_dataframe):
        """Should fail when column is not numeric."""
        validator = DataValidator(valid_config)
        valid_dataframe["order_size"] = valid_dataframe["order_size"].astype(str)
        assert validator._check_positive_numeric(valid_dataframe, "order_size") is False
        assert any("must be numeric" in e for e in validator.errors)

    def test_validate_all_checks_pass(self, valid_config, valid_dataframe):
        """Complete validation should pass with valid data."""
        validator = DataValidator(valid_config)
        report = validator.validate(valid_dataframe)

        assert report.passed is True
        assert all(report.checks.values())
        assert len(report.errors) == 0
        assert isinstance(report, ValidationReport)

    def test_validate_with_warnings(self, valid_config, valid_dataframe):
        """Validation should pass but include warnings with minor issues."""
        validator = DataValidator(valid_config)
        # Add nulls within threshold
        valid_dataframe.loc[0:2, "price"] = np.nan

        report = validator.validate(valid_dataframe)

        assert report.passed is True
        assert len(report.warnings) > 0
        assert any("nulls" in w for w in report.warnings)


class TestValidateRawData:
    """Integration tests for the main validate_raw_data function."""

    def test_successful_validation(self, valid_config, temp_csv_file):
        """Should successfully validate a valid CSV file."""
        valid_config["paths"]["raw_data"] = str(temp_csv_file)
        
        result = validate_raw_data(valid_config)
        
        assert result["passed"] is True
        assert all(result["checks"].values())
        assert len(result["errors"]) == 0
        assert result["num_rows"] > 0
        assert result["num_traders"] > 0

    def test_file_not_found(self, valid_config):
        """Should raise FileNotFoundError when CSV doesn't exist."""
        valid_config["paths"]["raw_data"] = "nonexistent.csv"
        
        with pytest.raises(FileNotFoundError, match="Raw data not found"):
            validate_raw_data(valid_config)

    def test_empty_file(self, valid_config):
        """Should raise EmptyDataError for empty CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("")
            f.flush()
            valid_config["paths"]["raw_data"] = f.name
        
        with pytest.raises(pd.errors.EmptyDataError, match="No columns to parse"):
            validate_raw_data(valid_config)
        
        Path(f.name).unlink()

    def test_validation_failure_critical(self, valid_config, valid_dataframe):
        """Should raise ValueError on critical validation failures."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Add invalid data
            valid_dataframe.loc[0, "label"] = 999
            valid_dataframe.to_csv(f.name, index=False)
            valid_config["paths"]["raw_data"] = f.name

        with pytest.raises(ValueError, match="Validation failed"):
            validate_raw_data(valid_config)

        Path(f.name).unlink()

    def test_validation_report_structure(self, valid_config, temp_csv_file):
        """Validation report should contain all expected keys."""
        valid_config["paths"]["raw_data"] = str(temp_csv_file)
        
        result = validate_raw_data(valid_config)
        
        expected_keys = {"passed", "checks", "errors", "warnings", "num_rows", "num_traders"}
        assert set(result.keys()) == expected_keys
        assert isinstance(result["checks"], dict)
        assert isinstance(result["errors"], list)
        assert isinstance(result["warnings"], list)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_null_threshold_boundary(self, valid_config, valid_dataframe):
        """Test behavior at null threshold boundaries."""
        validator = DataValidator(valid_config)
        
        # Exactly at threshold (5% nulls in 100 rows = 5 rows)
        valid_dataframe.loc[0:4, "price"] = np.nan
        assert validator._check_null_values(valid_dataframe) is True
        
        # Just above threshold (6% nulls)
        valid_dataframe.loc[5, "price"] = np.nan
        assert validator._check_null_values(valid_dataframe) is False

    def test_dataframe_with_all_nulls_in_column(self, valid_config, valid_dataframe):
        """Should handle columns that are completely null."""
        validator = DataValidator(valid_config)
        valid_dataframe["price"] = np.nan
        
        assert validator._check_null_values(valid_dataframe) is False
        assert any("100.0% nulls" in e for e in validator.errors)

    def test_mixed_valid_invalid_event_types(self, valid_config, valid_dataframe):
        """Should detect any invalid event type even with valid ones present."""
        validator = DataValidator(valid_config)
        valid_dataframe.loc[[0, 1, 2], "event_type"] = ["invalid", "also_invalid", "placement"]
        
        assert validator._check_event_types(valid_dataframe) is False
        # Should report all unique invalid types
        assert "invalid" in str(validator.errors)
        assert "also_invalid" in str(validator.errors)

    def test_large_dataframe_performance(self, valid_config):
        """Ensure validator handles larger DataFrames efficiently."""
        # Create larger dataset (10k rows)
        n_rows = 10000
        df = pd.DataFrame({
            "trader_id": np.repeat(range(100), 100),
            "event_type": np.random.choice(["placement", "cancellation", "execution"], n_rows),
            "order_size": np.random.randint(1, 1000, n_rows),
            "timestamp": np.cumsum(np.random.exponential(0.2, n_rows)),
            "price": np.random.normal(100, 2, n_rows),
            "mid_price": 100.0,
            "label": np.random.choice([0, 1], n_rows)
        })
        
        validator = DataValidator(valid_config)
        report = validator.validate(df)
        
        assert report.passed is True
        assert len(df) == n_rows


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])