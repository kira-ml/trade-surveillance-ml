"""
tests/test_simulate_integration.py
────────────────────────────────────────────────────────────────────────────────
Integration tests for src/data/simulate.py.

Unlike the unit tests in test_simulate.py, these tests load the *real*
configs/config.yaml and write to the project's actual data/raw/ directory.
They validate that the config schema consumed by simulate() stays in sync with
the values declared in config.yaml.

Run selectively:
    pytest -m "not integration"   # skip during fast unit-test runs
    pytest -m integration          # run only integration tests
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data.simulate import simulate

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"


@pytest.fixture(scope="module")
def real_cfg() -> dict:
    """Load the actual project configs/config.yaml."""
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


@pytest.mark.integration
class TestConfigYamlSmoke:
    """Smoke tests using the real config.yaml → real data/raw/lob_events.csv."""

    def test_config_file_exists(self):
        """configs/config.yaml must be present for the pipeline to run."""
        assert CONFIG_PATH.exists(), f"Config not found: {CONFIG_PATH}"

    def test_config_simulation_keys_present(self, real_cfg):
        """All keys consumed by simulate() must exist in config.yaml."""
        sim = real_cfg["data"]["simulation"]
        required = {
            "num_traders",
            "num_events_per_trader",
            "manipulation_fraction",
            "poisson_lambda",
            "price_mean",
            "price_std",
            "seed",
        }
        missing = required - sim.keys()
        assert not missing, f"config.yaml missing simulation keys: {missing}"

    def test_config_paths_keys_present(self, real_cfg):
        """The 'raw' output path key must exist under data.paths."""
        paths = real_cfg["data"]["paths"]
        assert "raw" in paths, "config.yaml missing data.paths.raw"

    def test_real_config_produces_output_file(self, real_cfg):
        """simulate() with real config must create lob_events.csv on disk."""
        simulate(real_cfg)
        output = ROOT / real_cfg["data"]["paths"]["raw"]
        assert output.exists(), f"Expected output file not found: {output}"

    def test_real_config_row_count(self, real_cfg):
        """Total rows must equal num_traders × num_events_per_trader."""
        simulate(real_cfg)
        output = ROOT / real_cfg["data"]["paths"]["raw"]
        df = pd.read_csv(output)
        sim = real_cfg["data"]["simulation"]
        expected = sim["num_traders"] * sim["num_events_per_trader"]
        assert len(df) == expected, (
            f"Expected {expected} rows, got {len(df)}"
        )

    def test_real_config_manipulator_count(self, real_cfg):
        """Exact number of unique manipulator traders must match floor(n × fraction)."""
        simulate(real_cfg)
        output = ROOT / real_cfg["data"]["paths"]["raw"]
        df = pd.read_csv(output)
        sim = real_cfg["data"]["simulation"]
        expected_manip = math.floor(
            sim["num_traders"] * sim["manipulation_fraction"]
        )
        actual_manip = df[df["label"] == 1]["trader_id"].nunique()
        assert actual_manip == expected_manip, (
            f"Expected {expected_manip} manipulator traders, got {actual_manip}"
        )

    def test_real_config_schema(self, real_cfg):
        """Output CSV produced from real config must have all required columns."""
        simulate(real_cfg)
        output = ROOT / real_cfg["data"]["paths"]["raw"]
        df = pd.read_csv(output)
        required_columns = {
            "trader_id", "event_type", "order_size",
            "timestamp", "price", "mid_price", "time_to_cancel", "label",
        }
        missing = required_columns - set(df.columns)
        assert not missing, f"Output CSV missing columns: {missing}"
