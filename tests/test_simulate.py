"""
tests/test_simulate.py
────────────────────────────────────────────────────────────────────────────────
Unit tests for src/data/simulate.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.simulate import simulate

# ── Minimal valid config fixture ──────────────────────────────────────────────

@pytest.fixture()
def base_cfg(tmp_path):
    """Minimal config dict that routes output to a temp directory."""
    return {
        "data": {
            "simulation": {
                "num_traders": 50,
                "num_events_per_trader": 20,
                "manipulation_fraction": 0.10,
                "poisson_lambda": 5.0,
                "price_mean": 100.0,
                "price_std": 2.0,
                "cancellation_rate_normal": 0.25,
                "cancellation_rate_manipulator": 0.80,
                "seed": 42,
            },
            "paths": {
                "raw": str(tmp_path / "data" / "raw" / "lob_events.csv"),
                "processed_dir": str(tmp_path / "data" / "processed/"),
                "scaler_params": str(tmp_path / "data" / "processed" / "scaler_params.json"),
                "inference_input": str(tmp_path / "data" / "raw" / "new_lob_events.csv"),
                "inference_output": str(tmp_path / "data" / "processed" / "inference_predictions.csv"),
            },
        }
    }


def load_output(cfg) -> pd.DataFrame:
    """Helper: run simulate and return the resulting DataFrame."""
    simulate(cfg)
    return pd.read_csv(cfg["data"]["paths"]["raw"])


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestOutputFile:
    def test_csv_is_created(self, base_cfg):
        simulate(base_cfg)
        assert Path(base_cfg["data"]["paths"]["raw"]).exists()

    def test_csv_is_non_empty(self, base_cfg):
        df = load_output(base_cfg)
        assert len(df) > 0


class TestSchema:
    EXPECTED_COLUMNS = {
        "trader_id", "event_type", "order_size",
        "timestamp", "price", "mid_price", "time_to_cancel", "label",
    }

    def test_all_columns_present(self, base_cfg):
        df = load_output(base_cfg)
        assert self.EXPECTED_COLUMNS.issubset(df.columns)

    def test_label_is_binary(self, base_cfg):
        df = load_output(base_cfg)
        assert set(df["label"].unique()).issubset({0, 1})

    def test_event_type_values(self, base_cfg):
        df = load_output(base_cfg)
        assert set(df["event_type"].unique()).issubset(
            {"placement", "cancellation", "execution"}
        )

    def test_order_size_dtype_and_min(self, base_cfg):
        df = load_output(base_cfg)
        assert pd.api.types.is_integer_dtype(df["order_size"])
        assert df["order_size"].min() >= 1


class TestRowCount:
    def test_total_row_count(self, base_cfg):
        sim = base_cfg["data"]["simulation"]
        df = load_output(base_cfg)
        expected = sim["num_traders"] * sim["num_events_per_trader"]
        assert len(df) == expected


class TestLabelAssignment:
    def test_exact_manipulator_count(self, base_cfg):
        sim = base_cfg["data"]["simulation"]
        df = load_output(base_cfg)
        expected_manip_traders = math.floor(
            sim["num_traders"] * sim["manipulation_fraction"]
        )
        actual_manip_traders = df[df["label"] == 1]["trader_id"].nunique()
        assert actual_manip_traders == expected_manip_traders

    def test_each_trader_has_single_label(self, base_cfg):
        df = load_output(base_cfg)
        labels_per_trader = df.groupby("trader_id")["label"].nunique()
        assert (labels_per_trader == 1).all()


class TestTimeToCancel:
    def test_nan_for_non_cancellations(self, base_cfg):
        df = load_output(base_cfg)
        non_cancel = df[df["event_type"] != "cancellation"]
        assert non_cancel["time_to_cancel"].isna().all()

    def test_non_nan_for_cancellations(self, base_cfg):
        df = load_output(base_cfg)
        cancels = df[df["event_type"] == "cancellation"]
        assert cancels["time_to_cancel"].notna().all()

    def test_time_to_cancel_positive(self, base_cfg):
        df = load_output(base_cfg)
        cancels = df[df["event_type"] == "cancellation"]
        assert (cancels["time_to_cancel"] > 0).all()


class TestReproducibility:
    def test_same_seed_produces_identical_output(self, base_cfg, tmp_path):
        # Second run with same seed written to a different path
        cfg2 = {
            "data": {
                "simulation": base_cfg["data"]["simulation"].copy(),
                "paths": {
                    **base_cfg["data"]["paths"],
                    "raw": str(tmp_path / "run2" / "lob_events.csv"),
                },
            }
        }
        df1 = load_output(base_cfg)
        df2 = load_output(cfg2)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_produces_different_output(self, base_cfg, tmp_path):
        sim2 = {**base_cfg["data"]["simulation"], "seed": 999}
        cfg2 = {
            "data": {
                "simulation": sim2,
                "paths": {
                    **base_cfg["data"]["paths"],
                    "raw": str(tmp_path / "run_diff" / "lob_events.csv"),
                },
            }
        }
        df1 = load_output(base_cfg)
        df2 = load_output(cfg2)
        assert not df1["price"].equals(df2["price"])


class TestInputValidation:
    def _cfg_with(self, base_cfg, tmp_path, **overrides):
        sim = {**base_cfg["data"]["simulation"], **overrides}
        return {
            "data": {
                "simulation": sim,
                "paths": {
                    **base_cfg["data"]["paths"],
                    "raw": str(tmp_path / "val_test" / "lob_events.csv"),
                },
            }
        }

    def test_zero_traders_raises(self, base_cfg, tmp_path):
        with pytest.raises(ValueError, match="num_traders"):
            simulate(self._cfg_with(base_cfg, tmp_path, num_traders=0))

    def test_zero_events_raises(self, base_cfg, tmp_path):
        with pytest.raises(ValueError, match="num_events_per_trader"):
            simulate(self._cfg_with(base_cfg, tmp_path, num_events_per_trader=0))

    def test_fraction_zero_raises(self, base_cfg, tmp_path):
        with pytest.raises(ValueError, match="manipulation_fraction"):
            simulate(self._cfg_with(base_cfg, tmp_path, manipulation_fraction=0.0))

    def test_fraction_one_raises(self, base_cfg, tmp_path):
        with pytest.raises(ValueError, match="manipulation_fraction"):
            simulate(self._cfg_with(base_cfg, tmp_path, manipulation_fraction=1.0))

    def test_negative_mid_price_raises(self, base_cfg, tmp_path):
        with pytest.raises(ValueError, match="price_mean"):
            simulate(self._cfg_with(base_cfg, tmp_path, price_mean=-1.0))
