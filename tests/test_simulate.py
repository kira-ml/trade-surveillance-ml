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


class TestDistributions:
    """Statistical checks: class-conditional parameters must produce distinct ML signal."""

    @pytest.fixture()
    def large_cfg(self, tmp_path):
        """Larger dataset for statistically stable assertions."""
        return {
            "data": {
                "simulation": {
                    "num_traders": 200,
                    "num_events_per_trader": 200,
                    "manipulation_fraction": 0.30,
                    "poisson_lambda": 5.0,
                    "price_mean": 100.0,
                    "price_std": 2.0,
                    "cancellation_rate_normal": 0.25,
                    "cancellation_rate_manipulator": 0.80,
                    "seed": 0,
                },
                "paths": {
                    "raw": str(tmp_path / "dist_test" / "lob_events.csv"),
                    "processed_dir": str(tmp_path / "processed/"),
                    "scaler_params": str(tmp_path / "processed" / "scaler_params.json"),
                    "inference_input": str(tmp_path / "raw" / "new_lob_events.csv"),
                    "inference_output": str(tmp_path / "processed" / "inference_predictions.csv"),
                },
            }
        }

    def _load(self, cfg) -> pd.DataFrame:
        simulate(cfg)
        return pd.read_csv(cfg["data"]["paths"]["raw"])

    def test_manipulator_higher_cancellation_rate(self, large_cfg):
        """_EVENT_PROBS_MANIP cancellation = 0.50 vs _EVENT_PROBS_NORMAL = 0.25."""
        df = self._load(large_cfg)
        cancel_rate = df.groupby("label")["event_type"].apply(
            lambda s: (s == "cancellation").mean()
        )
        assert cancel_rate[1] > cancel_rate[0], (
            "Manipulators should have a higher cancellation rate than normals"
        )

    def test_manipulator_price_tighter_around_mid(self, large_cfg):
        """_PRICE_STD_MANIP = 0.5 vs _PRICE_STD_NORMAL = 2.0."""
        df = self._load(large_cfg)
        price_std = df.groupby("label")["price"].std()
        assert price_std[1] < price_std[0], (
            "Manipulator prices should cluster tighter around mid_price"
        )

    def test_manipulator_shorter_time_to_cancel(self, large_cfg):
        """_CANCEL_MEAN_MANIP = 2.0 vs _CANCEL_MEAN_NORMAL = 30.0."""
        df = self._load(large_cfg)
        cancels = df[df["event_type"] == "cancellation"]
        mean_ttc = cancels.groupby("label")["time_to_cancel"].mean()
        assert mean_ttc[1] < mean_ttc[0], (
            "Manipulator time_to_cancel should be shorter than normals"
        )

    def test_manipulator_denser_order_arrivals(self, large_cfg):
        """_LAMBDA_MANIP = 15.0 (3× normal) → smaller mean inter-arrival gap."""
        df = self._load(large_cfg)

        per_trader = (
            df.groupby(["trader_id", "label"])["timestamp"]
            .apply(lambda ts: ts.sort_values().diff().dropna().mean())
            .reset_index(name="mean_gap")
        )
        avg_gap = per_trader.groupby("label")["mean_gap"].mean()
        assert avg_gap[1] < avg_gap[0], (
            "Manipulators should have shorter mean inter-arrival times (denser activity)"
        )


class TestEdgeCases:
    """Edge-case inputs that are valid but sit at boundary conditions."""

    def _make_cfg(self, tmp_path, tag="edge", **overrides):
        sim = {
            "num_traders": 10,
            "num_events_per_trader": 5,
            "manipulation_fraction": 0.10,
            "poisson_lambda": 5.0,
            "price_mean": 100.0,
            "price_std": 2.0,
            "cancellation_rate_normal": 0.25,
            "cancellation_rate_manipulator": 0.80,
            "seed": 1,
        }
        sim.update(overrides)
        return {
            "data": {
                "simulation": sim,
                "paths": {
                    "raw": str(tmp_path / tag / "lob_events.csv"),
                    "processed_dir": str(tmp_path / "processed/"),
                    "scaler_params": str(tmp_path / "processed" / "scaler_params.json"),
                    "inference_input": str(tmp_path / "raw" / "new_lob_events.csv"),
                    "inference_output": str(tmp_path / "processed" / "inference_predictions.csv"),
                },
            }
        }

    def test_zero_manipulators_from_rounding_no_crash(self, tmp_path):
        """floor(3 × 0.10) == 0: should not crash; all labels must be 0."""
        cfg = self._make_cfg(
            tmp_path, tag="zero_manip", num_traders=3, manipulation_fraction=0.10
        )
        simulate(cfg)
        df = pd.read_csv(cfg["data"]["paths"]["raw"])
        assert (df["label"] == 0).all(), "Expected all-normal dataset when floor rounds to 0"

    def test_high_manipulation_fraction(self, tmp_path):
        """floor(10 × 0.99) == 9 manipulators out of 10 traders."""
        cfg = self._make_cfg(
            tmp_path, tag="high_manip", num_traders=10, manipulation_fraction=0.99
        )
        simulate(cfg)
        df = pd.read_csv(cfg["data"]["paths"]["raw"])
        expected_manip = math.floor(10 * 0.99)
        actual_manip = df[df["label"] == 1]["trader_id"].nunique()
        assert actual_manip == expected_manip

    def test_deeply_nested_output_path_created(self, tmp_path):
        """mkdir(parents=True) must create all intermediate directories."""
        deep_path = tmp_path / "a" / "b" / "c" / "d" / "lob_events.csv"
        cfg = self._make_cfg(tmp_path, tag="deep")
        cfg["data"]["paths"]["raw"] = str(deep_path)
        simulate(cfg)
        assert deep_path.exists(), "Deeply nested output path should be created automatically"
