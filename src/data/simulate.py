"""
src/data/simulate.py
────────────────────────────────────────────────────────────────────────────────
Synthetic limit-order-book (LOB) event dataset generator.

Generates per-trader order events using statistically grounded stochastic
processes:
  - Homogeneous Poisson process for order arrival times
  - Gaussian distribution (class-conditional σ) for order prices
  - Log-Normal distribution for order sizes
  - Categorical distribution (class-conditional p) for event types
  - Exponential distribution (class-conditional μ) for time-to-cancel

Writes a flat CSV of raw LOB events to the path specified in config.

Responsibilities
────────────────
  ✓ Generate synthetic LOB events with class-conditional parameters
  ✓ Save raw events to data/raw/lob_events.csv
  ✓ Log summary statistics after saving

Non-responsibilities
────────────────────
  ✗ Feature aggregation      → engineer.py
  ✗ Schema validation        → validate.py
  ✗ Normalization / scaling  → preprocess.py
  ✗ Train/test splitting     → preprocess.py
  ✗ Config loading           → receives cfg dict from main.py
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Class-conditional constants not exposed in config.yaml ───────────────────

# Poisson arrival rates (orders per unit time)
_LAMBDA_NORMAL: Final[float] = 5.0   # from config poisson_lambda
_LAMBDA_MANIP: Final[float] = 15.0   # 3× normal — burst activity of spoofing

# Gaussian price std (distance from mid-price)
_PRICE_STD_NORMAL: Final[float] = 2.0   # broad — reasonable distance from mid
_PRICE_STD_MANIP: Final[float] = 0.5    # tight — clustered near best bid/ask

# Log-Normal order size parameters  (size = exp(μ_s + σ_s · Z))
_SIZE_MU_NORMAL: Final[float] = 4.0
_SIZE_SIGMA_NORMAL: Final[float] = 0.8
_SIZE_MU_MANIP: Final[float] = 3.0
_SIZE_SIGMA_MANIP: Final[float] = 0.5

# Event-type categorical probabilities  [placement, cancellation, execution]
# Probabilities reflect spoofing signature:
#   high placement + high cancellation + near-zero execution for manipulators
_EVENT_PROBS_NORMAL: Final[list[float]] = [0.50, 0.25, 0.25]
_EVENT_PROBS_MANIP: Final[list[float]] = [0.45, 0.50, 0.05]
_EVENT_TYPES: Final[list[str]] = ["placement", "cancellation", "execution"]

# Exponential time-to-cancel mean (seconds)
_CANCEL_MEAN_NORMAL: Final[float] = 30.0   # unhurried cancellation
_CANCEL_MEAN_MANIP: Final[float] = 2.0     # 15× faster — strong ML signal


# ── Internal helpers ──────────────────────────────────────────────────────────

def _assign_labels(
    num_traders: int,
    manipulation_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a binary label array of shape (num_traders,).

    Exactly ``floor(num_traders × manipulation_fraction)`` traders receive
    label 1 (manipulator); the rest receive label 0 (normal).  Labels are
    shuffled so manipulators are not contiguous.

    Parameters
    ----------
    num_traders:
        Total number of traders to generate.
    manipulation_fraction:
        Fraction of traders that are manipulators, in (0, 1).
    rng:
        Seeded NumPy random generator.

    Returns
    -------
    np.ndarray of dtype int8, shape (num_traders,).
    """
    num_manip = math.floor(num_traders * manipulation_fraction)
    labels = np.zeros(num_traders, dtype=np.int8)
    labels[:num_manip] = 1
    rng.shuffle(labels)
    return labels


def _generate_timestamps(
    num_events: int,
    lam: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``num_events`` cumulative arrival timestamps via Exp(1/λ).

    Inter-arrival times ~ Exponential(1/λ).  Cumulative sum gives absolute
    timestamps starting from 0.

    Parameters
    ----------
    num_events:
        Number of order arrivals to generate.
    lam:
        Poisson arrival rate (λ).
    rng:
        Seeded NumPy random generator.

    Returns
    -------
    np.ndarray of float64, shape (num_events,), monotonically increasing.
    """
    inter_arrivals = rng.exponential(scale=1.0 / lam, size=num_events)
    return np.cumsum(inter_arrivals)


def _generate_prices(
    num_events: int,
    mid_price: float,
    price_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample order prices from N(mid_price, price_std²).

    Parameters
    ----------
    num_events:
        Number of prices to sample.
    mid_price:
        Centre of the Gaussian (fixed scalar mid-price).
    price_std:
        Class-conditional standard deviation.
    rng:
        Seeded NumPy random generator.

    Returns
    -------
    np.ndarray of float64, shape (num_events,).
    """
    return rng.normal(loc=mid_price, scale=price_std, size=num_events)


def _generate_order_sizes(
    num_events: int,
    mu_s: float,
    sigma_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample order sizes from LogNormal(μ_s, σ_s²), floored to integers.

    Parameters
    ----------
    num_events:
        Number of sizes to sample.
    mu_s:
        Log-scale mean of the Log-Normal distribution.
    sigma_s:
        Log-scale standard deviation of the Log-Normal distribution.
    rng:
        Seeded NumPy random generator.

    Returns
    -------
    np.ndarray of int32, shape (num_events,).  Minimum value is 1.
    """
    raw = rng.lognormal(mean=mu_s, sigma=sigma_s, size=num_events)
    return np.maximum(np.floor(raw).astype(np.int32), 1)


def _generate_event_types(
    num_events: int,
    probs: list[float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample event types from a Categorical distribution.

    Parameters
    ----------
    num_events:
        Number of events to sample.
    probs:
        Probability vector [p_placement, p_cancellation, p_execution].
    rng:
        Seeded NumPy random generator.

    Returns
    -------
    np.ndarray of str, shape (num_events,), values in _EVENT_TYPES.
    """
    indices = rng.choice(len(_EVENT_TYPES), size=num_events, p=probs)
    return np.array(_EVENT_TYPES)[indices]


def _generate_time_to_cancel(
    event_types: np.ndarray,
    cancel_mean: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample time-to-cancel for cancellation events; NaN otherwise.

    For each event labelled "cancellation", sample from Exp(1/μ_τ).
    All other events receive np.nan.

    Parameters
    ----------
    event_types:
        Array of event type strings, shape (num_events,).
    cancel_mean:
        Class-conditional mean cancel time (μ_τ) in seconds.
    rng:
        Seeded NumPy random generator.

    Returns
    -------
    np.ndarray of float64, shape (num_events,).
    """
    mask = event_types == "cancellation"
    num_cancels = int(mask.sum())
    times = np.full(len(event_types), np.nan, dtype=np.float64)
    if num_cancels > 0:
        times[mask] = rng.exponential(scale=cancel_mean, size=num_cancels)
    return times


def _build_trader_events(
    trader_id: int,
    label: int,
    num_events: int,
    mid_price: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate all LOB events for a single trader.

    Parameters
    ----------
    trader_id:
        Integer trader identifier.
    label:
        0 for normal, 1 for manipulator.
    num_events:
        Number of order events to generate.
    mid_price:
        Fixed scalar mid-price used as the Gaussian mean.
    rng:
        Seeded NumPy random generator (shared across all traders).

    Returns
    -------
    pd.DataFrame with columns: trader_id, event_type, order_size, timestamp,
    price, mid_price, time_to_cancel, label.
    """
    is_manip = label == 1

    lam = _LAMBDA_MANIP if is_manip else _LAMBDA_NORMAL
    price_std = _PRICE_STD_MANIP if is_manip else _PRICE_STD_NORMAL
    size_mu = _SIZE_MU_MANIP if is_manip else _SIZE_MU_NORMAL
    size_sigma = _SIZE_SIGMA_MANIP if is_manip else _SIZE_SIGMA_NORMAL
    event_probs = _EVENT_PROBS_MANIP if is_manip else _EVENT_PROBS_NORMAL
    cancel_mean = _CANCEL_MEAN_MANIP if is_manip else _CANCEL_MEAN_NORMAL

    timestamps = _generate_timestamps(num_events, lam, rng)
    prices = _generate_prices(num_events, mid_price, price_std, rng)
    sizes = _generate_order_sizes(num_events, size_mu, size_sigma, rng)
    event_types = _generate_event_types(num_events, event_probs, rng)
    ttc = _generate_time_to_cancel(event_types, cancel_mean, rng)

    return pd.DataFrame(
        {
            "trader_id": trader_id,
            "event_type": event_types,
            "order_size": sizes,
            "timestamp": np.round(timestamps, 6),
            "price": np.round(prices, 4),
            "mid_price": mid_price,
            "time_to_cancel": ttc,
            "label": label,
        }
    )


# ── Public API ────────────────────────────────────────────────────────────────

def simulate(cfg: dict) -> None:
    """Generate a synthetic LOB event dataset and save it to CSV.

    Reads all simulation parameters from the validated config dict passed by
    ``main.py``.  Does not load config itself.

    Parameters
    ----------
    cfg:
        Validated configuration dictionary (mirrors configs/config.yaml).
        Expected keys used:
          cfg["data"]["simulation"]["num_traders"]
          cfg["data"]["simulation"]["num_events_per_trader"]
          cfg["data"]["simulation"]["manipulation_fraction"]
          cfg["data"]["simulation"]["poisson_lambda"]   (normal λ; manip = 3×)
          cfg["data"]["simulation"]["price_mean"]
          cfg["data"]["simulation"]["price_std"]
          cfg["data"]["simulation"]["seed"]
          cfg["data"]["paths"]["raw"]

    Raises
    ------
    ValueError
        If simulation parameters are out of valid ranges.
    OSError
        If the output directory cannot be created or the file cannot be written.
    """
    sim_cfg = cfg["data"]["simulation"]
    paths_cfg = cfg["data"]["paths"]

    num_traders: int = int(sim_cfg["num_traders"])
    num_events: int = int(sim_cfg["num_events_per_trader"])
    manip_fraction: float = float(sim_cfg["manipulation_fraction"])
    mid_price: float = float(sim_cfg["price_mean"])
    seed: int = int(sim_cfg["seed"])
    output_path: Path = Path(paths_cfg["raw"])

    # ── Parameter validation ─────────────────────────────────────────────────
    if num_traders <= 0:
        raise ValueError(f"num_traders must be positive, got {num_traders}.")
    if num_events <= 0:
        raise ValueError(f"num_events_per_trader must be positive, got {num_events}.")
    if not 0.0 < manip_fraction < 1.0:
        raise ValueError(
            f"manipulation_fraction must be in (0, 1), got {manip_fraction}."
        )
    if mid_price <= 0.0:
        raise ValueError(f"price_mean (mid_price) must be positive, got {mid_price}.")

    logger.info(
        "Starting LOB simulation | traders=%d | events_per_trader=%d | "
        "manipulation_fraction=%.2f | seed=%d",
        num_traders,
        num_events,
        manip_fraction,
        seed,
    )

    # ── Initialise reproducible RNG ──────────────────────────────────────────
    rng = np.random.default_rng(seed)

    # ── Assign trader labels ──────────────────────────────────────────────────
    labels = _assign_labels(num_traders, manip_fraction, rng)
    num_manip = int(labels.sum())
    num_normal = num_traders - num_manip

    logger.info(
        "Trader allocation | normal=%d | manipulator=%d | class_ratio=%.4f",
        num_normal,
        num_manip,
        num_manip / num_traders,
    )

    # ── Generate events for every trader ────────────────────────────────────
    frames: list[pd.DataFrame] = []
    for trader_id in range(num_traders):
        df = _build_trader_events(
            trader_id=trader_id,
            label=int(labels[trader_id]),
            num_events=num_events,
            mid_price=mid_price,
            rng=rng,
        )
        frames.append(df)

    dataset: pd.DataFrame = pd.concat(frames, ignore_index=True)

    # ── Persist to CSV ────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    total_events = len(dataset)
    manip_events = int((dataset["label"] == 1).sum())

    logger.info(
        "Dataset saved | path=%s | total_events=%d | manip_events=%d "
        "| normal_events=%d | event_class_ratio=%.4f",
        output_path,
        total_events,
        manip_events,
        total_events - manip_events,
        manip_events / total_events,
    )
