from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .math import ParticleFilterConfig, compute_particle_filter_frame

FINAL_DATASET_COLUMNS = [
    "contract_start",
    "contract_end",
    "interval_start_price",
    "pf_fair_value_at_entry",
    "pf_distance",
    "actual_gap",
    "entry_price",
    "exit_price",
    "confidence",
    "binary_win",
    "trade_side",
    "gross_pnl_per_trade",
    "fee",
    "net_pnl_per_trade",
    "skip_reason",
]


@dataclass(frozen=True)
class BacktestConfig:
    market_up_price: float = 0.50
    market_down_price: float = 0.50
    contracts_per_trade: float = 1.0
    particle_filter_particles: int = 300
    particle_filter_lookback: int = 240
    particle_filter_resample_threshold: float = 0.50
    particle_filter_use_regime_context: bool = False
    min_abs_gap: float = 25.0
    min_pf_confidence: float = 0.01


def normalize_candles(candles: pd.DataFrame) -> pd.DataFrame:
    """Return a UTC-indexed OHLCV frame sorted by candle start time."""

    if candles.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    frame = candles.copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp")
    elif not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, utc=True)
    elif frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")

    required = ["open", "high", "low", "close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"candles are missing required columns: {missing}")

    numeric_cols = [column for column in ["open", "high", "low", "close", "volume"] if column in frame.columns]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if "volume" not in frame.columns:
        frame["volume"] = 0.0

    frame = frame.sort_index().drop_duplicates(keep="last")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    return frame[["open", "high", "low", "close", "volume"]]


def build_interval_frame(
    candles_15m: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Build one row per 15m contract using only information known at entry.

    Lookahead prevention:
    - The row for contract start ``T`` uses the PF snapshot from the previous
      completed 15-minute bar.
    - That PF snapshot is the latest fair value available exactly at ``T``.
    - The current bar's close is used only as the realized settlement price.
    """

    active_config = config or BacktestConfig()
    candles = normalize_candles(candles_15m)
    if candles.empty or len(candles) < 2:
        return pd.DataFrame()

    pf_lookback = (
        max(int(active_config.particle_filter_lookback), len(candles))
        if int(active_config.particle_filter_lookback) > 0
        else 0
    )
    pf_frame = compute_particle_filter_frame(
        candles_15m=candles,
        config=ParticleFilterConfig(
            num_particles=active_config.particle_filter_particles,
            resample_threshold=active_config.particle_filter_resample_threshold,
            lookback=pf_lookback,
            use_regime_context=active_config.particle_filter_use_regime_context,
        ),
    )
    if pf_frame.empty:
        return pd.DataFrame()

    bar_delta = _infer_bar_delta(candles.index)
    rows: list[dict[str, object]] = []
    for row_number in range(1, len(candles)):
        signal_row = pf_frame.iloc[row_number - 1]
        entry_row = candles.iloc[row_number]
        contract_start = candles.index[row_number]

        rows.append(
            {
                "contract_start": contract_start,
                "contract_end": contract_start + bar_delta,
                "interval_start_price": float(entry_row["open"]),
                "pf_fair_value_at_entry": _finite_or_nan(signal_row.get("pf_fair_price")),
                "entry_price": float(entry_row["open"]),
                "exit_price": float(entry_row["close"]),
                "confidence": _finite_or_nan(signal_row.get("pf_confidence")),
            }
        )

    frame = pd.DataFrame(rows)
    return compute_pf_distance(frame)


def build_interval_signal_frame(
    candles_15m: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Backward-compatible alias for interval frame construction."""

    return build_interval_frame(candles_15m, config=config)


def build_interval_signal_dataframe(
    candles_15m: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Backward-compatible alias for interval frame construction."""

    return build_interval_frame(candles_15m, config=config)


def build_backtest_signal_frame(
    candles_15m: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Backward-compatible alias for interval frame construction."""

    return build_interval_frame(candles_15m, config=config)


def compute_pf_distance(interval_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute PF fair-value distance versus the interval start price."""

    frame = interval_frame.copy()
    frame["pf_distance"] = frame["pf_fair_value_at_entry"] - frame["interval_start_price"]
    frame["actual_gap"] = frame["pf_distance"].abs()
    return frame


def generate_signal_from_pf_distance(
    interval_frame: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Generate raw UP/DOWN/NO_SIGNAL decisions from the PF distance threshold."""

    active_config = config or BacktestConfig()
    frame = interval_frame.copy()
    frame["trade_side"] = "NO_TRADE"

    pf_distance = pd.to_numeric(frame["pf_distance"], errors="coerce")
    confidence = pd.to_numeric(frame["confidence"], errors="coerce")
    confidence_ok = confidence >= float(active_config.min_pf_confidence)
    up_mask = pf_distance >= float(active_config.min_abs_gap)
    down_mask = pf_distance <= -float(active_config.min_abs_gap)

    frame.loc[up_mask & confidence_ok, "trade_side"] = "UP"
    frame.loc[down_mask & confidence_ok, "trade_side"] = "DOWN"
    return frame


def generate_pf_gap_signal(
    interval_frame: pd.DataFrame,
    config: BacktestConfig | None = None,
    **_: object,
) -> pd.DataFrame:
    """Backward-compatible wrapper for the stricter PF-distance signal."""

    return generate_signal_from_pf_distance(interval_frame, config=config)


def generate_threshold_signals(
    interval_frame: pd.DataFrame,
    config: BacktestConfig | None = None,
    **_: object,
) -> pd.DataFrame:
    """Backward-compatible alias for PF-distance signal generation."""

    return generate_signal_from_pf_distance(interval_frame, config=config)


def apply_hour_filter(
    signal_frame: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Backward-compatible no-op: hour filtering is disabled in this version."""

    frame = signal_frame.copy()
    frame["blocked_hour_flag"] = False
    return frame


def apply_three_loss_pause(
    signal_frame: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Run the 3-loss pause as a simple trade-loop state machine."""

    cooldown_length = 1

    rows: list[dict[str, object]] = []
    consecutive_losses = 0
    cooldown_remaining = 0

    for row in signal_frame.to_dict(orient="records"):
        trade_side = str(row["trade_side"])
        eligible_signal = trade_side in {"UP", "DOWN"}
        skip_reason = "NO_SIGNAL"
        trade_taken = False
        binary_win = float("nan")
        actual_gap = float(row.get("actual_gap", abs(float(row.get("pf_distance", float("nan"))))))

        if not eligible_signal:
            skip_reason = "NO_SIGNAL"
        elif cooldown_remaining > 0:
            skip_reason = "COOLDOWN"
        else:
            skip_reason = "TRADE_EXECUTED"
            trade_taken = True
            binary_win = _binary_win_from_trade(trade_side, row["entry_price"], row["exit_price"])

            if binary_win >= 1.0:
                consecutive_losses = 0
            else:
                consecutive_losses += 1
                if consecutive_losses >= 3:
                    cooldown_remaining = cooldown_length
                    consecutive_losses = 0

        rows.append(
            {
                "contract_start": row["contract_start"],
                "contract_end": row["contract_end"],
                "interval_start_price": row["interval_start_price"],
                "pf_fair_value_at_entry": row["pf_fair_value_at_entry"],
                "pf_distance": row["pf_distance"],
                "actual_gap": actual_gap,
                "entry_price": row["entry_price"],
                "exit_price": row["exit_price"],
                "confidence": row["confidence"],
                "binary_win": binary_win,
                "trade_side": trade_side,
                "skip_reason": skip_reason,
                "_trade_taken": int(trade_taken),
            }
        )

        if skip_reason == "COOLDOWN":
            cooldown_remaining = max(cooldown_remaining - 1, 0)

    return pd.DataFrame(rows)


def apply_three_loss_cooldown(
    signal_frame: pd.DataFrame,
    config: BacktestConfig | None = None,
    **_: object,
) -> pd.DataFrame:
    """Backward-compatible alias for the 3-loss pause."""

    return apply_three_loss_pause(signal_frame, config=config)


def kalshi_taker_fee(
    contracts: float,
    price: float,
    *,
    fee_multiplier: float = 1.0,
) -> float:
    """Return Kalshi taker fee dollars, rounded up to the nearest cent."""

    contracts_value = float(contracts)
    price_value = float(price)
    if contracts_value <= 0 or not np.isfinite(contracts_value):
        return 0.0
    if not np.isfinite(price_value):
        return 0.0
    if not 0.0 <= price_value <= 1.0:
        raise ValueError("price must be between 0 and 1 inclusive")

    raw_fee = 0.07 * fee_multiplier * contracts_value * price_value * (1.0 - price_value)
    return float(math.ceil(max(raw_fee, 0.0) * 100.0 - 1e-12) / 100.0)


def evaluate_trade_pnl(
    trade_frame: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Evaluate per-trade binary PnL with Kalshi taker fees."""

    active_config = config or BacktestConfig()
    rows: list[dict[str, object]] = []

    for row in trade_frame.to_dict(orient="records"):
        trade_side = str(row["trade_side"])
        trade_taken = int(row.get("_trade_taken", 0)) == 1
        binary_win = pd.NA
        gross_pnl_per_trade = float("nan")
        fee = float("nan")
        net_pnl_per_trade = float("nan")
        actual_gap = float(row.get("actual_gap", abs(float(row.get("pf_distance", float("nan"))))))

        if trade_taken:
            contract_price = _contract_price_for_side(trade_side, active_config)
            binary_win = int(_binary_win_from_trade(trade_side, row["entry_price"], row["exit_price"]))
            gross_pnl_per_trade = active_config.contracts_per_trade * (
                binary_win - contract_price
            )
            fee = kalshi_taker_fee(active_config.contracts_per_trade, contract_price)
            net_pnl_per_trade = gross_pnl_per_trade - fee

        rows.append(
            {
                "contract_start": row["contract_start"],
                "contract_end": row["contract_end"],
                "interval_start_price": row["interval_start_price"],
                "pf_fair_value_at_entry": row["pf_fair_value_at_entry"],
                "pf_distance": row["pf_distance"],
                "actual_gap": actual_gap,
                "entry_price": row["entry_price"],
                "exit_price": row["exit_price"],
                "confidence": row["confidence"],
                "binary_win": binary_win,
                "trade_side": trade_side,
                "gross_pnl_per_trade": gross_pnl_per_trade,
                "fee": fee,
                "net_pnl_per_trade": net_pnl_per_trade,
                "skip_reason": row["skip_reason"],
            }
        )

    results = pd.DataFrame(rows)
    results["binary_win"] = results["binary_win"].astype("Int64")
    return results[FINAL_DATASET_COLUMNS]


def evaluate_contract_pnl(
    signal_frame: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Backward-compatible alias for trade PnL evaluation."""

    return evaluate_trade_pnl(signal_frame, config=config)


def compute_drawdown_and_streaks(results: pd.DataFrame) -> dict[str, float | int]:
    """Compute max drawdown and longest streaks across executed trades."""

    trades = results.loc[results["skip_reason"] == "TRADE_EXECUTED"].copy()
    if trades.empty:
        return {
            "max_drawdown": 0.0,
            "longest_win_streak": 0,
            "longest_loss_streak": 0,
        }

    cumulative_net = pd.to_numeric(trades["net_pnl_per_trade"], errors="coerce").fillna(0.0).cumsum()
    running_peak = cumulative_net.cummax()
    max_drawdown = float((running_peak - cumulative_net).max()) if not cumulative_net.empty else 0.0

    longest_win_streak = 0
    longest_loss_streak = 0
    current_win_streak = 0
    current_loss_streak = 0
    for binary_win in pd.to_numeric(trades["binary_win"], errors="coerce"):
        if binary_win >= 1.0:
            current_win_streak += 1
            current_loss_streak = 0
            longest_win_streak = max(longest_win_streak, current_win_streak)
        else:
            current_loss_streak += 1
            current_win_streak = 0
            longest_loss_streak = max(longest_loss_streak, current_loss_streak)

    return {
        "max_drawdown": max_drawdown,
        "longest_win_streak": int(longest_win_streak),
        "longest_loss_streak": int(longest_loss_streak),
    }


def summarize_backtest(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize executed-trade performance."""

    if results.empty:
        return pd.DataFrame()

    trades = results.loc[results["skip_reason"] == "TRADE_EXECUTED"].copy()
    streaks = compute_drawdown_and_streaks(results)
    return pd.DataFrame(
        [
            {
                "total_trades": int(len(trades)),
                "win_rate": float(pd.to_numeric(trades["binary_win"], errors="coerce").mean()) if not trades.empty else np.nan,
                "gross_pnl": float(pd.to_numeric(trades["gross_pnl_per_trade"], errors="coerce").sum()) if not trades.empty else 0.0,
                "total_fees": float(pd.to_numeric(trades["fee"], errors="coerce").sum()) if not trades.empty else 0.0,
                "net_pnl": float(pd.to_numeric(trades["net_pnl_per_trade"], errors="coerce").sum()) if not trades.empty else 0.0,
                "average_net_pnl_per_trade": float(pd.to_numeric(trades["net_pnl_per_trade"], errors="coerce").mean()) if not trades.empty else np.nan,
                "max_drawdown": float(streaks["max_drawdown"]),
                "longest_win_streak": int(streaks["longest_win_streak"]),
                "longest_loss_streak": int(streaks["longest_loss_streak"]),
            }
        ]
    )


def summarize_skip_reasons(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize why rows were not traded."""

    if results.empty:
        return pd.DataFrame()

    skip_counts = results["skip_reason"].value_counts()
    return pd.DataFrame(
        [
            {
                "cooldown_skips": int(skip_counts.get("COOLDOWN", 0)),
                "no_signal_rows": int(skip_counts.get("NO_SIGNAL", 0)),
            }
        ]
    )


def run_backtest(
    candles_15m: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Run the lean PF-distance backtest and return the clean final dataset."""

    active_config = config or BacktestConfig()
    interval_frame = build_interval_frame(candles_15m, config=active_config)
    if interval_frame.empty:
        return pd.DataFrame(columns=FINAL_DATASET_COLUMNS)
    signal_frame = generate_signal_from_pf_distance(interval_frame, config=active_config)
    cooled = apply_three_loss_pause(signal_frame, config=active_config)
    return evaluate_trade_pnl(cooled, config=active_config)


def export_backtest_excel(
    output_path: str | Path,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    skip_summary: pd.DataFrame | None = None,
) -> Path:
    """Write the lean backtest outputs to Excel."""

    workbook_path = Path(output_path)
    excel_results = _excel_safe_frame(results)
    excel_summary = _excel_safe_frame(summary)
    excel_skip_summary = _excel_safe_frame(skip_summary) if skip_summary is not None else None
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    temp_workbook_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"{workbook_path.stem}_",
            suffix=workbook_path.suffix or ".xlsx",
            dir=tempfile.gettempdir(),
            delete=False,
        ) as temp_file:
            temp_workbook_path = Path(temp_file.name)

        with pd.ExcelWriter(temp_workbook_path, engine="openpyxl") as writer:
            excel_results.to_excel(writer, sheet_name="trade_log", index=False)
            excel_summary.to_excel(writer, sheet_name="summary", index=False)
            if excel_skip_summary is not None and not excel_skip_summary.empty:
                excel_skip_summary.to_excel(writer, sheet_name="skip_summary", index=False)
        os.replace(temp_workbook_path, workbook_path)
    except ModuleNotFoundError as exc:  # pragma: no cover
        if exc.name == "openpyxl":
            raise RuntimeError(
                "openpyxl is required to export Excel files. Install it with "
                "`python -m pip install openpyxl` and rerun the backtest."
            ) from exc
        raise
    finally:
        if temp_workbook_path is not None and temp_workbook_path.exists():
            try:
                temp_workbook_path.unlink()
            except OSError:
                pass

    return workbook_path


def _contract_price_for_side(trade_side: str, config: BacktestConfig) -> float:
    if trade_side == "UP":
        return float(config.market_up_price)
    if trade_side == "DOWN":
        return float(config.market_down_price)
    return float("nan")


def _binary_win_from_trade(trade_side: str, entry_price: float, exit_price: float) -> float:
    if trade_side == "UP":
        return float(exit_price > entry_price)
    if trade_side == "DOWN":
        return float(exit_price < entry_price)
    return float("nan")


def _finite_or_nan(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(numeric):
        return float("nan")
    return numeric


def _infer_bar_delta(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        return pd.Timedelta(minutes=15)
    deltas = index.to_series().diff().dropna()
    if deltas.empty:
        return pd.Timedelta(minutes=15)
    median_delta = deltas.median()
    if pd.isna(median_delta) or median_delta <= pd.Timedelta(0):
        return pd.Timedelta(minutes=15)
    return median_delta


def _excel_safe_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()

    safe = frame.copy()
    for column in safe.columns:
        series = safe[column]
        if isinstance(series.dtype, pd.DatetimeTZDtype):
            safe[column] = series.dt.tz_convert("UTC").dt.tz_localize(None)
    return safe
