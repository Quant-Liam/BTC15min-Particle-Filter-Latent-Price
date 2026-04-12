from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover
    missing = exc.name or "a required package"
    print(
        "Missing Python dependency: "
        f"{missing}\n"
        "Install the project requirements, then rerun the backtest:\n"
        "python3 -m pip install -r /Users/liamrodgers/Desktop/Python/Personal/requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

from btc15m import CoinbaseClient
from btc15m.backtest import (
    BacktestConfig,
    export_backtest_excel,
    normalize_candles,
    run_backtest,
    summarize_backtest,
    summarize_skip_reasons,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the lean 15m BTC PF-distance backtest with blocked hours, a 3-loss pause, and Kalshi taker fees."
    )
    parser.add_argument("--product-id", default="BTC-USD")
    parser.add_argument("--years", type=float, default=1.0)
    parser.add_argument("--granularity-15m", type=int, default=900)
    parser.add_argument("--candles-15m-csv", type=Path, default=None)
    parser.add_argument("--output-xlsx", type=Path, default=Path("pf_distance_backtest.xlsx"))
    parser.add_argument("--market-up-price", type=float, default=0.50)
    parser.add_argument("--market-down-price", type=float, default=0.50)
    parser.add_argument("--contracts-per-trade", type=float, default=1.0)
    parser.add_argument("--particle-filter-particles", type=int, default=300)
    parser.add_argument("--particle-filter-lookback", type=int, default=240)
    parser.add_argument("--particle-filter-resample-threshold", type=float, default=0.50)
    parser.add_argument("--use-regime-context", action="store_true")
    parser.add_argument("--min-abs-gap", type=float, default=25.0)
    parser.add_argument("--min-pf-confidence", type=float, default=0.01)
    parser.add_argument("--timeout", type=int, default=10)
    return parser.parse_args()


def estimate_candle_count(years: float, granularity: int) -> int:
    seconds = years * 365.25 * 24 * 60 * 60
    return int((seconds + granularity - 1) // granularity)


def load_15m_candles(args: argparse.Namespace) -> pd.DataFrame:
    if args.candles_15m_csv is not None:
        return normalize_candles(pd.read_csv(args.candles_15m_csv))

    bars = estimate_candle_count(args.years, args.granularity_15m)
    print(f"Fetching {bars} x 15m candles for {args.product_id} ({args.years:.2f} years)...")
    client = CoinbaseClient(timeout=args.timeout)
    return client.fetch_candles(
        product_id=args.product_id,
        granularity=args.granularity_15m,
        limit=bars,
    )


def main() -> None:
    args = parse_args()
    candles_15m = load_15m_candles(args)

    config = BacktestConfig(
        market_up_price=args.market_up_price,
        market_down_price=args.market_down_price,
        contracts_per_trade=args.contracts_per_trade,
        particle_filter_particles=args.particle_filter_particles,
        particle_filter_lookback=args.particle_filter_lookback,
        particle_filter_resample_threshold=args.particle_filter_resample_threshold,
        particle_filter_use_regime_context=args.use_regime_context,
        min_abs_gap=args.min_abs_gap,
        min_pf_confidence=args.min_pf_confidence,
    )

    results = run_backtest(candles_15m=candles_15m, config=config)
    summary = summarize_backtest(results)
    skip_summary = summarize_skip_reasons(results)
    output_path = export_backtest_excel(args.output_xlsx, results, summary, skip_summary=skip_summary)

    print("Backtest complete")
    print(f"Window: {candles_15m.index.min()} -> {candles_15m.index.max()}")
    print(f"Saved workbook: {output_path.resolve()}")
    print(results.to_string(index=False))
    if not summary.empty:
        print("\nSummary:")
        print(summary.to_string(index=False))
    if not skip_summary.empty:
        print("\nSkip summary:")
        print(skip_summary.to_string(index=False))


if __name__ == "__main__":
    main()
