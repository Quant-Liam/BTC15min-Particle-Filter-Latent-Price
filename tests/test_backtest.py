from __future__ import annotations

import unittest

import pandas as pd

from btc15m.backtest import (
    BacktestConfig,
    apply_hour_filter,
    apply_three_loss_pause,
    build_interval_frame,
    compute_pf_distance,
    evaluate_trade_pnl,
    generate_signal_from_pf_distance,
    kalshi_taker_fee,
    run_backtest,
    summarize_backtest,
    summarize_skip_reasons,
)


class BacktestTests(unittest.TestCase):
    def test_interval_frame_uses_prior_completed_bar_pf_snapshot(self) -> None:
        index = pd.date_range("2024-01-01 00:00:00+00:00", periods=8, freq="15min")
        candles = pd.DataFrame(
            {
                "open": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0, 107.0, 108.0],
                "high": [101.0, 104.0, 104.0, 105.0, 106.0, 108.0, 109.0, 110.0],
                "low": [99.0, 100.0, 101.0, 101.0, 103.0, 104.0, 106.0, 107.0],
                "close": [101.0, 103.0, 102.0, 104.0, 105.0, 107.0, 108.0, 109.0],
                "volume": [1.0] * 8,
            },
            index=index,
        )

        results = build_interval_frame(
            candles,
            config=BacktestConfig(
                particle_filter_use_regime_context=False,
                particle_filter_lookback=8,
            ),
        )

        self.assertEqual(len(results), len(candles) - 1)
        self.assertEqual(list(results["contract_start"]), list(index[1:]))

    def test_pf_distance_signal_uses_plus_minus_25_threshold_only(self) -> None:
        interval_frame = pd.DataFrame(
            [
                {"contract_start": pd.Timestamp("2024-01-01 00:00:00+00:00"), "contract_end": pd.Timestamp("2024-01-01 00:15:00+00:00"), "interval_start_price": 100.0, "pf_fair_value_at_entry": 175.0, "entry_price": 100.0, "exit_price": 101.0, "confidence": 0.4},
                {"contract_start": pd.Timestamp("2024-01-01 00:15:00+00:00"), "contract_end": pd.Timestamp("2024-01-01 00:30:00+00:00"), "interval_start_price": 100.0, "pf_fair_value_at_entry": 160.0, "entry_price": 100.0, "exit_price": 99.0, "confidence": 0.4},
                {"contract_start": pd.Timestamp("2024-01-01 00:30:00+00:00"), "contract_end": pd.Timestamp("2024-01-01 00:45:00+00:00"), "interval_start_price": 100.0, "pf_fair_value_at_entry": 20.0, "entry_price": 100.0, "exit_price": 99.0, "confidence": 0.4},
            ]
        )

        with_distance = compute_pf_distance(interval_frame)
        signals = generate_signal_from_pf_distance(with_distance, config=BacktestConfig())

        self.assertEqual(list(signals["trade_side"]), ["UP", "UP", "DOWN"])

    def test_pf_distance_signal_requires_minimum_pf_confidence(self) -> None:
        interval_frame = pd.DataFrame(
            [
                {"contract_start": pd.Timestamp("2024-01-01 00:00:00+00:00"), "contract_end": pd.Timestamp("2024-01-01 00:15:00+00:00"), "interval_start_price": 100.0, "pf_fair_value_at_entry": 175.0, "entry_price": 100.0, "exit_price": 101.0, "confidence": 0.009},
                {"contract_start": pd.Timestamp("2024-01-01 00:15:00+00:00"), "contract_end": pd.Timestamp("2024-01-01 00:30:00+00:00"), "interval_start_price": 100.0, "pf_fair_value_at_entry": 175.0, "entry_price": 100.0, "exit_price": 101.0, "confidence": 0.01},
                {"contract_start": pd.Timestamp("2024-01-01 00:30:00+00:00"), "contract_end": pd.Timestamp("2024-01-01 00:45:00+00:00"), "interval_start_price": 100.0, "pf_fair_value_at_entry": 20.0, "entry_price": 100.0, "exit_price": 99.0, "confidence": 0.60},
            ]
        )

        with_distance = compute_pf_distance(interval_frame)
        signals = generate_signal_from_pf_distance(with_distance, config=BacktestConfig())

        self.assertEqual(list(signals["trade_side"]), ["NO_TRADE", "UP", "DOWN"])

    def test_hour_filter_does_not_block_trades_anymore(self) -> None:
        frame = pd.DataFrame(
            [
                {"contract_start": pd.Timestamp("2024-01-01 07:00:00+00:00"), "trade_side": "UP"},
                {"contract_start": pd.Timestamp("2024-01-01 09:00:00+00:00"), "trade_side": "UP"},
            ]
        )
        filtered = apply_hour_filter(frame, config=BacktestConfig())
        self.assertEqual(list(filtered["blocked_hour_flag"]), [False, False])

    def test_three_loss_pause_skips_one_interval_after_third_loss(self) -> None:
        frame = pd.DataFrame(
            [
                {"contract_start": pd.Timestamp("2024-01-01 00:00:00+00:00"), "contract_end": pd.Timestamp("2024-01-01 00:15:00+00:00"), "interval_start_price": 100.0, "pf_fair_value_at_entry": 450.0, "pf_distance": 350.0, "entry_price": 100.0, "exit_price": 99.0, "confidence": 0.2, "trade_side": "UP", "blocked_hour_flag": False},
                {"contract_start": pd.Timestamp("2024-01-01 00:15:00+00:00"), "contract_end": pd.Timestamp("2024-01-01 00:30:00+00:00"), "interval_start_price": 100.0, "pf_fair_value_at_entry": 450.0, "pf_distance": 350.0, "entry_price": 100.0, "exit_price": 99.0, "confidence": 0.2, "trade_side": "UP", "blocked_hour_flag": False},
                {"contract_start": pd.Timestamp("2024-01-01 00:30:00+00:00"), "contract_end": pd.Timestamp("2024-01-01 00:45:00+00:00"), "interval_start_price": 100.0, "pf_fair_value_at_entry": 450.0, "pf_distance": 350.0, "entry_price": 100.0, "exit_price": 99.0, "confidence": 0.2, "trade_side": "UP", "blocked_hour_flag": False},
                {"contract_start": pd.Timestamp("2024-01-01 00:45:00+00:00"), "contract_end": pd.Timestamp("2024-01-01 01:00:00+00:00"), "interval_start_price": 100.0, "pf_fair_value_at_entry": 450.0, "pf_distance": 350.0, "entry_price": 100.0, "exit_price": 101.0, "confidence": 0.2, "trade_side": "UP", "blocked_hour_flag": False},
                {"contract_start": pd.Timestamp("2024-01-01 01:00:00+00:00"), "contract_end": pd.Timestamp("2024-01-01 01:15:00+00:00"), "interval_start_price": 100.0, "pf_fair_value_at_entry": 450.0, "pf_distance": 350.0, "entry_price": 100.0, "exit_price": 101.0, "confidence": 0.2, "trade_side": "UP", "blocked_hour_flag": False},
            ]
        )

        paused = apply_three_loss_pause(frame, config=BacktestConfig())
        self.assertEqual(list(paused["skip_reason"]), ["TRADE_EXECUTED", "TRADE_EXECUTED", "TRADE_EXECUTED", "COOLDOWN", "TRADE_EXECUTED"])

    def test_evaluate_trade_pnl_uses_binary_payout_and_kalshi_fee(self) -> None:
        trade_rows = pd.DataFrame(
            [
                {
                    "contract_start": pd.Timestamp("2024-01-01 00:00:00+00:00"),
                    "contract_end": pd.Timestamp("2024-01-01 00:15:00+00:00"),
                    "interval_start_price": 100.0,
                    "pf_fair_value_at_entry": 450.0,
                    "pf_distance": 350.0,
                    "entry_price": 100.0,
                    "exit_price": 101.0,
                    "confidence": 0.3,
                    "binary_win": 1.0,
                    "trade_side": "UP",
                    "skip_reason": "TRADE_EXECUTED",
                    "_trade_taken": 1,
                },
                {
                    "contract_start": pd.Timestamp("2024-01-01 00:15:00+00:00"),
                    "contract_end": pd.Timestamp("2024-01-01 00:30:00+00:00"),
                    "interval_start_price": 100.0,
                    "pf_fair_value_at_entry": -250.0,
                    "pf_distance": -350.0,
                    "entry_price": 100.0,
                    "exit_price": 101.0,
                    "confidence": 0.3,
                    "binary_win": 0.0,
                    "trade_side": "DOWN",
                    "skip_reason": "TRADE_EXECUTED",
                    "_trade_taken": 1,
                },
            ]
        )

        results = evaluate_trade_pnl(
            trade_rows,
            config=BacktestConfig(market_up_price=0.55, market_down_price=0.45),
        )

        self.assertEqual(results["binary_win"].dtype.name, "Int64")
        self.assertEqual(list(results["binary_win"]), [1, 0])
        self.assertAlmostEqual(results.loc[0, "gross_pnl_per_trade"], 0.45, places=6)
        self.assertAlmostEqual(results.loc[0, "fee"], 0.02, places=6)
        self.assertAlmostEqual(results.loc[0, "net_pnl_per_trade"], 0.43, places=6)
        self.assertAlmostEqual(results.loc[1, "gross_pnl_per_trade"], -0.45, places=6)
        self.assertAlmostEqual(results.loc[1, "net_pnl_per_trade"], -0.47, places=6)

    def test_kalshi_taker_fee_rounds_up_to_nearest_cent(self) -> None:
        self.assertAlmostEqual(kalshi_taker_fee(1.0, 0.50), 0.02, places=6)
        self.assertAlmostEqual(kalshi_taker_fee(10.0, 0.20), 0.12, places=6)

    def test_run_backtest_returns_only_requested_columns_and_summaries(self) -> None:
        index = pd.to_datetime(
            [
                "2024-01-01 05:45:00+00:00",
                "2024-01-01 06:00:00+00:00",
                "2024-01-01 06:15:00+00:00",
                "2024-01-01 06:30:00+00:00",
                "2024-01-01 06:45:00+00:00",
                "2024-01-01 07:00:00+00:00",
            ]
        )
        candles = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
                "high": [101.0, 101.0, 101.0, 101.0, 101.0, 101.0],
                "low": [99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
                "close": [100.0, 101.0, 99.0, 99.0, 101.0, 101.0],
                "volume": [1.0] * 6,
            },
            index=index,
        )

        results = run_backtest(
            candles,
            config=BacktestConfig(
                particle_filter_use_regime_context=False,
                particle_filter_lookback=6,
            ),
        )

        self.assertEqual(list(results.columns), [
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
        ])
        self.assertTrue("actual_gap" in results.columns)
        self.assertFalse(summarize_backtest(results).empty)
        skip_summary = summarize_skip_reasons(results)
        self.assertFalse(skip_summary.empty)
        self.assertEqual(list(skip_summary.columns), ["cooldown_skips", "no_signal_rows"])


if __name__ == "__main__":
    unittest.main()
