from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from config.settings import AppSettings


class TrainingSettingsTest(unittest.TestCase):
    def test_live_mode_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            AppSettings(
                mode="LIVE",
                kalshi_base_url="https://demo-api.kalshi.co/trade-api/v2",
                kalshi_api_key_id=None,
                kalshi_private_key_path=None,
                kalshi_series_prefix="KXBTC15M",
                state_db_path=Path("storage/test.sqlite3"),
                auto_start_trading=False,
                loop_poll_seconds=5,
                cycle_min_seconds_between_runs=60,
                candle_limit_15m=320,
                candle_limit_1m=720,
                starting_bankroll=100.0,
                allocation_fraction=0.20,
                stop_bankroll_threshold=20.0,
                regime_confidence_threshold=0.45,
                pf_alpha=1.5,
                pf_min_gap_scale=0.001,
                fee_rate=0.0156,
                fractional_kelly=0.5,
                max_fraction=0.2,
                use_confidence_shrink=True,
                edge_buffer=0.02,
                max_spread_dollars=0.12,
                min_liquidity_dollars=100.0,
                min_depth_contracts=10.0,
                min_seconds_to_expiry=120,
                min_trade_notional=1.0,
                coinbase_base_url="https://api.exchange.coinbase.com",
                coinbase_product_id="BTC-USD",
            )

    def test_non_demo_kalshi_base_url_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            AppSettings(
                mode="TRAINING",
                kalshi_base_url="https://api.elections.kalshi.com/trade-api/v2",
                kalshi_api_key_id=None,
                kalshi_private_key_path=None,
                kalshi_series_prefix="KXBTC15M",
                state_db_path=Path("storage/test.sqlite3"),
                auto_start_trading=False,
                loop_poll_seconds=5,
                cycle_min_seconds_between_runs=60,
                candle_limit_15m=320,
                candle_limit_1m=720,
                starting_bankroll=100.0,
                allocation_fraction=0.20,
                stop_bankroll_threshold=20.0,
                regime_confidence_threshold=0.45,
                pf_alpha=1.5,
                pf_min_gap_scale=0.001,
                fee_rate=0.0156,
                fractional_kelly=0.5,
                max_fraction=0.2,
                use_confidence_shrink=True,
                edge_buffer=0.02,
                max_spread_dollars=0.12,
                min_liquidity_dollars=100.0,
                min_depth_contracts=10.0,
                min_seconds_to_expiry=120,
                min_trade_notional=1.0,
                coinbase_base_url="https://api.exchange.coinbase.com",
                coinbase_product_id="BTC-USD",
            )
