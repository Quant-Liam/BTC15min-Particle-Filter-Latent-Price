from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import pandas as pd

from analytics.pnl_engine import PnlEngine
from config.settings import AppSettings
from data.kalshi_market_client import KalshiMarketSnapshot
from engine.trading_loop import StrategyCycleSnapshot, TradingLoop
from storage.session_store import SessionStore


def make_settings(db_path: Path) -> AppSettings:
    return AppSettings(
        mode="TRAINING",
        kalshi_base_url="https://demo-api.kalshi.co/trade-api/v2",
        kalshi_api_key_id="demo-key",
        kalshi_private_key_path=db_path.parent / "demo.pem",
        kalshi_series_prefix="KXBTC15M",
        state_db_path=db_path,
        auto_start_trading=True,
        loop_poll_seconds=5,
        cycle_min_seconds_between_runs=60,
        candle_limit_15m=320,
        candle_limit_1m=720,
        starting_bankroll=100.0,
        allocation_fraction=0.20,
        stop_bankroll_threshold=20.0,
        regime_confidence_threshold=0.45,
        min_pf_confidence=0.50,
        min_pf_gap_dollars=25.0,
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


def make_market() -> KalshiMarketSnapshot:
    return KalshiMarketSnapshot(
        ticker="KXBTC15M-TEST",
        title="Bitcoin 15m Test",
        subtitle="Training",
        status="open",
        event_ticker="KXBTC15M",
        close_time=pd.Timestamp("2026-04-06T10:45:00+00:00"),
        expected_expiration_time=pd.Timestamp("2026-04-06T10:45:00+00:00"),
        result=None,
        yes_bid_dollars=0.48,
        yes_ask_dollars=0.52,
        no_bid_dollars=0.48,
        no_ask_dollars=0.52,
        yes_bid_size=100.0,
        yes_ask_size=100.0,
        no_bid_size=100.0,
        liquidity_dollars=500.0,
        volume=1000.0,
        open_interest=250.0,
        last_price_dollars=0.50,
        fractional_trading_enabled=True,
        rules_primary="Test rules",
        yes_mid_dollars=0.50,
        no_mid_dollars=0.50,
    )


class FakeReferenceFeed:
    pass


class FakeMarketClient:
    def __init__(self, market: KalshiMarketSnapshot) -> None:
        self.market = market

    def discover_active_btc_market(self, series_prefix: str = "KXBTC15M") -> KalshiMarketSnapshot | None:
        return self.market


class FakeExecutor:
    def sync_settlements(self) -> list[dict[str, object]]:
        return []

    def mark_prices_for_open_positions(self) -> dict[str, dict[str, float]]:
        return {}

    def execute_trade(self, **_: object) -> None:
        raise AssertionError("execute_trade should not be called in this test")


class TradingLoopTest(unittest.TestCase):
    def test_strategy_is_analyzed_each_poll_but_only_evaluated_once_per_15m_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "session.sqlite3"
            settings = make_settings(db_path)
            store = SessionStore(db_path)
            store.seed_defaults(
                starting_bankroll=settings.starting_bankroll,
                allocation_fraction=settings.allocation_fraction,
                trading_enabled=True,
            )

            loop = TradingLoop(
                settings=settings,
                store=store,
                reference_feed=FakeReferenceFeed(),
                market_client=FakeMarketClient(make_market()),
                executor=FakeExecutor(),
                pnl_engine=PnlEngine(store),
            )

            first_cycle = StrategyCycleSnapshot(
                market_ticker="KXBTC15M-TEST",
                window_start_utc="2026-04-06T10:15:00+00:00",
                window_end_utc="2026-04-06T10:30:00+00:00",
                reference_source="FakeReferenceFeed",
                btc_reference_price=70000.0,
                pf_fair_value=70010.0,
                pf_gap=-10.0,
                pf_confidence=0.60,
                regime="bull",
                allowed_side="UP",
                regime_confidence=0.75,
                p_win=0.56,
                kelly_fraction=0.10,
                raw_kelly=0.20,
                expected_log_growth=0.01,
                decision_side="NO_TRADE",
                decision_reason="No trade for this interval.",
                model_probability=0.56,
                market_yes_ask=0.52,
                market_no_ask=0.48,
            )
            second_cycle = StrategyCycleSnapshot(
                market_ticker="KXBTC15M-TEST",
                window_start_utc="2026-04-06T10:30:00+00:00",
                window_end_utc="2026-04-06T10:45:00+00:00",
                reference_source="FakeReferenceFeed",
                btc_reference_price=70020.0,
                pf_fair_value=70030.0,
                pf_gap=-10.0,
                pf_confidence=0.60,
                regime="bull",
                allowed_side="UP",
                regime_confidence=0.80,
                p_win=0.57,
                kelly_fraction=0.10,
                raw_kelly=0.20,
                expected_log_growth=0.01,
                decision_side="NO_TRADE",
                decision_reason="No trade for this interval.",
                model_probability=0.57,
                market_yes_ask=0.51,
                market_no_ask=0.49,
            )
            mid_interval_cycle = StrategyCycleSnapshot(
                market_ticker="KXBTC15M-TEST",
                window_start_utc="2026-04-06T10:00:00+00:00",
                window_end_utc="2026-04-06T10:15:00+00:00",
                reference_source="FakeReferenceFeed",
                btc_reference_price=69990.0,
                pf_fair_value=70005.0,
                pf_gap=-15.0,
                pf_confidence=0.60,
                regime="bull",
                allowed_side="UP",
                regime_confidence=0.72,
                p_win=0.55,
                kelly_fraction=0.08,
                raw_kelly=0.16,
                expected_log_growth=0.01,
                decision_side="UP",
                decision_reason="Bull regime allows only UP trades.",
                model_probability=0.55,
                market_yes_ask=0.51,
                market_no_ask=0.49,
            )
            repeat_poll_cycle = StrategyCycleSnapshot(
                market_ticker="KXBTC15M-TEST",
                window_start_utc="2026-04-06T10:15:00+00:00",
                window_end_utc="2026-04-06T10:30:00+00:00",
                reference_source="FakeReferenceFeed",
                btc_reference_price=70005.0,
                pf_fair_value=70015.0,
                pf_gap=-10.0,
                pf_confidence=0.60,
                regime="bull",
                allowed_side="UP",
                regime_confidence=0.78,
                p_win=0.565,
                kelly_fraction=0.10,
                raw_kelly=0.20,
                expected_log_growth=0.01,
                decision_side="UP",
                decision_reason="Bull regime allows only UP trades.",
                model_probability=0.565,
                market_yes_ask=0.52,
                market_no_ask=0.48,
            )
            loop._build_strategy_snapshot = Mock(side_effect=[mid_interval_cycle, first_cycle, repeat_poll_cycle, second_cycle])

            loop._tick(now_utc=pd.Timestamp("2026-04-06T10:07:00+00:00"))
            self.assertEqual(loop._build_strategy_snapshot.call_count, 1)
            self.assertEqual(len(store.get_recent_signals(limit=10)), 0)
            self.assertEqual(store.get_state("last_analysis_decision_side"), "UP")
            self.assertEqual(store.get_state("last_analysis_window_start_utc"), "2026-04-06T10:00:00+00:00")

            loop._tick(now_utc=pd.Timestamp("2026-04-06T10:15:02+00:00"))
            self.assertEqual(loop._build_strategy_snapshot.call_count, 2)
            self.assertEqual(len(store.get_recent_signals(limit=10)), 1)

            loop._tick(now_utc=pd.Timestamp("2026-04-06T10:15:08+00:00"))
            self.assertEqual(loop._build_strategy_snapshot.call_count, 3)
            self.assertEqual(len(store.get_recent_signals(limit=10)), 1)
            self.assertEqual(store.get_state("last_analysis_window_start_utc"), "2026-04-06T10:15:00+00:00")

            loop._tick(now_utc=pd.Timestamp("2026-04-06T10:30:03+00:00"))
            self.assertEqual(loop._build_strategy_snapshot.call_count, 4)
            self.assertEqual(len(store.get_recent_signals(limit=10)), 2)


if __name__ == "__main__":
    unittest.main()
