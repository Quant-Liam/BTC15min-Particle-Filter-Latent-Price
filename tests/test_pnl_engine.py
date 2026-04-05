from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from analytics.pnl_engine import PnlEngine
from storage.session_store import SessionStore


class PnlEngineTest(unittest.TestCase):
    def test_open_trade_marks_unrealized_and_settled_trade_counts_realized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SessionStore(Path(tmpdir) / "session.sqlite3")
            store.seed_defaults(starting_bankroll=100.0, allocation_fraction=0.20, trading_enabled=False)
            store.create_trade(
                {
                    "market_ticker": "KXBTC15M-OPEN",
                    "strategy_side": "UP",
                    "kalshi_side": "yes",
                    "contracts": 10.0,
                    "fill_price": 0.40,
                    "fee_paid": 0.10,
                    "stake_usd": 4.10,
                    "allocation_fraction": 0.20,
                    "status": "OPEN",
                }
            )
            settled_id = store.create_trade(
                {
                    "market_ticker": "KXBTC15M-SETTLED",
                    "strategy_side": "DOWN",
                    "kalshi_side": "no",
                    "contracts": 10.0,
                    "fill_price": 0.30,
                    "fee_paid": 0.10,
                    "stake_usd": 3.10,
                    "allocation_fraction": 0.20,
                    "status": "OPEN",
                }
            )
            store.settle_trade(
                trade_id=settled_id,
                settlement_result="no",
                settlement_value=1.0,
                realized_pnl=6.90,
            )

            metrics = PnlEngine(store).calculate(
                marks_by_market={
                    "KXBTC15M-OPEN": {"yes_bid": 0.50, "no_bid": 0.0},
                }
            )

            self.assertAlmostEqual(metrics.realized_pnl, 6.90, places=6)
            self.assertAlmostEqual(metrics.unrealized_pnl, 0.90, places=6)
            self.assertAlmostEqual(metrics.current_bankroll, 107.80, places=6)
            self.assertEqual(metrics.trade_count, 2)
            self.assertEqual(metrics.open_trade_count, 1)
            self.assertEqual(metrics.wins, 1)
