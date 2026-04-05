from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from storage.session_store import SessionStore


@dataclass(frozen=True)
class PortfolioMetrics:
    starting_bankroll: float
    current_bankroll: float
    available_cash: float
    realized_pnl: float
    unrealized_pnl: float
    cumulative_return: float
    win_rate: float
    drawdown: float
    peak_bankroll: float
    trade_count: int
    open_trade_count: int
    wins: int
    losses: int


class PnlEngine:
    def __init__(self, store: SessionStore) -> None:
        self.store = store

    def calculate(self, marks_by_market: dict[str, dict[str, float]] | None = None) -> PortfolioMetrics:
        trades = self.store.get_trades_frame()
        starting_bankroll = float(self.store.get_state("starting_bankroll", 100.0))
        if trades.empty:
            return PortfolioMetrics(
                starting_bankroll=starting_bankroll,
                current_bankroll=starting_bankroll,
                available_cash=starting_bankroll,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                cumulative_return=0.0,
                win_rate=0.0,
                drawdown=0.0,
                peak_bankroll=max(starting_bankroll, self._history_peak_or_default(starting_bankroll)),
                trade_count=0,
                open_trade_count=0,
                wins=0,
                losses=0,
            )

        trades["realized_pnl"] = pd.to_numeric(trades["realized_pnl"], errors="coerce").fillna(0.0)
        trades["stake_usd"] = pd.to_numeric(trades["stake_usd"], errors="coerce").fillna(0.0)
        trades["fill_price"] = pd.to_numeric(trades["fill_price"], errors="coerce").fillna(0.0)
        trades["contracts"] = pd.to_numeric(trades["contracts"], errors="coerce").fillna(0.0)
        trades["fee_paid"] = pd.to_numeric(trades["fee_paid"], errors="coerce").fillna(0.0)

        settled = trades[trades["status"] == "SETTLED"].copy()
        open_trades = trades[trades["status"] == "OPEN"].copy()

        realized_pnl = float(settled["realized_pnl"].sum()) if not settled.empty else 0.0
        open_cost = float(open_trades["stake_usd"].sum()) if not open_trades.empty else 0.0

        unrealized_pnl = 0.0
        for row in open_trades.to_dict(orient="records"):
            mark = self._mark_price(row, marks_by_market or {})
            cost = float(row["stake_usd"])
            current_value = float(row["contracts"]) * mark
            unrealized_pnl += current_value - cost

        available_cash = starting_bankroll + realized_pnl - open_cost
        current_bankroll = available_cash + open_cost + unrealized_pnl
        cumulative_return = (current_bankroll / starting_bankroll) - 1.0 if starting_bankroll > 0 else 0.0

        wins = int((settled["realized_pnl"] > 0).sum()) if not settled.empty else 0
        losses = int((settled["realized_pnl"] < 0).sum()) if not settled.empty else 0
        settled_count = len(settled)
        win_rate = wins / settled_count if settled_count > 0 else 0.0

        peak_bankroll = max(current_bankroll, self._history_peak_or_default(starting_bankroll))
        drawdown = 0.0 if peak_bankroll <= 0 else max(0.0, 1.0 - (current_bankroll / peak_bankroll))

        return PortfolioMetrics(
            starting_bankroll=starting_bankroll,
            current_bankroll=current_bankroll,
            available_cash=available_cash,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            cumulative_return=cumulative_return,
            win_rate=win_rate,
            drawdown=drawdown,
            peak_bankroll=peak_bankroll,
            trade_count=int(len(trades)),
            open_trade_count=int(len(open_trades)),
            wins=wins,
            losses=losses,
        )

    def persist_snapshot(self, metrics: PortfolioMetrics) -> None:
        self.store.record_bankroll_snapshot(asdict(metrics))

    def _history_peak_or_default(self, default: float) -> float:
        latest = self.store.get_recent_bankroll_history(limit=1000)
        if latest.empty:
            return default
        return float(max(default, latest["peak_bankroll"].max(), latest["current_bankroll"].max()))

    @staticmethod
    def _mark_price(trade: dict[str, Any], marks_by_market: dict[str, dict[str, float]]) -> float:
        market_marks = marks_by_market.get(str(trade["market_ticker"]), {})
        side = str(trade["kalshi_side"]).lower()
        if side == "yes":
            return float(market_marks.get("yes_bid", trade["fill_price"]))
        return float(market_marks.get("no_bid", trade["fill_price"]))
