from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any
import uuid

import pandas as pd

from config.settings import AppSettings
from data.kalshi_market_client import KalshiDemoClient, KalshiMarketSnapshot
from storage.session_store import SessionStore


class TrainingExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingExecutionResult:
    trade_id: int
    client_order_id: str
    kalshi_order_id: str | None
    market_ticker: str
    strategy_side: str
    kalshi_side: str
    contracts: float
    fill_price: float
    fee_paid: float
    stake_usd: float
    status: str


class TrainingExecutor:
    def __init__(self, settings: AppSettings, store: SessionStore, client: KalshiDemoClient) -> None:
        if "demo-api.kalshi.co" not in client.base_url:
            raise RuntimeError("TrainingExecutor refuses to route anywhere except Kalshi DEMO.")
        self.settings = settings
        self.store = store
        self.client = client

    def execute_trade(
        self,
        *,
        market: KalshiMarketSnapshot,
        strategy_side: str,
        allocation_fraction: float,
        bankroll: float,
        signal_id: int,
        signal_payload: dict[str, Any],
    ) -> TrainingExecutionResult:
        self.settings.validate_trading_credentials()

        kalshi_side = self._kalshi_side(strategy_side)
        ask_price = market.normalized_yes_ask_dollars if kalshi_side == "yes" else market.normalized_no_ask_dollars
        if not 0 < ask_price < 1:
            raise TrainingExecutionError(f"Invalid {kalshi_side} ask price for {market.ticker}: {ask_price}")

        target_stake = bankroll * float(allocation_fraction)
        if target_stake < self.settings.min_trade_notional:
            raise TrainingExecutionError(
                f"Allocation ${target_stake:,.2f} is below the minimum trade notional "
                f"${self.settings.min_trade_notional:,.2f}."
            )

        contracts = self._contracts_for_stake(
            stake_usd=target_stake,
            ask_price=ask_price,
            fractional_trading_enabled=market.fractional_trading_enabled,
        )
        if contracts <= 0:
            raise TrainingExecutionError("Calculated contract size was zero after Kalshi lot-size rules.")

        client_order_id = f"training-{market.ticker.lower()}-{uuid.uuid4().hex[:12]}"
        response = self.client.create_buy_order(
            ticker=market.ticker,
            side=kalshi_side,
            contracts=Decimal(str(contracts)),
            price_dollars=Decimal(str(ask_price)),
            client_order_id=client_order_id,
        )

        order_payload = response.get("order", response)
        kalshi_order_id = order_payload.get("order_id")
        order_status = str(order_payload.get("status", "unknown"))
        self.store.upsert_order(
            {
                "client_order_id": client_order_id,
                "kalshi_order_id": kalshi_order_id,
                "market_ticker": market.ticker,
                "side": kalshi_side,
                "order_action": "buy",
                "contracts": contracts,
                "requested_price": ask_price,
                "order_status": order_status,
                "response_payload": order_payload,
            }
        )

        if order_status.lower() not in {"filled", "executed"}:
            self.store.set_state("last_order_status", f"{market.ticker}: {order_status}")
            raise TrainingExecutionError(f"Kalshi demo order was not filled immediately: status={order_status}")

        fill_summary = self._extract_fill_summary(
            order_payload=order_payload,
            market=market,
            client_order_id=client_order_id,
            kalshi_order_id=kalshi_order_id,
            kalshi_side=kalshi_side,
        )
        stake_usd = fill_summary["contracts"] * fill_summary["fill_price"] + fill_summary["fee_paid"]

        trade_id = self.store.create_trade(
            {
                "signal_id": signal_id,
                "client_order_id": client_order_id,
                "kalshi_order_id": kalshi_order_id,
                "market_ticker": market.ticker,
                "event_ticker": market.event_ticker,
                "title": market.title,
                "strategy_side": strategy_side,
                "kalshi_side": kalshi_side,
                "contracts": fill_summary["contracts"],
                "fill_price": fill_summary["fill_price"],
                "fee_paid": fill_summary["fee_paid"],
                "stake_usd": stake_usd,
                "allocation_fraction": allocation_fraction,
                "btc_reference_price": signal_payload.get("btc_reference_price"),
                "pf_fair_value": signal_payload.get("pf_fair_value"),
                "pf_gap": signal_payload.get("pf_gap"),
                "regime": signal_payload.get("regime"),
                "regime_confidence": signal_payload.get("regime_confidence"),
                "p_win": signal_payload.get("p_win"),
                "kelly_fraction": signal_payload.get("kelly_fraction"),
                "raw_kelly": signal_payload.get("raw_kelly"),
                "expected_log_growth": signal_payload.get("expected_log_growth"),
                "status": "OPEN",
                "close_time_utc": market.close_time.isoformat() if market.close_time is not None else None,
                "notes": "TRAINING mode via Kalshi DEMO",
            }
        )
        self.store.set_state(
            "last_trade_summary",
            f"{strategy_side} {market.ticker} @ {fill_summary['fill_price']:.4f} x {fill_summary['contracts']:.2f}",
        )
        self.store.set_state("last_order_status", f"{market.ticker}: FILLED")

        return TrainingExecutionResult(
            trade_id=trade_id,
            client_order_id=client_order_id,
            kalshi_order_id=kalshi_order_id,
            market_ticker=market.ticker,
            strategy_side=strategy_side,
            kalshi_side=kalshi_side,
            contracts=fill_summary["contracts"],
            fill_price=fill_summary["fill_price"],
            fee_paid=fill_summary["fee_paid"],
            stake_usd=stake_usd,
            status="OPEN",
        )

    def sync_settlements(self) -> list[dict[str, Any]]:
        settled_rows: list[dict[str, Any]] = []
        for trade in self.store.get_open_trades():
            market = self.client.get_market(str(trade["market_ticker"]))
            if market.result not in {"yes", "no"}:
                continue
            settlement_value = self._settlement_value(kalshi_side=str(trade["kalshi_side"]), result=market.result)
            gross_value = float(trade["contracts"]) * settlement_value
            realized_pnl = gross_value - float(trade["stake_usd"])
            self.store.settle_trade(
                trade_id=int(trade["id"]),
                settlement_result=market.result,
                settlement_value=settlement_value,
                realized_pnl=realized_pnl,
            )
            settled = {
                "trade_id": int(trade["id"]),
                "market_ticker": str(trade["market_ticker"]),
                "settlement_result": market.result,
                "realized_pnl": realized_pnl,
            }
            settled_rows.append(settled)
            self.store.set_state("last_trade_summary", f"Settled {trade['market_ticker']} {market.result} pnl={realized_pnl:.2f}")
        return settled_rows

    def mark_prices_for_open_positions(self) -> dict[str, dict[str, float]]:
        marks: dict[str, dict[str, float]] = {}
        for trade in self.store.get_open_trades():
            ticker = str(trade["market_ticker"])
            if ticker in marks:
                continue
            market = self.client.get_market(ticker)
            marks[ticker] = {
                "yes_bid": market.yes_bid_dollars,
                "no_bid": market.no_bid_dollars,
            }
        return marks

    def _extract_fill_summary(
        self,
        *,
        order_payload: dict[str, Any],
        market: KalshiMarketSnapshot,
        client_order_id: str,
        kalshi_order_id: str | None,
        kalshi_side: str,
    ) -> dict[str, float]:
        contracts = float(order_payload.get("fill_count_fp") or order_payload.get("initial_count_fp") or 0.0)
        if contracts <= 0:
            contracts = float(order_payload.get("fill_count") or order_payload.get("initial_count") or 0.0)
        fill_price = (
            float(order_payload.get("yes_price_dollars"))
            if kalshi_side == "yes"
            else float(order_payload.get("no_price_dollars"))
        )
        fee_paid = float(order_payload.get("taker_fees_dollars") or 0.0)

        fills_payload = self.client.get_fills(order_id=kalshi_order_id, limit=50)
        fills = fills_payload.get("fills", [])
        if fills:
            total_contracts = 0.0
            weighted_price = 0.0
            total_fees = 0.0
            for fill in fills:
                fill_contracts = float(fill.get("count_fp") or fill.get("count") or 0.0)
                if fill_contracts <= 0:
                    continue
                price_key = "yes_price_dollars" if kalshi_side == "yes" else "no_price_dollars"
                this_price = float(fill.get(price_key) or fill_price)
                weighted_price += fill_contracts * this_price
                total_contracts += fill_contracts
                total_fees += float(fill.get("fee_paid_dollars") or fill.get("fee_paid") or 0.0)
                self.store.record_fill(
                    {
                        "fill_id": fill.get("fill_id"),
                        "client_order_id": client_order_id,
                        "kalshi_order_id": kalshi_order_id,
                        "market_ticker": market.ticker,
                        "side": kalshi_side,
                        "contracts": fill_contracts,
                        "fill_price": this_price,
                        "fee_paid": float(fill.get("fee_paid_dollars") or fill.get("fee_paid") or 0.0),
                        "fill_payload": fill,
                    }
                )
            if total_contracts > 0:
                contracts = total_contracts
                fill_price = weighted_price / total_contracts
                fee_paid = total_fees

        return {
            "contracts": contracts,
            "fill_price": fill_price,
            "fee_paid": fee_paid,
        }

    @staticmethod
    def _kalshi_side(strategy_side: str) -> str:
        side = strategy_side.upper()
        if side == "UP":
            return "yes"
        if side == "DOWN":
            return "no"
        raise TrainingExecutionError(f"Cannot route strategy side {strategy_side!r} to Kalshi.")

    @staticmethod
    def _settlement_value(*, kalshi_side: str, result: str) -> float:
        return 1.0 if kalshi_side.lower() == result.lower() else 0.0

    @staticmethod
    def _contracts_for_stake(*, stake_usd: float, ask_price: float, fractional_trading_enabled: bool) -> float:
        raw_contracts = Decimal(str(stake_usd)) / Decimal(str(ask_price))
        if fractional_trading_enabled:
            contracts = raw_contracts.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        else:
            contracts = raw_contracts.quantize(Decimal("1"), rounding=ROUND_DOWN)
        return float(contracts)
