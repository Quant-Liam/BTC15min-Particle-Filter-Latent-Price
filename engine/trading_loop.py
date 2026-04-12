from __future__ import annotations

from dataclasses import asdict, dataclass
import threading
import time
from typing import Any

import pandas as pd

from analytics.pnl_engine import PnlEngine, PortfolioMetrics
from btc15m import (
    ParticleFilterConfig,
    compute_kelly_from_pf,
    compute_particle_filter_frame,
    compute_regime_frame,
    current_15m_window,
    project_particle_filter_to_time,
    snapshot_from_particle_filter_frame,
    snapshot_from_regime_frame,
)
from btc15m.strategy import apply_particle_filter_entry_filter, decide_trade_side
from config.settings import AppSettings, RuntimeOverrides
from data.kalshi_market_client import KalshiDemoClient, KalshiMarketSnapshot
from data.reference_price_feed import ReferencePriceFeed
from execution.training_executor import TrainingExecutionError, TrainingExecutor
from storage.session_store import SessionStore


@dataclass(frozen=True)
class StrategyCycleSnapshot:
    market_ticker: str | None
    window_start_utc: str
    window_end_utc: str
    reference_source: str
    btc_reference_price: float
    pf_fair_value: float
    pf_gap: float
    pf_confidence: float
    regime: str
    allowed_side: str
    regime_confidence: float
    p_win: float | None
    kelly_fraction: float
    raw_kelly: float
    expected_log_growth: float
    decision_side: str
    decision_reason: str
    model_probability: float | None
    market_yes_ask: float
    market_no_ask: float


@dataclass(frozen=True)
class EntryWindowGate:
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    should_evaluate: bool
    reason: str


class TradingLoop:
    def __init__(
        self,
        *,
        settings: AppSettings,
        store: SessionStore,
        reference_feed: ReferencePriceFeed,
        market_client: KalshiDemoClient,
        executor: TrainingExecutor,
        pnl_engine: PnlEngine,
    ) -> None:
        self.settings = settings
        self.store = store
        self.reference_feed = reference_feed
        self.market_client = market_client
        self.executor = executor
        self.pnl_engine = pnl_engine

    def run_forever(self, stop_event: threading.Event) -> None:
        self.store.seed_defaults(
            starting_bankroll=self.settings.starting_bankroll,
            allocation_fraction=self.settings.allocation_fraction,
            trading_enabled=self.settings.auto_start_trading,
        )
        self.store.set_state("stop_bankroll_threshold", self.settings.stop_bankroll_threshold)

        while not stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:  # pragma: no cover - runtime safety net
                self.store.set_state("last_error", str(exc))
                self.store.set_bot_status("ERROR", str(exc))
                self.store.add_log("ERROR", "trading_loop", str(exc))
            stop_event.wait(self.settings.loop_poll_seconds)

    def _tick(self, now_utc: pd.Timestamp | None = None) -> None:
        now_utc = self._normalize_utc_timestamp(now_utc)
        self.store.set_state("last_heartbeat_utc", now_utc.isoformat())
        settled = self.executor.sync_settlements()
        for row in settled:
            self.store.add_log("INFO", "executor", f"Settled {row['market_ticker']} with P&L {row['realized_pnl']:.2f}", row)

        marks = self.executor.mark_prices_for_open_positions()
        metrics = self.pnl_engine.calculate(marks_by_market=marks)
        self.pnl_engine.persist_snapshot(metrics)

        stop_threshold = float(self.store.get_state("stop_bankroll_threshold", self.settings.stop_bankroll_threshold))
        if metrics.current_bankroll <= stop_threshold:
            self.store.set_state("trading_enabled", False)
            self.store.set_bot_status("STOPPED", f"Bankroll fell to ${metrics.current_bankroll:,.2f}, below stop threshold.")
            return

        trading_enabled = bool(self.store.get_state("trading_enabled", self.settings.auto_start_trading))
        if not trading_enabled:
            self.store.set_bot_status("RUNNING", "Loop is alive. Trading is paused.")
            return

        market = self.market_client.discover_active_btc_market(series_prefix=self.settings.kalshi_series_prefix)
        if market is None:
            self.store.set_bot_status("RUNNING", "No active BTC 15-minute Kalshi demo market found.")
            return

        cycle = self._build_strategy_snapshot(market, now_utc=now_utc)
        self._persist_latest_analysis(cycle=cycle, analyzed_at_utc=now_utc)

        entry_gate = self._entry_window_gate(now_utc=now_utc)
        if not entry_gate.should_evaluate:
            self.store.set_bot_status(
                "RUNNING",
                f"{cycle.decision_reason} Monitoring until the next 15-minute entry window at {entry_gate.window_end.isoformat()}.",
            )
            return

        signal_id = self.store.record_signal(asdict(cycle))
        self._mark_window_evaluated(entry_gate.window_start)
        self.store.set_state("last_cycle_market", market.ticker)
        self.store.set_state("last_cycle_completed_utc", now_utc.isoformat())

        if cycle.decision_side == "NO_TRADE":
            self.store.set_bot_status("RUNNING", cycle.decision_reason)
            self.store.add_log("INFO", "strategy", cycle.decision_reason, asdict(cycle))
            return

        risk_failures = self._risk_failures(market=market, cycle=cycle, metrics=metrics)
        if risk_failures:
            reason = "; ".join(risk_failures)
            self.store.set_bot_status("RUNNING", f"Skipped {cycle.decision_side}: {reason}")
            self.store.add_log("WARNING", "risk", reason, {"market_ticker": market.ticker, "side": cycle.decision_side})
            return

        allocation_fraction = float(self.store.get_state("allocation_fraction", self.settings.allocation_fraction))
        try:
            execution = self.executor.execute_trade(
                market=market,
                strategy_side=cycle.decision_side,
                allocation_fraction=allocation_fraction,
                bankroll=metrics.current_bankroll,
                signal_id=signal_id,
                signal_payload=asdict(cycle),
            )
        except TrainingExecutionError as exc:
            self.store.set_bot_status("ERROR", str(exc))
            self.store.add_log("ERROR", "executor", str(exc), {"market_ticker": market.ticker, "side": cycle.decision_side})
            return

        self.store.set_bot_status(
            "RUNNING",
            f"Executed {execution.strategy_side} on {execution.market_ticker} using {allocation_fraction * 100:.1f}% training allocation.",
        )
        self.store.add_log(
            "INFO",
            "executor",
            f"Filled {execution.strategy_side} {execution.market_ticker} @ {execution.fill_price:.4f} x {execution.contracts:.2f}",
            {
                "trade_id": execution.trade_id,
                "stake_usd": execution.stake_usd,
                "fee_paid": execution.fee_paid,
            },
        )

    def _build_strategy_snapshot(
        self,
        market: KalshiMarketSnapshot,
        *,
        now_utc: pd.Timestamp | None = None,
    ) -> StrategyCycleSnapshot:
        now_utc = self._normalize_utc_timestamp(now_utc)
        candles_15m = self.reference_feed.fetch_recent_candles("15m", limit=self.settings.candle_limit_15m)
        live_price = float(self.reference_feed.get_current_btc_price())

        regime_frame = compute_regime_frame(candles_15m=candles_15m, lookback=len(candles_15m))
        regime_snapshot = snapshot_from_regime_frame(regime_frame)
        pf_frame = compute_particle_filter_frame(
            candles_15m=candles_15m,
            regime_frame=regime_frame,
            config=ParticleFilterConfig(num_particles=300, lookback=len(candles_15m)),
        )
        pf_snapshot = snapshot_from_particle_filter_frame(pf_frame)

        last_observation_time = candles_15m.index[-1]
        if last_observation_time.tzinfo is None:
            last_observation_time = last_observation_time.tz_localize("UTC")
        else:
            last_observation_time = last_observation_time.tz_convert("UTC")

        fair_value = project_particle_filter_to_time(
            snapshot=pf_snapshot,
            last_observation_time=last_observation_time,
            target_time=now_utc,
        )
        pf_gap = live_price - fair_value

        market_yes_ask = market.normalized_yes_ask_dollars
        market_no_ask = market.normalized_no_ask_dollars
        market_yes_bid = market.normalized_yes_bid_dollars
        market_no_bid = market.normalized_no_bid_dollars

        active_share_price = market_yes_ask
        if regime_snapshot.allowed_side == "DOWN":
            active_share_price = market_no_ask

        pf_kelly = compute_kelly_from_pf(
            live_price=live_price,
            fair_price_pf=fair_value,
            pf_uncertainty=pf_snapshot.uncertainty,
            pf_confidence=pf_snapshot.confidence,
            regime_label=regime_snapshot.regime,
            regime_confidence=regime_snapshot.confidence,
            market_share_price=active_share_price,
            fee_rate=self.settings.fee_rate,
            alpha=self.settings.pf_alpha,
            min_gap_scale=self.settings.pf_min_gap_scale,
            fractional_kelly=self.settings.fractional_kelly,
            max_fraction=self.settings.max_fraction,
            use_confidence_shrink=self.settings.use_confidence_shrink,
        )

        p_up = self._p_up_from_kelly(pf_kelly.trade_side, pf_kelly.p_final)
        decision = decide_trade_side(
            p_up=p_up,
            market_up_price=market_yes_ask,
            market_down_price=market_no_ask,
            edge_buffer=self.settings.edge_buffer,
            regime=regime_snapshot.regime,
            allowed_side=regime_snapshot.allowed_side,
        )
        decision = apply_particle_filter_entry_filter(
            decision=decision,
            observed_price=live_price,
            fair_price=fair_value,
            min_gap=self.settings.min_pf_gap_dollars,
        )

        decision_reason = decision.reason
        decision_side = decision.side
        if pf_snapshot.confidence < self.settings.min_pf_confidence:
            decision_side = "NO_TRADE"
            decision_reason = "Particle-filter confidence is below the live trading threshold."
        elif regime_snapshot.allowed_side in {"UP", "DOWN"} and regime_snapshot.confidence <= self.settings.regime_confidence_threshold:
            decision_side = "NO_TRADE"
            decision_reason = "Regime confidence is below the live trading threshold."
        elif pf_kelly.no_trade_reason:
            decision_side = "NO_TRADE"
            decision_reason = pf_kelly.no_trade_reason

        window_start, window_end = current_15m_window(now_utc.to_pydatetime())
        return StrategyCycleSnapshot(
            market_ticker=market.ticker,
            window_start_utc=window_start.isoformat(),
            window_end_utc=window_end.isoformat(),
            reference_source=type(self.reference_feed).__name__,
            btc_reference_price=live_price,
            pf_fair_value=fair_value,
            pf_gap=pf_gap,
            pf_confidence=pf_snapshot.confidence,
            regime=regime_snapshot.regime,
            allowed_side=regime_snapshot.allowed_side,
            regime_confidence=regime_snapshot.confidence,
            p_win=pf_kelly.p_final,
            kelly_fraction=pf_kelly.kelly_fraction,
            raw_kelly=pf_kelly.raw_kelly,
            expected_log_growth=pf_kelly.expected_log_growth,
            decision_side=decision_side,
            decision_reason=decision_reason,
            model_probability=pf_kelly.p_final,
            market_yes_ask=market_yes_ask,
            market_no_ask=market_no_ask,
        )

    def _entry_window_gate(self, *, now_utc: pd.Timestamp | None = None) -> EntryWindowGate:
        now_utc = self._normalize_utc_timestamp(now_utc)
        window_start, window_end = current_15m_window(now_utc.to_pydatetime())
        last_evaluated = self.store.get_state("last_strategy_window_start_utc")
        if last_evaluated == window_start.isoformat():
            return EntryWindowGate(
                window_start=window_start,
                window_end=window_end,
                should_evaluate=False,
                reason=f"Monitoring BTC. Entry already evaluated for the {window_start.isoformat()} interval.",
            )

        grace_seconds = max(self.settings.loop_poll_seconds * 3, 15)
        seconds_since_window_start = float((now_utc - window_start).total_seconds())
        if seconds_since_window_start > grace_seconds:
            return EntryWindowGate(
                window_start=window_start,
                window_end=window_end,
                should_evaluate=False,
                reason=(
                    "Monitoring BTC. Entries are only evaluated at the start of each 15-minute interval; "
                    f"next window opens at {window_end.isoformat()}."
                ),
            )

        return EntryWindowGate(
            window_start=window_start,
            window_end=window_end,
            should_evaluate=True,
            reason=f"Evaluating BTC entry for the {window_start.isoformat()} interval.",
        )

    def _mark_window_evaluated(self, window_start: pd.Timestamp) -> None:
        self.store.set_state("last_strategy_window_start_utc", window_start.isoformat())

    def _persist_latest_analysis(self, *, cycle: StrategyCycleSnapshot, analyzed_at_utc: pd.Timestamp) -> None:
        self.store.set_state("last_analysis_completed_utc", analyzed_at_utc.isoformat())
        self.store.set_state("last_analysis_market", cycle.market_ticker)
        self.store.set_state("last_analysis_window_start_utc", cycle.window_start_utc)
        self.store.set_state("last_analysis_decision_side", cycle.decision_side)
        self.store.set_state("last_analysis_reason", cycle.decision_reason)
        self.store.set_state("last_analysis_snapshot", asdict(cycle))

    def _risk_failures(
        self,
        *,
        market: KalshiMarketSnapshot,
        cycle: StrategyCycleSnapshot,
        metrics: PortfolioMetrics,
    ) -> list[str]:
        failures: list[str] = []
        seconds_to_close = market.seconds_to_close
        if seconds_to_close is None:
            failures.append("no active market close time")
        elif seconds_to_close <= self.settings.min_seconds_to_expiry:
            failures.append("too close to expiration")

        if cycle.decision_side == "UP":
            spread = market.normalized_yes_ask_dollars - market.normalized_yes_bid_dollars
            depth = market.yes_ask_size
        else:
            spread = market.normalized_no_ask_dollars - market.normalized_no_bid_dollars
            depth = market.no_bid_size

        if spread > self.settings.max_spread_dollars:
            failures.append("spread too wide")
        if market.liquidity_dollars < self.settings.min_liquidity_dollars or depth < self.settings.min_depth_contracts:
            failures.append("liquidity too low")
        if self.store.has_market_trade(market.ticker) or self.store.get_open_trades():
            failures.append("duplicate trade")
        if metrics.current_bankroll <= self.settings.stop_bankroll_threshold:
            failures.append("bankroll too small")
        if metrics.current_bankroll * float(self.store.get_state("allocation_fraction", self.settings.allocation_fraction)) < self.settings.min_trade_notional:
            failures.append("allocation below minimum notional")
        return failures

    @staticmethod
    def _p_up_from_kelly(trade_side: str | None, p_final: float | None) -> float:
        if p_final is None:
            return 0.5
        if trade_side == "UP":
            return float(p_final)
        if trade_side == "DOWN":
            return float(1.0 - p_final)
        return 0.5

    @staticmethod
    def _normalize_utc_timestamp(ts: pd.Timestamp | None) -> pd.Timestamp:
        if ts is None:
            return pd.Timestamp.now(tz="UTC")
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")


class TradingLoopController:
    def __init__(self, loop: TradingLoop) -> None:
        self.loop = loop
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self.loop.run_forever, args=(self._stop_event,), daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
