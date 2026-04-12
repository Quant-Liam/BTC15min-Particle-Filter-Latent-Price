from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from analytics.pnl_engine import PnlEngine
from btc15m import (
    ParticleFilterConfig,
    compute_particle_filter_frame,
    compute_regime_frame,
    current_15m_window,
    infer_window_start_price,
    project_particle_filter_to_time,
    snapshot_from_particle_filter_frame,
)
from config.settings import load_settings
from data.kalshi_market_client import KalshiDemoClient
from data.reference_price_feed import CoinbaseBTCReferenceFeed
from engine.trading_loop import TradingLoop, TradingLoopController
from execution.training_executor import TrainingExecutor
from storage.session_store import SessionStore


@st.cache_resource
def get_runtime() -> dict[str, object]:
    settings = load_settings()
    store = SessionStore(settings.state_db_path)
    store.seed_defaults(
        starting_bankroll=settings.starting_bankroll,
        allocation_fraction=settings.allocation_fraction,
        trading_enabled=settings.auto_start_trading,
    )
    reference_feed = CoinbaseBTCReferenceFeed(
        base_url=settings.coinbase_base_url,
        product_id=settings.coinbase_product_id,
        timeout=10,
    )
    market_client = KalshiDemoClient(
        base_url=settings.kalshi_base_url,
        api_key_id=settings.kalshi_api_key_id,
        private_key_path=settings.kalshi_private_key_path,
        timeout=10,
    )
    pnl_engine = PnlEngine(store)
    executor = TrainingExecutor(settings=settings, store=store, client=market_client)
    loop = TradingLoop(
        settings=settings,
        store=store,
        reference_feed=reference_feed,
        market_client=market_client,
        executor=executor,
        pnl_engine=pnl_engine,
    )
    controller = TradingLoopController(loop)
    controller.start()
    return {
        "settings": settings,
        "store": store,
        "reference_feed": reference_feed,
        "market_client": market_client,
        "pnl_engine": pnl_engine,
        "controller": controller,
    }


def _fmt_money(value: float | None) -> str:
    if value is None or not pd.notna(value):
        return "n/a"
    return f"${float(value):,.2f}"


def _fmt_pct(value: float | None, decimals: int = 2) -> str:
    if value is None or not pd.notna(value):
        return "n/a"
    return f"{float(value) * 100:.{decimals}f}%"


def _latest_row(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


@st.cache_data(ttl=5, show_spinner=False)
def _fetch_chart_inputs(base_url: str, product_id: str, candle_limit_1m: int, candle_limit_15m: int) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    feed = CoinbaseBTCReferenceFeed(base_url=base_url, product_id=product_id, timeout=10)
    candles_1m = feed.fetch_recent_candles("1m", limit=candle_limit_1m)
    candles_15m = feed.fetch_recent_candles("15m", limit=candle_limit_15m)
    live_price = float(feed.get_current_btc_price())
    return candles_1m, candles_15m, live_price


@st.cache_data(ttl=15, show_spinner=False)
def _compute_pf_overlay(candles_15m: pd.DataFrame) -> pd.DataFrame:
    regime_frame = compute_regime_frame(candles_15m=candles_15m, lookback=len(candles_15m))
    return compute_particle_filter_frame(
        candles_15m=candles_15m,
        regime_frame=regime_frame,
        config=ParticleFilterConfig(num_particles=300, lookback=len(candles_15m)),
    )


def run_dashboard() -> None:
    st.set_page_config(page_title="BTC 15m Kalshi Training Bot", layout="wide")
    runtime = get_runtime()
    settings = runtime["settings"]
    store: SessionStore = runtime["store"]  # type: ignore[assignment]
    pnl_engine: PnlEngine = runtime["pnl_engine"]  # type: ignore[assignment]
    market_client: KalshiDemoClient = runtime["market_client"]  # type: ignore[assignment]
    controller: TradingLoopController = runtime["controller"]  # type: ignore[assignment]

    st.title("BTC 15m Kalshi Training Bot")
    st.caption(
        "Automated execution layer on top of the existing BTC 15-minute strategy. "
        "Strategy decides. The system executes. The dashboard monitors."
    )

    with st.sidebar:
        st.header("Controls")
        refresh_sec = st.slider("Auto refresh (seconds)", min_value=2, max_value=30, value=5, step=1)
        trading_enabled = bool(store.get_state("trading_enabled", settings.auto_start_trading))
        allocation_fraction = st.number_input(
            "Allocation per trade",
            min_value=0.01,
            max_value=1.00,
            value=float(store.get_state("allocation_fraction", settings.allocation_fraction)),
            step=0.01,
            format="%.2f",
            help="TRAINING mode position sizing. Kelly is logged but not used to size live demo trades.",
        )
        stop_threshold = st.number_input(
            "Stop bankroll threshold",
            min_value=1.0,
            max_value=1000.0,
            value=float(store.get_state("stop_bankroll_threshold", settings.stop_bankroll_threshold)),
            step=1.0,
            format="%.2f",
        )
        start_clicked = st.button("Start Trading", use_container_width=True)
        stop_clicked = st.button("Pause Trading", use_container_width=True)
        reset_clicked = st.button("Reset Training Session", use_container_width=True)

        st.header("Safety")
        st.code(settings.kalshi_base_url)
        st.caption("Execution is hard-locked to Kalshi DEMO. LIVE mode is disabled in code.")

    st_autorefresh(interval=refresh_sec * 1000, limit=None, key="training-dashboard-refresh")

    if start_clicked:
        store.set_state("allocation_fraction", float(allocation_fraction))
        store.set_state("stop_bankroll_threshold", float(stop_threshold))
        store.set_state("trading_enabled", True)
        store.add_log("INFO", "dashboard", "Trading enabled from dashboard.")
    if stop_clicked:
        store.set_state("trading_enabled", False)
        store.add_log("INFO", "dashboard", "Trading paused from dashboard.")
    if reset_clicked:
        store.set_state("trading_enabled", False)
        store.reset_session(
            starting_bankroll=settings.starting_bankroll,
            allocation_fraction=float(allocation_fraction),
            trading_enabled=False,
        )
        store.set_state("stop_bankroll_threshold", float(stop_threshold))

    latest_signal = _latest_row(store.get_recent_signals(limit=1))
    trades = store.get_trades_frame()
    fills = store.get_recent_fills(limit=50)
    logs = store.get_recent_logs(limit=200)
    bankroll_history = store.get_recent_bankroll_history(limit=500)
    try:
        open_marks = controller.loop.executor.mark_prices_for_open_positions() if controller.is_running() else {}  # type: ignore[attr-defined]
    except Exception:
        open_marks = {}
    marks = pnl_engine.calculate(marks_by_market=open_marks)
    runtime_state = store.get_runtime_overrides()

    top_cols = st.columns(4)
    top_cols[0].metric("Mode", settings.mode)
    top_cols[1].metric("Bot Status", str(runtime_state.get("bot_status", "IDLE")))
    top_cols[2].metric("Trading", "ACTIVE" if bool(store.get_state("trading_enabled", False)) else "STOPPED")
    top_cols[3].metric("Safety", "DEMO ONLY")

    portfolio_col, strategy_col = st.columns((5, 5))

    with portfolio_col:
        st.subheader("Portfolio")
        port_cols = st.columns(4)
        port_cols[0].metric("Current Bankroll", _fmt_money(marks.current_bankroll), delta=_fmt_pct(marks.cumulative_return))
        port_cols[1].metric("Starting Bankroll", _fmt_money(marks.starting_bankroll))
        port_cols[2].metric("Allocation", _fmt_pct(float(store.get_state("allocation_fraction", settings.allocation_fraction))))
        port_cols[3].metric("TRAINING P&L", _fmt_money(marks.realized_pnl + marks.unrealized_pnl))

        port_cols = st.columns(4)
        port_cols[0].metric("Realized P&L", _fmt_money(marks.realized_pnl))
        port_cols[1].metric("Unrealized P&L", _fmt_money(marks.unrealized_pnl))
        port_cols[2].metric("Win Rate", _fmt_pct(marks.win_rate))
        port_cols[3].metric("Drawdown", _fmt_pct(marks.drawdown))

        port_cols = st.columns(3)
        port_cols[0].metric("Trade Count", str(marks.trade_count))
        port_cols[1].metric("Open Trades", str(marks.open_trade_count))
        port_cols[2].metric("Available Cash", _fmt_money(marks.available_cash))

        if not bankroll_history.empty:
            history_plot = bankroll_history.copy()
            history_plot["created_at_utc"] = pd.to_datetime(history_plot["created_at_utc"], utc=True)
            st.line_chart(
                history_plot.set_index("created_at_utc")[["current_bankroll"]],
                height=220,
                use_container_width=True,
            )
        else:
            st.caption("TRAINING P&L history will appear here after the first loop cycle.")

    with strategy_col:
        st.subheader("Strategy")
        strat_cols = st.columns(4)
        strat_cols[0].metric("BTC Reference Price", _fmt_money(latest_signal.get("btc_reference_price")))
        strat_cols[1].metric("PF Fair Value", _fmt_money(latest_signal.get("pf_fair_value")))
        strat_cols[2].metric("PF Gap", _fmt_money(latest_signal.get("pf_gap")))
        strat_cols[3].metric("Regime", str(latest_signal.get("regime", "n/a")).upper())

        strat_cols = st.columns(4)
        strat_cols[0].metric("Allowed Side", str(latest_signal.get("allowed_side", "n/a")))
        strat_cols[1].metric("PF Confidence", _fmt_pct(latest_signal.get("pf_confidence")))
        strat_cols[2].metric("Model Probability", _fmt_pct(latest_signal.get("model_probability")))
        strat_cols[3].metric("Regime Confidence", _fmt_pct(latest_signal.get("regime_confidence")))

        strat_cols = st.columns(4)
        strat_cols[0].metric("Kelly Fraction", _fmt_pct(latest_signal.get("kelly_fraction")))
        strat_cols[1].metric("PF Min Confidence", _fmt_pct(settings.min_pf_confidence))
        strat_cols[2].metric("PF Min Gap", _fmt_money(settings.min_pf_gap_dollars))
        strat_cols[3].metric("Raw Kelly", _fmt_pct(latest_signal.get("raw_kelly")))

        strat_cols = st.columns(4)
        strat_cols[0].metric("Expected Log Growth", f"{float(latest_signal.get('expected_log_growth', 0.0)):.6f}" if latest_signal else "n/a")
        strat_cols[1].metric("Reference source", latest_signal.get("reference_source", "n/a"))
        strat_cols[2].metric("Decision", latest_signal.get("decision_side", "n/a"))
        strat_cols[3].metric("Reason", latest_signal.get("decision_reason", "n/a"))

        signal_table = pd.DataFrame(
            [
                {"item": "Decision", "value": latest_signal.get("decision_side", "n/a")},
                {"item": "Reason", "value": latest_signal.get("decision_reason", "n/a")},
            ]
        )
        st.dataframe(signal_table, use_container_width=True, hide_index=True)

    st.subheader("Live BTC Reference")
    chart_col, chart_info_col = st.columns((7, 3))
    try:
        candles_1m, candles_15m, live_chart_price = _fetch_chart_inputs(
            base_url=settings.coinbase_base_url,
            product_id=settings.coinbase_product_id,
            candle_limit_1m=settings.candle_limit_1m,
            candle_limit_15m=settings.candle_limit_15m,
        )
        pf_frame = _compute_pf_overlay(candles_15m)
        pf_snapshot = snapshot_from_particle_filter_frame(pf_frame)
        last_observation_time = candles_15m.index[-1]
        if last_observation_time.tzinfo is None:
            last_observation_time = last_observation_time.tz_localize("UTC")
        else:
            last_observation_time = last_observation_time.tz_convert("UTC")

        now_utc = pd.Timestamp.now(tz="UTC")
        live_pf_fair = project_particle_filter_to_time(
            snapshot=pf_snapshot,
            last_observation_time=last_observation_time,
            target_time=now_utc,
        )
        window_start, window_end = current_15m_window()
        window_start_price = infer_window_start_price(
            candles_1m=candles_1m,
            window_start=window_start,
            fallback_price=live_chart_price,
        )
        chart_start = max(candles_1m.index.min(), now_utc - pd.Timedelta(minutes=120))
        chart_df = candles_1m.loc[candles_1m.index >= chart_start].copy()
        pf_plot = pf_frame.loc[pf_frame.index >= chart_start, ["pf_fair_price"]].copy()
        pf_plot = pd.concat(
            [
                pf_plot,
                pd.DataFrame({"pf_fair_price": [live_pf_fair]}, index=pd.DatetimeIndex([now_utc])),
            ]
        )
        pf_plot = pf_plot[~pf_plot.index.duplicated(keep="last")]

        fig_price = go.Figure()
        fig_price.add_trace(
            go.Candlestick(
                x=chart_df.index,
                open=chart_df["open"],
                high=chart_df["high"],
                low=chart_df["low"],
                close=chart_df["close"],
                name="BTC reference (1m)",
            )
        )
        fig_price.add_trace(
            go.Scatter(
                x=pf_plot.index,
                y=pf_plot["pf_fair_price"],
                mode="lines+markers",
                name="PF fair value",
                line=dict(color="#f97316", width=2),
                marker=dict(size=5),
            )
        )
        fig_price.add_trace(
            go.Scatter(
                x=[window_start],
                y=[window_start_price],
                mode="markers",
                name="15m start",
                marker=dict(color="#f59e0b", size=9, symbol="diamond"),
            )
        )
        fig_price.add_trace(
            go.Scatter(
                x=[now_utc],
                y=[live_chart_price],
                mode="markers",
                name="Live BTC",
                marker=dict(color="#10b981", size=10),
            )
        )
        fig_price.add_hline(
            y=window_start_price,
            line_dash="dot",
            line_color="#f59e0b",
            annotation_text="current 15m open",
            annotation_position="top left",
        )
        fig_price.add_vrect(
            x0=window_start,
            x1=window_end,
            fillcolor="#dbeafe",
            opacity=0.08,
            line_width=0,
        )
        fig_price.update_layout(
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=25, b=10),
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        fig_price.update_xaxes(title_text="UTC time")
        fig_price.update_yaxes(title_text="BTC price")

        with chart_col:
            st.plotly_chart(fig_price, use_container_width=True)

        with chart_info_col:
            seconds_left = max(0, int((window_end - now_utc).total_seconds()))
            st.metric("Live BTC", _fmt_money(live_chart_price))
            st.metric("PF Fair Value", _fmt_money(live_pf_fair))
            st.metric("Spot - PF", _fmt_money(live_chart_price - live_pf_fair))
            st.metric("15m Start", _fmt_money(window_start_price))
            st.metric("Time Left", f"{seconds_left}s")
            st.caption(
                "Chart uses Coinbase for the same BTC reference feed the bot uses for strategy evaluation, "
                "with the active Kalshi market shown in the execution panel."
            )
    except Exception as exc:
        st.warning(f"Unable to render the live BTC reference chart right now: {exc}")

    st.subheader("Execution")
    exec_cols = st.columns(4)
    exec_cols[0].metric("Current Kalshi Market", str(runtime_state.get("last_cycle_market") or latest_signal.get("market_ticker", "n/a")))
    exec_cols[1].metric("YES / NO Ask", f"{float(latest_signal.get('market_yes_ask', 0.0)):.4f} / {float(latest_signal.get('market_no_ask', 0.0)):.4f}" if latest_signal else "n/a")
    open_trade_text = "None"
    if not trades.empty:
        open_trades = trades[trades["status"] == "OPEN"]
        if not open_trades.empty:
            row = open_trades.iloc[-1]
            open_trade_text = f"{row['strategy_side']} {row['market_ticker']} x {float(row['contracts']):.2f}"
    exec_cols[2].metric("Current Position", open_trade_text)
    exec_cols[3].metric("Last Order Status", str(runtime_state.get("last_order_status", "n/a")))

    last_trade = str(runtime_state.get("last_trade_summary", "n/a"))
    st.caption(f"Last trade: {last_trade}")

    market_snapshot = None
    market_error = None
    try:
        market_snapshot = market_client.discover_active_btc_market(series_prefix=settings.kalshi_series_prefix)
    except Exception as exc:  # pragma: no cover - dashboard safety
        market_error = str(exc)

    if market_snapshot is not None:
        execution_table = pd.DataFrame(
            [
                {"item": "Ticker", "value": market_snapshot.ticker},
                {"item": "Status", "value": market_snapshot.status},
                {"item": "Title", "value": market_snapshot.title},
                {"item": "YES bid / ask", "value": f"{market_snapshot.normalized_yes_bid_dollars:.4f} / {market_snapshot.normalized_yes_ask_dollars:.4f}"},
                {"item": "NO bid / ask", "value": f"{market_snapshot.normalized_no_bid_dollars:.4f} / {market_snapshot.normalized_no_ask_dollars:.4f}"},
                {"item": "Liquidity", "value": _fmt_money(market_snapshot.liquidity_dollars)},
                {"item": "Open interest", "value": f"{market_snapshot.open_interest:,.2f}"},
                {"item": "Close time", "value": market_snapshot.close_time.isoformat() if market_snapshot.close_time is not None else "n/a"},
            ]
        )
        st.dataframe(execution_table, use_container_width=True, hide_index=True)
    elif market_error:
        st.warning(f"Kalshi market lookup failed: {market_error}")

    log_col, fills_col = st.columns((6, 4))
    with log_col:
        st.subheader("Log Panel")
        if logs.empty:
            st.caption("No bot logs yet.")
        else:
            display_logs = logs.copy()
            display_logs = display_logs.rename(columns={"created_at_utc": "time_utc"})
            st.dataframe(display_logs[["time_utc", "level", "component", "message"]], use_container_width=True, hide_index=True)

    with fills_col:
        st.subheader("Fills")
        if fills.empty:
            st.caption("No fills recorded yet.")
        else:
            st.dataframe(
                fills[["created_at_utc", "market_ticker", "side", "contracts", "fill_price", "fee_paid"]],
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Trade History")
    if trades.empty:
        st.caption("No training trades placed yet.")
    else:
        trade_display = trades.copy()
        keep = [
            "created_at_utc",
            "market_ticker",
            "strategy_side",
            "contracts",
            "fill_price",
            "stake_usd",
            "status",
            "realized_pnl",
        ]
        st.dataframe(trade_display[keep].iloc[::-1], use_container_width=True, hide_index=True)

    if bool(store.get_state("trading_enabled", False)):
        if settings.kalshi_api_key_id and settings.kalshi_private_key_path:
            st.success("TRAINING execution is armed for Kalshi DEMO.")
        else:
            st.warning("Trading is enabled, but Kalshi API credentials are incomplete. Orders will fail until demo credentials are configured.")
