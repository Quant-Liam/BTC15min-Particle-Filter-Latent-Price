from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator

import pandas as pd


class SessionStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._ensure_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def _ensure_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runtime_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_utc TEXT NOT NULL,
                    market_ticker TEXT,
                    window_start_utc TEXT,
                    window_end_utc TEXT,
                    reference_source TEXT,
                    btc_reference_price REAL,
                    pf_fair_value REAL,
                    pf_gap REAL,
                    regime TEXT,
                    allowed_side TEXT,
                    regime_confidence REAL,
                    p_win REAL,
                    model_probability REAL,
                    kelly_fraction REAL,
                    raw_kelly REAL,
                    expected_log_growth REAL,
                    decision_side TEXT,
                    decision_reason TEXT,
                    market_yes_ask REAL,
                    market_no_ask REAL,
                    signal_payload_json TEXT
                );

                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    client_order_id TEXT UNIQUE NOT NULL,
                    kalshi_order_id TEXT,
                    market_ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_action TEXT NOT NULL,
                    contracts REAL NOT NULL,
                    requested_price REAL NOT NULL,
                    order_status TEXT NOT NULL,
                    response_payload_json TEXT
                );

                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_utc TEXT NOT NULL,
                    fill_id TEXT,
                    client_order_id TEXT NOT NULL,
                    kalshi_order_id TEXT,
                    market_ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    contracts REAL NOT NULL,
                    fill_price REAL NOT NULL,
                    fee_paid REAL NOT NULL,
                    fill_payload_json TEXT
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    signal_id INTEGER,
                    client_order_id TEXT,
                    kalshi_order_id TEXT,
                    market_ticker TEXT NOT NULL,
                    event_ticker TEXT,
                    title TEXT,
                    strategy_side TEXT NOT NULL,
                    kalshi_side TEXT NOT NULL,
                    contracts REAL NOT NULL,
                    fill_price REAL NOT NULL,
                    fee_paid REAL NOT NULL,
                    stake_usd REAL NOT NULL,
                    allocation_fraction REAL NOT NULL,
                    btc_reference_price REAL,
                    pf_fair_value REAL,
                    pf_gap REAL,
                    regime TEXT,
                    regime_confidence REAL,
                    p_win REAL,
                    kelly_fraction REAL,
                    raw_kelly REAL,
                    expected_log_growth REAL,
                    status TEXT NOT NULL,
                    close_time_utc TEXT,
                    settled_at_utc TEXT,
                    settlement_result TEXT,
                    settlement_value REAL,
                    realized_pnl REAL,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS bankroll_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_utc TEXT NOT NULL,
                    current_bankroll REAL NOT NULL,
                    available_cash REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    cumulative_return REAL NOT NULL,
                    drawdown REAL NOT NULL,
                    peak_bankroll REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at_utc TEXT NOT NULL,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    context_json TEXT
                );
                """
            )
            self._ensure_column(conn, "signals", "model_probability", "REAL")
            self._ensure_column(conn, "signals", "market_yes_ask", "REAL")
            self._ensure_column(conn, "signals", "market_no_ask", "REAL")
            conn.commit()

    def seed_defaults(self, *, starting_bankroll: float, allocation_fraction: float, trading_enabled: bool) -> None:
        if self.get_state("starting_bankroll") is None:
            self.set_state("starting_bankroll", starting_bankroll)
        if self.get_state("allocation_fraction") is None:
            self.set_state("allocation_fraction", allocation_fraction)
        if self.get_state("trading_enabled") is None:
            self.set_state("trading_enabled", trading_enabled)
        if self.get_state("bot_status") is None:
            self.set_state("bot_status", "IDLE")
        if self.get_state("bot_message") is None:
            self.set_state("bot_message", "Waiting for dashboard start.")

    def set_state(self, key: str, value: Any) -> None:
        payload = json.dumps(value)
        now = pd.Timestamp.now(tz="UTC").isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runtime_state (key, value, updated_at_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (key, payload, now),
            )
            conn.commit()

    def get_state(self, key: str, default: Any = None) -> Any:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM runtime_state WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return default

    def get_runtime_overrides(self) -> dict[str, Any]:
        return {
            "starting_bankroll": self.get_state("starting_bankroll"),
            "allocation_fraction": self.get_state("allocation_fraction"),
            "stop_bankroll_threshold": self.get_state("stop_bankroll_threshold"),
            "trading_enabled": self.get_state("trading_enabled"),
            "bot_status": self.get_state("bot_status", "IDLE"),
            "bot_message": self.get_state("bot_message", ""),
            "last_cycle_market": self.get_state("last_cycle_market"),
            "last_order_status": self.get_state("last_order_status"),
            "last_trade_summary": self.get_state("last_trade_summary"),
            "last_error": self.get_state("last_error"),
            "last_heartbeat_utc": self.get_state("last_heartbeat_utc"),
            "last_cycle_completed_utc": self.get_state("last_cycle_completed_utc"),
        }

    def set_bot_status(self, status: str, message: str) -> None:
        self.set_state("bot_status", status)
        self.set_state("bot_message", message)
        self.set_state("last_heartbeat_utc", pd.Timestamp.now(tz="UTC").isoformat())

    def add_log(self, level: str, component: str, message: str, context: dict[str, Any] | None = None) -> None:
        now = pd.Timestamp.now(tz="UTC").isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO logs (created_at_utc, level, component, message, context_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (now, level.upper(), component, message, json.dumps(context or {})),
            )
            conn.commit()

    def record_signal(self, payload: dict[str, Any]) -> int:
        now = pd.Timestamp.now(tz="UTC").isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO signals (
                    created_at_utc, market_ticker, window_start_utc, window_end_utc,
                    reference_source, btc_reference_price, pf_fair_value, pf_gap,
                    regime, allowed_side, regime_confidence, p_win, model_probability,
                    kelly_fraction, raw_kelly, expected_log_growth, decision_side, decision_reason,
                    market_yes_ask, market_no_ask,
                    signal_payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    payload.get("market_ticker"),
                    payload.get("window_start_utc"),
                    payload.get("window_end_utc"),
                    payload.get("reference_source"),
                    payload.get("btc_reference_price"),
                    payload.get("pf_fair_value"),
                    payload.get("pf_gap"),
                    payload.get("regime"),
                    payload.get("allowed_side"),
                    payload.get("regime_confidence"),
                    payload.get("p_win"),
                    payload.get("model_probability"),
                    payload.get("kelly_fraction"),
                    payload.get("raw_kelly"),
                    payload.get("expected_log_growth"),
                    payload.get("decision_side"),
                    payload.get("decision_reason"),
                    payload.get("market_yes_ask"),
                    payload.get("market_no_ask"),
                    json.dumps(payload),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def upsert_order(self, payload: dict[str, Any]) -> None:
        now = pd.Timestamp.now(tz="UTC").isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO orders (
                    created_at_utc, updated_at_utc, client_order_id, kalshi_order_id,
                    market_ticker, side, order_action, contracts, requested_price,
                    order_status, response_payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_order_id) DO UPDATE SET
                    updated_at_utc = excluded.updated_at_utc,
                    kalshi_order_id = excluded.kalshi_order_id,
                    order_status = excluded.order_status,
                    response_payload_json = excluded.response_payload_json
                """,
                (
                    now,
                    now,
                    payload["client_order_id"],
                    payload.get("kalshi_order_id"),
                    payload["market_ticker"],
                    payload["side"],
                    payload.get("order_action", "buy"),
                    float(payload["contracts"]),
                    float(payload["requested_price"]),
                    payload["order_status"],
                    json.dumps(payload.get("response_payload") or {}),
                ),
            )
            conn.commit()

    def record_fill(self, payload: dict[str, Any]) -> None:
        now = pd.Timestamp.now(tz="UTC").isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO fills (
                    created_at_utc, fill_id, client_order_id, kalshi_order_id,
                    market_ticker, side, contracts, fill_price, fee_paid, fill_payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    payload.get("fill_id"),
                    payload["client_order_id"],
                    payload.get("kalshi_order_id"),
                    payload["market_ticker"],
                    payload["side"],
                    float(payload["contracts"]),
                    float(payload["fill_price"]),
                    float(payload.get("fee_paid", 0.0)),
                    json.dumps(payload.get("fill_payload") or {}),
                ),
            )
            conn.commit()

    def create_trade(self, payload: dict[str, Any]) -> int:
        now = pd.Timestamp.now(tz="UTC").isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO trades (
                    created_at_utc, updated_at_utc, signal_id, client_order_id,
                    kalshi_order_id, market_ticker, event_ticker, title, strategy_side,
                    kalshi_side, contracts, fill_price, fee_paid, stake_usd,
                    allocation_fraction, btc_reference_price, pf_fair_value, pf_gap,
                    regime, regime_confidence, p_win, kelly_fraction, raw_kelly,
                    expected_log_growth, status, close_time_utc, settled_at_utc,
                    settlement_result, settlement_value, realized_pnl, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    now,
                    payload.get("signal_id"),
                    payload.get("client_order_id"),
                    payload.get("kalshi_order_id"),
                    payload["market_ticker"],
                    payload.get("event_ticker"),
                    payload.get("title"),
                    payload["strategy_side"],
                    payload["kalshi_side"],
                    float(payload["contracts"]),
                    float(payload["fill_price"]),
                    float(payload.get("fee_paid", 0.0)),
                    float(payload["stake_usd"]),
                    float(payload["allocation_fraction"]),
                    payload.get("btc_reference_price"),
                    payload.get("pf_fair_value"),
                    payload.get("pf_gap"),
                    payload.get("regime"),
                    payload.get("regime_confidence"),
                    payload.get("p_win"),
                    payload.get("kelly_fraction"),
                    payload.get("raw_kelly"),
                    payload.get("expected_log_growth"),
                    payload.get("status", "OPEN"),
                    payload.get("close_time_utc"),
                    payload.get("settled_at_utc"),
                    payload.get("settlement_result"),
                    payload.get("settlement_value"),
                    payload.get("realized_pnl"),
                    payload.get("notes"),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def settle_trade(
        self,
        *,
        trade_id: int,
        settlement_result: str,
        settlement_value: float,
        realized_pnl: float,
    ) -> None:
        now = pd.Timestamp.now(tz="UTC").isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE trades
                SET updated_at_utc = ?,
                    status = 'SETTLED',
                    settled_at_utc = ?,
                    settlement_result = ?,
                    settlement_value = ?,
                    realized_pnl = ?
                WHERE id = ?
                """,
                (now, now, settlement_result, settlement_value, realized_pnl, trade_id),
            )
            conn.commit()

    def get_open_trades(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY created_at_utc ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def has_market_trade(self, market_ticker: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM trades
                WHERE market_ticker = ?
                LIMIT 1
                """,
                (market_ticker,),
            ).fetchone()
        return row is not None

    def get_trades_frame(self) -> pd.DataFrame:
        with self._connect() as conn:
            frame = pd.read_sql_query("SELECT * FROM trades ORDER BY created_at_utc ASC", conn)
        return frame

    def record_bankroll_snapshot(self, payload: dict[str, float]) -> None:
        now = pd.Timestamp.now(tz="UTC").isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO bankroll_history (
                    created_at_utc, current_bankroll, available_cash, realized_pnl,
                    unrealized_pnl, cumulative_return, drawdown, peak_bankroll
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    float(payload["current_bankroll"]),
                    float(payload["available_cash"]),
                    float(payload["realized_pnl"]),
                    float(payload["unrealized_pnl"]),
                    float(payload["cumulative_return"]),
                    float(payload["drawdown"]),
                    float(payload["peak_bankroll"]),
                ),
            )
            conn.commit()

    def get_recent_bankroll_history(self, limit: int = 500) -> pd.DataFrame:
        with self._connect() as conn:
            frame = pd.read_sql_query(
                """
                SELECT *
                FROM bankroll_history
                ORDER BY id DESC
                LIMIT ?
                """,
                conn,
                params=(limit,),
            )
        return frame.iloc[::-1].reset_index(drop=True) if not frame.empty else frame

    def get_recent_signals(self, limit: int = 200) -> pd.DataFrame:
        with self._connect() as conn:
            frame = pd.read_sql_query(
                "SELECT * FROM signals ORDER BY id DESC LIMIT ?",
                conn,
                params=(limit,),
            )
        return frame

    def get_recent_orders(self, limit: int = 200) -> pd.DataFrame:
        with self._connect() as conn:
            frame = pd.read_sql_query(
                "SELECT * FROM orders ORDER BY id DESC LIMIT ?",
                conn,
                params=(limit,),
            )
        return frame

    def get_recent_fills(self, limit: int = 200) -> pd.DataFrame:
        with self._connect() as conn:
            frame = pd.read_sql_query(
                "SELECT * FROM fills ORDER BY id DESC LIMIT ?",
                conn,
                params=(limit,),
            )
        return frame

    def get_recent_logs(self, limit: int = 300) -> pd.DataFrame:
        with self._connect() as conn:
            frame = pd.read_sql_query(
                "SELECT * FROM logs ORDER BY id DESC LIMIT ?",
                conn,
                params=(limit,),
            )
        return frame

    def get_latest_bankroll_snapshot(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM bankroll_history ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row is not None else None

    def reset_session(self, *, starting_bankroll: float, allocation_fraction: float, trading_enabled: bool) -> None:
        with self._lock, self._connect() as conn:
            for table in ("signals", "orders", "fills", "trades", "bankroll_history", "logs", "runtime_state"):
                conn.execute(f"DELETE FROM {table}")
            conn.commit()
        self.seed_defaults(
            starting_bankroll=starting_bankroll,
            allocation_fraction=allocation_fraction,
            trading_enabled=trading_enabled,
        )
        self.set_state("stop_bankroll_threshold", None)
        self.add_log("INFO", "session_store", "Training session reset.")

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {row[1] for row in rows}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
