from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv


Mode = Literal["TRAINING", "LIVE"]

DEFAULT_KALSHI_DEMO_BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"
DEFAULT_STATE_DB_PATH = Path("storage/training_session.sqlite3")
DEFAULT_COINBASE_BASE_URL = "https://api.exchange.coinbase.com"


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


@dataclass(frozen=True)
class RuntimeOverrides:
    allocation_fraction: float | None = None
    starting_bankroll: float | None = None
    stop_bankroll_threshold: float | None = None
    trading_enabled: bool | None = None


@dataclass(frozen=True)
class AppSettings:
    mode: Mode
    kalshi_base_url: str
    kalshi_api_key_id: str | None
    kalshi_private_key_path: Path | None
    kalshi_series_prefix: str
    state_db_path: Path
    auto_start_trading: bool
    loop_poll_seconds: int
    cycle_min_seconds_between_runs: int
    candle_limit_15m: int
    candle_limit_1m: int
    starting_bankroll: float
    allocation_fraction: float
    stop_bankroll_threshold: float
    regime_confidence_threshold: float
    min_pf_confidence: float
    min_pf_gap_dollars: float
    pf_alpha: float
    pf_min_gap_scale: float
    fee_rate: float
    fractional_kelly: float
    max_fraction: float
    use_confidence_shrink: bool
    edge_buffer: float
    max_spread_dollars: float
    min_liquidity_dollars: float
    min_depth_contracts: float
    min_seconds_to_expiry: int
    min_trade_notional: float
    coinbase_base_url: str
    coinbase_product_id: str

    def __post_init__(self) -> None:
        if self.mode == "LIVE":
            raise RuntimeError("LIVE mode is intentionally disabled. Set APP_MODE=TRAINING to run the bot.")
        if "demo-api.kalshi.co" not in self.kalshi_base_url:
            raise RuntimeError(
                "Kalshi routing is restricted to demo only. "
                f"Configured base URL was {self.kalshi_base_url!r}."
            )
        if not 0 < self.allocation_fraction <= 1:
            raise RuntimeError("TRAINING allocation fraction must be between 0 and 1.")
        if self.starting_bankroll <= 0:
            raise RuntimeError("Starting bankroll must be positive.")

    def with_overrides(self, overrides: RuntimeOverrides | None) -> "AppSettings":
        if overrides is None:
            return self
        return AppSettings(
            mode=self.mode,
            kalshi_base_url=self.kalshi_base_url,
            kalshi_api_key_id=self.kalshi_api_key_id,
            kalshi_private_key_path=self.kalshi_private_key_path,
            kalshi_series_prefix=self.kalshi_series_prefix,
            state_db_path=self.state_db_path,
            auto_start_trading=self.auto_start_trading if overrides.trading_enabled is None else overrides.trading_enabled,
            loop_poll_seconds=self.loop_poll_seconds,
            cycle_min_seconds_between_runs=self.cycle_min_seconds_between_runs,
            candle_limit_15m=self.candle_limit_15m,
            candle_limit_1m=self.candle_limit_1m,
            starting_bankroll=self.starting_bankroll if overrides.starting_bankroll is None else float(overrides.starting_bankroll),
            allocation_fraction=self.allocation_fraction if overrides.allocation_fraction is None else float(overrides.allocation_fraction),
            stop_bankroll_threshold=(
                self.stop_bankroll_threshold
                if overrides.stop_bankroll_threshold is None
                else float(overrides.stop_bankroll_threshold)
            ),
            regime_confidence_threshold=self.regime_confidence_threshold,
            min_pf_confidence=self.min_pf_confidence,
            min_pf_gap_dollars=self.min_pf_gap_dollars,
            pf_alpha=self.pf_alpha,
            pf_min_gap_scale=self.pf_min_gap_scale,
            fee_rate=self.fee_rate,
            fractional_kelly=self.fractional_kelly,
            max_fraction=self.max_fraction,
            use_confidence_shrink=self.use_confidence_shrink,
            edge_buffer=self.edge_buffer,
            max_spread_dollars=self.max_spread_dollars,
            min_liquidity_dollars=self.min_liquidity_dollars,
            min_depth_contracts=self.min_depth_contracts,
            min_seconds_to_expiry=self.min_seconds_to_expiry,
            min_trade_notional=self.min_trade_notional,
            coinbase_base_url=self.coinbase_base_url,
            coinbase_product_id=self.coinbase_product_id,
        )

    def validate_trading_credentials(self) -> None:
        if not self.kalshi_api_key_id:
            raise RuntimeError("Missing KALSHI_API_KEY_ID for demo execution.")
        if self.kalshi_private_key_path is None:
            raise RuntimeError("Missing KALSHI_PRIVATE_KEY_PATH for demo execution.")
        if not self.kalshi_private_key_path.exists():
            raise RuntimeError(
                f"Kalshi private key file was not found at {self.kalshi_private_key_path}."
            )


def load_settings(env_path: str | Path | None = None) -> AppSettings:
    load_dotenv(dotenv_path=env_path)

    key_path_raw = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    key_path = Path(key_path_raw).expanduser() if key_path_raw else None

    return AppSettings(
        mode=os.getenv("APP_MODE", "TRAINING").strip().upper(),  # type: ignore[arg-type]
        kalshi_base_url=os.getenv("KALSHI_BASE_URL", DEFAULT_KALSHI_DEMO_BASE_URL).rstrip("/"),
        kalshi_api_key_id=os.getenv("KALSHI_API_KEY_ID"),
        kalshi_private_key_path=key_path,
        kalshi_series_prefix=os.getenv("KALSHI_BTC_SERIES_PREFIX", "KXBTC15M").strip().upper(),
        state_db_path=Path(os.getenv("STATE_DB_PATH", str(DEFAULT_STATE_DB_PATH))),
        auto_start_trading=_bool_env("AUTO_START_TRADING", False),
        loop_poll_seconds=_int_env("LOOP_POLL_SECONDS", 5),
        cycle_min_seconds_between_runs=_int_env("CYCLE_MIN_SECONDS_BETWEEN_RUNS", 60),
        candle_limit_15m=_int_env("BTC_15M_CANDLE_LIMIT", 320),
        candle_limit_1m=_int_env("BTC_1M_CANDLE_LIMIT", 720),
        starting_bankroll=_float_env("TRAINING_STARTING_BANKROLL", 100.0),
        allocation_fraction=_float_env("TRAINING_ALLOCATION_FRACTION", 0.20),
        stop_bankroll_threshold=_float_env("TRAINING_STOP_BANKROLL_THRESHOLD", 20.0),
        regime_confidence_threshold=_float_env("REGIME_CONFIDENCE_THRESHOLD", 0.45),
        min_pf_confidence=_float_env("PF_MIN_CONFIDENCE", 0.01),
        min_pf_gap_dollars=_float_env("PF_MIN_GAP_DOLLARS", 25.0),
        pf_alpha=_float_env("PF_ALPHA", 1.5),
        pf_min_gap_scale=_float_env("PF_MIN_GAP_SCALE", 0.001),
        fee_rate=_float_env("KALSHI_FEE_RATE", 0.0156),
        fractional_kelly=_float_env("FRACTIONAL_KELLY", 0.50),
        max_fraction=_float_env("MAX_FRACTION", 0.20),
        use_confidence_shrink=_bool_env("USE_CONFIDENCE_SHRINK", True),
        edge_buffer=_float_env("EDGE_BUFFER", 0.02),
        max_spread_dollars=_float_env("MAX_SPREAD_DOLLARS", 0.12),
        min_liquidity_dollars=_float_env("MIN_LIQUIDITY_DOLLARS", 100.0),
        min_depth_contracts=_float_env("MIN_DEPTH_CONTRACTS", 10.0),
        min_seconds_to_expiry=_int_env("MIN_SECONDS_TO_EXPIRY", 120),
        min_trade_notional=_float_env("MIN_TRADE_NOTIONAL", 1.0),
        coinbase_base_url=os.getenv("COINBASE_BASE_URL", DEFAULT_COINBASE_BASE_URL).rstrip("/"),
        coinbase_product_id=os.getenv("COINBASE_PRODUCT_ID", "BTC-USD").strip().upper(),
    )
