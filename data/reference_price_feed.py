from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import pandas as pd
import requests

from btc15m.coinbase import CoinbaseClient


CandleInterval = Literal["1m", "15m"]


class ReferencePriceFeed(Protocol):
    def get_current_btc_price(self) -> float:
        ...

    def fetch_recent_candles(self, interval: CandleInterval, limit: int) -> pd.DataFrame:
        ...


@dataclass
class CoinbaseBTCReferenceFeed:
    base_url: str = "https://api.exchange.coinbase.com"
    product_id: str = "BTC-USD"
    timeout: int = 10

    def __post_init__(self) -> None:
        self.client = CoinbaseClient(base_url=self.base_url, timeout=self.timeout)

    def get_current_btc_price(self) -> float:
        return float(self.client.fetch_live_price(product_id=self.product_id))

    def fetch_recent_candles(self, interval: CandleInterval, limit: int) -> pd.DataFrame:
        granularity_map = {"1m": 60, "15m": 900}
        if interval not in granularity_map:
            raise ValueError("interval must be '1m' or '15m'")
        return self.client.fetch_candles(
            product_id=self.product_id,
            granularity=granularity_map[interval],
            limit=limit,
        )


@dataclass
class BinanceBTCReferenceFeed:
    base_url: str = "https://api.binance.com"
    symbol: str = "BTCUSDT"
    timeout: int = 10

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "btc-15m-training-bot/1.0",
            }
        )

    def get_current_btc_price(self) -> float:
        payload = self._get("/api/v3/ticker/price", params={"symbol": self.symbol})
        return float(payload["price"])

    def fetch_recent_candles(self, interval: CandleInterval, limit: int) -> pd.DataFrame:
        if interval not in {"1m", "15m"}:
            raise ValueError("interval must be '1m' or '15m'")
        if limit <= 0:
            raise ValueError("limit must be positive")

        payload = self._get(
            "/api/v3/klines",
            params={
                "symbol": self.symbol,
                "interval": interval,
                "limit": min(limit, 1000),
            },
        )
        if not payload:
            raise RuntimeError("Reference feed returned no BTC candles.")

        frame = pd.DataFrame(
            payload,
            columns=[
                "open_time_ms",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time_ms",
                "quote_volume",
                "trade_count",
                "taker_base_volume",
                "taker_quote_volume",
                "ignore",
            ],
        )
        frame["timestamp"] = pd.to_datetime(frame["open_time_ms"], unit="ms", utc=True)
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = (
            frame.set_index("timestamp")
            .sort_index()
            .dropna(subset=["open", "high", "low", "close"])
            .tail(limit)
        )
        return frame[["open", "high", "low", "close", "volume"]]

    def _get(self, path: str, params: dict[str, object] | None = None) -> dict | list:
        response = self.session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
