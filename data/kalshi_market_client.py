from __future__ import annotations

import base64
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlencode

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import pandas as pd
import requests

from config.settings import DEFAULT_KALSHI_DEMO_BASE_URL


@dataclass(frozen=True)
class KalshiOrderbook:
    yes_bids: list[tuple[float, float]]
    no_bids: list[tuple[float, float]]


@dataclass(frozen=True)
class KalshiMarketSnapshot:
    ticker: str
    title: str
    subtitle: str
    status: str
    event_ticker: str
    close_time: pd.Timestamp | None
    expected_expiration_time: pd.Timestamp | None
    result: str | None
    yes_bid_dollars: float
    yes_ask_dollars: float
    no_bid_dollars: float
    no_ask_dollars: float
    yes_bid_size: float
    yes_ask_size: float
    no_bid_size: float
    liquidity_dollars: float
    volume: float
    open_interest: float
    last_price_dollars: float | None
    fractional_trading_enabled: bool
    rules_primary: str
    yes_mid_dollars: float
    no_mid_dollars: float

    @property
    def seconds_to_close(self) -> float | None:
        if self.close_time is None:
            return None
        return float((self.close_time - pd.Timestamp.now(tz="UTC")).total_seconds())

    @property
    def normalized_yes_ask_dollars(self) -> float:
        return _normalized_binary_price(
            primary=self.yes_ask_dollars,
            complementary_bid=self.no_bid_dollars,
            fallback_bid=self.yes_bid_dollars,
        )

    @property
    def normalized_no_ask_dollars(self) -> float:
        return _normalized_binary_price(
            primary=self.no_ask_dollars,
            complementary_bid=self.yes_bid_dollars,
            fallback_bid=self.no_bid_dollars,
        )

    @property
    def normalized_yes_bid_dollars(self) -> float:
        return _normalized_binary_price(
            primary=self.yes_bid_dollars,
            complementary_bid=self.no_ask_dollars,
            fallback_bid=self.yes_ask_dollars,
        )

    @property
    def normalized_no_bid_dollars(self) -> float:
        return _normalized_binary_price(
            primary=self.no_bid_dollars,
            complementary_bid=self.yes_ask_dollars,
            fallback_bid=self.no_ask_dollars,
        )


class KalshiDemoClient:
    def __init__(
        self,
        base_url: str = DEFAULT_KALSHI_DEMO_BASE_URL,
        api_key_id: str | None = None,
        private_key_path: str | Path | None = None,
        timeout: int = 10,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        if "demo-api.kalshi.co" not in self.base_url:
            raise RuntimeError(
                "Execution routing must stay on Kalshi DEMO only. "
                f"Refusing base URL {self.base_url!r}."
            )
        self.api_key_id = api_key_id
        self.private_key_path = Path(private_key_path).expanduser() if private_key_path else None
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "btc-15m-training-bot/1.0",
            }
        )
        self._private_key = None

    def discover_active_btc_market(self, series_prefix: str = "KXBTC15M") -> KalshiMarketSnapshot | None:
        raw_prefix = series_prefix.strip()
        prefixes = []
        for candidate in (raw_prefix, raw_prefix.upper(), raw_prefix.lower()):
            if candidate and candidate not in prefixes:
                prefixes.append(candidate)

        candidates: list[dict[str, Any]] = []
        for prefix in prefixes:
            payload = self.get_markets(limit=200, status="open", series_ticker=prefix)
            markets = payload.get("markets", [])
            candidates.extend(
                market
                for market in markets
                if self._looks_like_btc_15m_market(market, prefix=raw_prefix.upper())
            )
            if candidates:
                break

        if not candidates:
            for prefix in prefixes:
                payload = self.get_markets(limit=200, series_ticker=prefix)
                markets = payload.get("markets", [])
                candidates.extend(
                    market
                    for market in markets
                    if str(market.get("status", "")).lower() in {"active", "open", "paused"}
                    and self._looks_like_btc_15m_market(market, prefix=raw_prefix.upper())
                )
                if candidates:
                    break

        if not candidates:
            return None

        candidates.sort(
            key=lambda market: (
                self._status_rank(str(market.get("status", ""))),
                self._parse_timestamp(market.get("close_time")) or pd.Timestamp.max.tz_localize("UTC"),
            )
        )
        return self._to_market_snapshot(candidates[0])

    def get_markets(
        self,
        *,
        limit: int = 100,
        cursor: str | None = None,
        status: str | None = None,
        tickers: list[str] | None = None,
        series_ticker: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, object] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        if tickers:
            params["tickers"] = ",".join(tickers)
        if series_ticker:
            params["series_ticker"] = series_ticker
        return self._request("GET", "/markets", params=params, auth=False)

    def get_market(self, ticker: str) -> KalshiMarketSnapshot:
        payload = self._request("GET", f"/markets/{ticker}", auth=False)
        return self._to_market_snapshot(payload["market"])

    def get_market_orderbook(self, ticker: str) -> KalshiOrderbook:
        payload = self._request("GET", f"/markets/{ticker}/orderbook", auth=False)
        raw = payload.get("orderbook_fp", {})
        return KalshiOrderbook(
            yes_bids=[(float(price), float(size)) for price, size in raw.get("yes_dollars", [])],
            no_bids=[(float(price), float(size)) for price, size in raw.get("no_dollars", [])],
        )

    def get_balance(self) -> dict[str, Any]:
        return self._request("GET", "/portfolio/balance", auth=True)

    def get_order(self, order_id: str) -> dict[str, Any]:
        return self._request("GET", f"/portfolio/orders/{order_id}", auth=True)

    def get_fills(self, *, order_id: str | None = None, ticker: str | None = None, limit: int = 100) -> dict[str, Any]:
        params: dict[str, object] = {"limit": limit}
        if order_id:
            params["order_id"] = order_id
        if ticker:
            params["ticker"] = ticker
        return self._request("GET", "/portfolio/fills", params=params, auth=True)

    def get_positions(self, *, ticker: str | None = None, limit: int = 100) -> dict[str, Any]:
        params: dict[str, object] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        return self._request("GET", "/portfolio/positions", params=params, auth=True)

    def create_buy_order(
        self,
        *,
        ticker: str,
        side: str,
        contracts: Decimal,
        price_dollars: Decimal,
        client_order_id: str,
    ) -> dict[str, Any]:
        side_lower = side.lower()
        if side_lower not in {"yes", "no"}:
            raise ValueError("Order side must be 'yes' or 'no'.")

        count_fp = self._format_count(contracts)
        price_fp = self._format_price(price_dollars)
        payload: dict[str, Any] = {
            "ticker": ticker,
            "side": side_lower,
            "action": "buy",
            "client_order_id": client_order_id,
            "count_fp": count_fp,
            "time_in_force": "fill_or_kill",
            "buy_max_cost": int((contracts * price_dollars * Decimal("100")).to_integral_value(rounding=ROUND_DOWN)),
        }
        if side_lower == "yes":
            payload["yes_price_dollars"] = price_fp
        else:
            payload["no_price_dollars"] = price_fp
        return self._request("POST", "/portfolio/orders", json_body=payload, auth=True)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, object] | None = None,
        json_body: dict[str, Any] | None = None,
        auth: bool,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers: dict[str, str] = {}
        if auth:
            headers.update(self._auth_headers(method=method, path=path))
        if json_body is not None:
            headers["Content-Type"] = "application/json"
        response = self.session.request(
            method=method,
            url=url,
            params=params,
            json=json_body,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _auth_headers(self, *, method: str, path: str) -> dict[str, str]:
        if not self.api_key_id:
            raise RuntimeError("Kalshi API key ID is required for authenticated demo execution.")
        private_key = self._load_private_key()
        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}{method.upper()}{path}"
        signature = private_key.sign(
            message.encode("utf-8"),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("ascii"),
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

    def _load_private_key(self):
        if self._private_key is not None:
            return self._private_key
        if self.private_key_path is None:
            raise RuntimeError("Kalshi private key path is required for authenticated demo execution.")
        key_bytes = self.private_key_path.read_bytes()
        self._private_key = serialization.load_pem_private_key(key_bytes, password=None)
        return self._private_key

    @staticmethod
    def _looks_like_btc_15m_market(market: dict[str, Any], *, prefix: str) -> bool:
        ticker = str(market.get("ticker", "")).upper()
        title = str(market.get("title", "")).upper()
        subtitle = str(market.get("subtitle", "")).upper()
        event_ticker = str(market.get("event_ticker", "")).upper()
        status = str(market.get("status", "")).lower()
        return (
            status in {"active", "open", "paused"}
            and (
            ticker.startswith(prefix)
            or event_ticker.startswith(prefix)
            or ("BITCOIN" in title and "15" in title)
            or ("BITCOIN" in subtitle and "15" in subtitle)
            )
        )

    @staticmethod
    def _parse_timestamp(value: object) -> pd.Timestamp | None:
        if value in (None, ""):
            return None
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def _to_market_snapshot(self, market: dict[str, Any]) -> KalshiMarketSnapshot:
        yes_bid = float(market.get("yes_bid_dollars") or 0.0)
        yes_ask = float(market.get("yes_ask_dollars") or (1.0 - float(market.get("no_bid_dollars") or 0.0)))
        no_bid = float(market.get("no_bid_dollars") or 0.0)
        no_ask = float(market.get("no_ask_dollars") or (1.0 - float(market.get("yes_bid_dollars") or 0.0)))
        yes_mid = self._midpoint(yes_bid, yes_ask)
        no_mid = self._midpoint(no_bid, no_ask)
        return KalshiMarketSnapshot(
            ticker=str(market.get("ticker", "")),
            title=str(market.get("title", "")),
            subtitle=str(market.get("subtitle", "")),
            status=str(market.get("status", "")),
            event_ticker=str(market.get("event_ticker", "")),
            close_time=self._parse_timestamp(market.get("close_time")),
            expected_expiration_time=self._parse_timestamp(market.get("expected_expiration_time")),
            result=str(market["result"]) if market.get("result") is not None else None,
            yes_bid_dollars=yes_bid,
            yes_ask_dollars=yes_ask,
            no_bid_dollars=no_bid,
            no_ask_dollars=no_ask,
            yes_bid_size=float(market.get("yes_bid_size_fp") or 0.0),
            yes_ask_size=float(market.get("yes_ask_size_fp") or 0.0),
            no_bid_size=float(market.get("no_bid_size_fp") or 0.0),
            liquidity_dollars=float(market.get("liquidity_dollars") or 0.0),
            volume=float(market.get("volume_fp") or 0.0),
            open_interest=float(market.get("open_interest_fp") or 0.0),
            last_price_dollars=float(market["last_price_dollars"]) if market.get("last_price_dollars") not in (None, "") else None,
            fractional_trading_enabled=bool(market.get("fractional_trading_enabled", False)),
            rules_primary=str(market.get("rules_primary", "")),
            yes_mid_dollars=yes_mid,
            no_mid_dollars=no_mid,
        )

    @staticmethod
    def _midpoint(bid: float, ask: float) -> float:
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        return max(bid, ask, 0.0)

    @staticmethod
    def _status_rank(status: str) -> int:
        status_lower = status.lower()
        if status_lower in {"active", "open"}:
            return 0
        if status_lower == "paused":
            return 1
        return 2

    @staticmethod
    def _format_price(value: Decimal) -> str:
        return str(value.quantize(Decimal("0.0001"), rounding=ROUND_DOWN))

    @staticmethod
    def _format_count(value: Decimal) -> str:
        return str(value.quantize(Decimal("0.01"), rounding=ROUND_DOWN))


def _normalized_binary_price(primary: float, complementary_bid: float, fallback_bid: float) -> float:
    if 0.0 < float(primary) < 1.0:
        return float(primary)
    if 0.0 < float(complementary_bid) < 1.0:
        return float(min(max(1.0 - float(complementary_bid), 0.01), 0.99))
    if 0.0 < float(fallback_bid) < 1.0:
        return float(min(max(float(fallback_bid), 0.01), 0.99))
    return 0.5
