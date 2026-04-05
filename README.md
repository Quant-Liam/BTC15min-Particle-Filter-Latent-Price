# BTC 15m Kalshi Training Bot

This repo now has an automated execution layer on top of the existing BTC 15-minute strategy framework.

The strategy is still the decision-maker:

- regime classification stays in place
- particle-filter fair value stays in place
- PF gap and entry direction logic stay in place
- `p_win` and Kelly outputs stay in place for analytics

The new system adds:

- automated Kalshi DEMO execution
- a continuous 15-minute trading loop
- local SQLite persistence for signals, orders, fills, trades, logs, and bankroll history
- live `TRAINING P&L` on the Streamlit dashboard
- hard environment isolation so `LIVE` mode fails closed
- manual training allocation sizing with default `20%` bankroll per trade

## Project Layout

- [/Users/liamrodgers/Desktop/Python/Personal/app/dashboard.py](/Users/liamrodgers/Desktop/Python/Personal/app/dashboard.py): Streamlit dashboard for the training bot
- [/Users/liamrodgers/Desktop/Python/Personal/dashboard.py](/Users/liamrodgers/Desktop/Python/Personal/dashboard.py): thin Streamlit entrypoint wrapper
- [/Users/liamrodgers/Desktop/Python/Personal/config/settings.py](/Users/liamrodgers/Desktop/Python/Personal/config/settings.py): environment, bankroll, and safety configuration
- [/Users/liamrodgers/Desktop/Python/Personal/data/kalshi_market_client.py](/Users/liamrodgers/Desktop/Python/Personal/data/kalshi_market_client.py): Kalshi DEMO market discovery and order routing
- [/Users/liamrodgers/Desktop/Python/Personal/data/reference_price_feed.py](/Users/liamrodgers/Desktop/Python/Personal/data/reference_price_feed.py): swappable BTC reference feed abstraction, now defaulting back to Coinbase
- [/Users/liamrodgers/Desktop/Python/Personal/execution/training_executor.py](/Users/liamrodgers/Desktop/Python/Personal/execution/training_executor.py): demo-only order execution and settlement tracking
- [/Users/liamrodgers/Desktop/Python/Personal/engine/trading_loop.py](/Users/liamrodgers/Desktop/Python/Personal/engine/trading_loop.py): continuous trading loop
- [/Users/liamrodgers/Desktop/Python/Personal/analytics/pnl_engine.py](/Users/liamrodgers/Desktop/Python/Personal/analytics/pnl_engine.py): training bankroll and P&L calculations
- [/Users/liamrodgers/Desktop/Python/Personal/storage/session_store.py](/Users/liamrodgers/Desktop/Python/Personal/storage/session_store.py): SQLite persistence layer
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/](/Users/liamrodgers/Desktop/Python/Personal/btc15m/math): all model math in one place
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/features.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/features.py): price and market feature engineering
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/model.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/model.py): probability model and feature contributions
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/kelly.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/kelly.py): Kelly sizing math
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/regime.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/regime.py): Markov switching plus GARCH regime model
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/particle_filter.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/particle_filter.py): regime-aware particle filter fair-value indicator
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/strategy.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/strategy.py): trade-direction rules and regime gating
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/paper.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/paper.py): paper-trade logging and settlement
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/backtest.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/backtest.py): historical simulation engine
- [/Users/liamrodgers/Desktop/Python/Personal/run_backtest.py](/Users/liamrodgers/Desktop/Python/Personal/run_backtest.py): CLI backtest runner

Top-level `btc15m/features.py`, `btc15m/model.py`, and `btc15m/kelly.py` remain as compatibility shims, but the actual math now lives under `btc15m/math/`.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy the demo env template and fill in your Kalshi DEMO API credentials:

```bash
cp .env.example .env
```

3. Run the dashboard:

```bash
streamlit run dashboard.py
```

4. In the sidebar:

- set the training allocation fraction
- set the bankroll stop threshold
- start or pause trading
- reset the training session when needed

## Dashboard Notes

- the top bar shows `TRAINING` mode, bot status, trading active/stopped, and a `DEMO ONLY` safety badge
- the portfolio panel reports current bankroll, realized/unrealized `TRAINING P&L`, win rate, drawdown, and trade count
- the strategy panel shows the live BTC reference price, PF fair value, PF gap, regime, regime confidence, model probability, and Kelly analytics
- the execution panel shows the active Kalshi BTC market, YES/NO prices, current position, and last order status
- the log area shows trade history, fills, and runtime errors from the SQLite-backed session store

## Training Execution Rules

- `TRAINING` mode is fully functional and routes only to `https://demo-api.kalshi.co/trade-api/v2`
- `LIVE` mode is intentionally blocked in code and raises an error if selected
- trade sizing uses a manual training allocation fraction, defaulting to `20%` of bankroll
- Kelly is logged for analytics only and is not used to size demo orders
- the loop will not trade when:
  - no active BTC 15-minute market is available
  - the market is too close to expiration
  - the spread is too wide
  - liquidity is too thin
  - a duplicate trade would be created
  - bankroll is below the configured stop threshold

## Persistence

The training bot persists session state locally in SQLite at the path from `STATE_DB_PATH`.

Stored data includes:

- signals
- orders
- fills
- trades
- bankroll history
- logs

This lets the bot resume an existing training session and also lets you reset the bankroll cleanly from the dashboard.

## Regime Model

The regime engine lives in [/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/regime.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/regime.py).

It does two things:

- fits a simple GARCH(1,1) conditional volatility filter on 15-minute log returns
- fits a 3-state Markov switching model on volatility-standardized returns

The resulting states are labeled `bull`, `neutral`, and `bear` from their weighted return characteristics.

Trading rules:

- `bull`: only `UP` trades are allowed
- `bear`: only `DOWN` trades are allowed
- `neutral`: no trades are allowed

## Particle Filter Indicator

The fair-value filter lives in [/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/particle_filter.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/particle_filter.py).

It is intentionally not a second regime classifier. It uses the current MS-GARCH regime as context and estimates:

- PF fair price
- PF drift
- PF uncertainty and confidence
- `real price - PF fair price`

Useful hooks exposed in code:

- `is_price_below_pf_fair_value(...)`
- `is_price_above_pf_fair_value(...)`
- `get_pf_gap(...)`

Typical usage:

- `bull` regime: look for price below PF fair price
- `bear` regime: look for price above PF fair price

## Paper Trading

The paper-trade log keeps:

- direction and market share price
- live BTC spot price at entry
- regime, allowed side, and regime confidence
- Kelly sizing, stake, settlement, and PnL

Existing CSV logs are auto-upgraded with the new columns when the dashboard starts.

## Backtesting

CLI example:

```bash
python run_backtest.py \
  --years 2 \
  --market-up-price 0.50 \
  --market-down-price 0.50 \
  --output-xlsx pf_regime_backtest.xlsx
```

The backtest engine lives in [/Users/liamrodgers/Desktop/Python/Personal/btc15m/backtest.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/backtest.py) and uses the regime model plus particle filter exactly as they are computed in the math modules. The strategy logic then applies two layers:

- a baseline trade gate
- an additional filtered-trade gate used to compare a stricter cohort against the baseline cohort

### Baseline Trade Gate

A row becomes a baseline trade only if all of the following are true:

- regime is `bull` or `bear`, not `neutral`
- `regime_confidence > regime_confidence_threshold` with default `0.45`
- the PF fair-value gap favors the regime direction
- estimated `p_win` is above the fee-adjusted break-even probability
- Kelly fraction is greater than zero after applying `fractional_kelly` and `max_fraction`

Common baseline no-trade reasons include:

- `neutral_regime`
- `regime_confidence_below_threshold`
- `pf_gap_not_favorable`
- `negative_edge_after_fees`
- `kelly_zero_after_risk_controls`

### Additional Filtered-Trade Gate

If a row is already a baseline trade, it can also be tested against a stricter filter layer. By default the filtered cohort requires:

- `normalized_gap >= 0.8`
- `raw_gap >= 100.0`
- `pf_confidence <= 0.4`

Those defaults come from [/Users/liamrodgers/Desktop/Python/Personal/run_backtest.py](/Users/liamrodgers/Desktop/Python/Personal/run_backtest.py).

Optional filters can also be enabled:

- `regime_confidence <= max_regime_confidence` when `use_regime_filter=True`
- `p_win >= min_p_win` when `use_pwin_filter=True`
- entry hour must be in `allowed_hours` if provided
- entry weekday must be in `allowed_days` if provided

Important note: the current code treats `pf_confidence` as a maximum allowed value in the filtered cohort, not a minimum.

### How `p_win` Is Calculated

The backtest does not use historical hit rate as `p_win`. It derives `p_win` from the PF fair-value gap.

The pipeline is:

1. The particle filter estimates:
   - PF fair price
   - PF uncertainty
   - PF confidence
2. The PF fair-value gap is turned into a directional edge score:
   - for `UP`: `raw_gap = fair_price_pf - live_price`
   - for `DOWN`: `raw_gap = live_price - fair_price_pf`
3. That gap is normalized:
   - `z = raw_gap / max(pf_uncertainty, live_price * min_gap_scale, epsilon)`
4. The normalized gap is mapped through a sigmoid:
   - `p_base = 1 / (1 + exp(-alpha * z))`
5. If confidence shrink is enabled, the probability is pulled back toward `0.5`:
   - `confidence_multiplier = pf_confidence * regime_confidence` when both are present
   - `p_final = 0.5 + (p_base - 0.5) * confidence_multiplier`

The backtest uses `p_final` as `p_win`.

Interpretation:

- larger favorable PF gap -> larger `z`
- larger `z` -> larger sigmoid probability
- lower confidence -> `p_win` gets shrunk back toward `50/50`

This is a model-derived probability, not an empirically calibrated win-rate estimate.

### Share-Market Payoff and Kelly Sizing

The payoff layer assumes a binary prediction-market share, not sportsbook-style odds.

For a share bought at market price `x`:

- the share settles to `1.0` if correct
- the share settles to `0.0` if incorrect
- gross win profit per share is `1 - x`
- gross loss per share is `x`
- expected value per share is `p - x`

Fees are handled by converting raw share price into an effective all-in share price. That makes the break-even condition intuitive:

- trade only if `p_win > effective_share_price`

The Kelly layer then sizes the position using binary-share math:

- `b = (1 - x_eff) / x_eff`
- `q = 1 - p`
- `f* = (b * p - q) / b`
- equivalently `f* = (p - x_eff) / (1 - x_eff)`

Risk controls are then applied:

- `fractional_kelly`
- `max_fraction`

The backtest export and dashboard expose these fields so the economics are inspectable:

- `market_share_price`
- `effective_share_price`
- `p_win`
- `break_even_prob`
- `kelly_fraction`
- `raw_kelly`
- `expected_log_growth`

`kelly_size_pct` is simply `100 * kelly_fraction`.

### Backtest Output Columns

The workbook export includes both directional and share-economics columns.

Key directional columns:

- `trade_side`
- `regime`
- `regime_confidence`
- `pf_fair_price`
- `raw_gap`
- `normalized_gap`
- `p_win`

Key share-economics columns:

- `market_share_price`
- `effective_share_price`
- `break_even_prob`
- `kelly_fraction`
- `kelly_size_pct`
- `raw_kelly`
- `expected_log_growth`
- `share_settlement_value`
- `gross_share_pnl_per_share`
- `net_share_pnl_per_share`
- `kelly_bankroll_return`

### Useful CLI Options

- `--market-up-price`
- `--market-down-price`
- `--fee-rate`
- `--alpha`
- `--min-gap-scale`
- `--fractional-kelly`
- `--max-fraction`
- `--regime-confidence-threshold`
- `--particle-filter-particles`
- `--min-normalized-gap`
- `--min-raw-gap`
- `--max-pf-confidence`
- `--use-regime-filter`
- `--max-regime-confidence`
- `--use-pwin-filter`
- `--min-p-win`
- `--allowed-hours`
- `--allowed-days`
- `--run-sweep`

## Dependencies

Current requirements file:

- `numpy`
- `pandas`
- `plotly`
- `requests`
- `scipy`
- `statsmodels`
- `streamlit`
- `streamlit-autorefresh`
- `python-dotenv`
