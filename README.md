# BTC 15m Kalshi Quant Research Stack

This repository contains a research and training environment for 15-minute BTC binary prediction markets on Kalshi. The project combines a particle-filter fair-value model, a historical backtesting pipeline, a demo-execution stack, and a monitoring dashboard into a single workflow designed for systematic strategy research.

The current research focus is a lean particle-filter study:

- signal generation is based on particle-filter fair value versus the 15-minute interval start price
- backtests are built to prevent lookahead bias
- trade P&L is modeled using Kalshi-style binary settlement and taker fees
- a simple three-loss circuit breaker is used to study robustness under losing streaks

Regime and Kelly components are still preserved in the codebase for future experimentation, but the current backtest is intentionally centered on the standalone particle-filter signal.

## Research Objective

The primary hypothesis under active testing is whether BTC price tends to mean revert toward the particle-filter fair value within the life of a single 15-minute contract.

At each contract start, the system asks a simple question:

- is particle-filter fair value materially above the interval start price
- is particle-filter fair value materially below the interval start price

If the model-implied distance is large enough and confidence clears the active threshold, the strategy enters a directional binary contract and holds to settlement.

## Current Strategy Configuration

The current backtest and research defaults are:

- `min_abs_gap = 25.0`
- `min_pf_confidence = 0.01`
- trade `UP` when `pf_fair_value_at_entry - interval_start_price >= 25`
- trade `DOWN` when `pf_fair_value_at_entry - interval_start_price <= -25`
- otherwise do not trade
- after 3 consecutive realized losses, skip exactly the next 15-minute interval

These thresholds can be changed in code, through environment settings for the live training stack, or from the backtest CLI.

## Methodology

### Signal Construction

The backtest signal is defined as:

```text
pf_distance = pf_fair_value_at_entry - interval_start_price
```

Where:

- `pf_fair_value_at_entry` is the particle-filter fair value available at the start of the contract
- `interval_start_price` is the opening price of that same 15-minute candle

The strategy does not compare fair value to a drifting intra-interval price for entry. It compares the model estimate to the contract's start reference price only.

### Lookahead Prevention

The backtest is explicitly built to avoid lookahead bias.

For contract start time `T`:

- the strategy uses the particle-filter snapshot from the previously completed 15-minute bar
- that snapshot is treated as the latest fair value known at `T`
- the current bar's close is used only as the realized settlement outcome

This alignment is implemented in [/Users/liamrodgers/Desktop/Python/Personal/btc15m/backtest.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/backtest.py).

### Binary Settlement and Fees

Executed trades are evaluated using binary contract settlement:

- `UP` wins if interval close is above interval open
- `DOWN` wins if interval close is below interval open
- `binary_win` is recorded as `1` for a win and `0` for a loss

Gross P&L is then adjusted for Kalshi taker fees using:

```text
fee = ceil_to_cents(0.07 * C * P * (1 - P))
```

Where:

- `C` is the number of contracts
- `P` is the entry contract price in dollars between `0` and `1`

The current implementation charges the fee once at entry for each executed trade and keeps the fee logic modular for later extensions.

### Risk Control

The current research version keeps risk controls intentionally minimal:

- PF gap threshold
- PF confidence threshold
- a one-interval pause after 3 consecutive realized losses

The cooldown is implemented as a simple state machine in the trade loop and is designed to test whether a short reset helps control clustered losses without overfitting the strategy.

## Backtest Outputs

The backtest returns a clean trade-level dataset with the fields used most directly for research:

- `contract_start`
- `contract_end`
- `interval_start_price`
- `pf_fair_value_at_entry`
- `pf_distance`
- `actual_gap`
- `entry_price`
- `exit_price`
- `confidence`
- `binary_win`
- `trade_side`
- `gross_pnl_per_trade`
- `fee`
- `net_pnl_per_trade`
- `skip_reason`

Summary outputs include:

- total trades
- win rate
- gross P&L
- total fees
- net P&L
- average net P&L per trade
- max drawdown
- longest win streak
- longest loss streak

The backtest also reports how many rows were skipped because there was no eligible signal or because the system was in the one-interval cooldown.

## Live Training System

In addition to the historical research engine, the repository includes a demo-execution environment for monitoring and training workflows against Kalshi DEMO.

The live stack includes:

- a Coinbase BTC reference feed
- Kalshi DEMO market discovery and order routing
- a continuous trading loop
- local SQLite persistence for signals, orders, fills, trades, and bankroll history
- a Streamlit dashboard for monitoring model state and training P&L

The live system currently enforces PF confidence and PF gap thresholds from configuration, while also retaining regime and Kelly analytics in the stack for future research extensions.

`LIVE` mode is intentionally blocked in code. The project is configured for training and research only.

## Project Structure

- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/backtest.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/backtest.py): historical backtest engine for the current PF-distance research workflow
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/particle_filter.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/particle_filter.py): particle-filter fair-value model
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/regime.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/regime.py): regime model preserved for future research
- [/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/kelly.py](/Users/liamrodgers/Desktop/Python/Personal/btc15m/math/kelly.py): Kelly sizing math preserved for future research
- [/Users/liamrodgers/Desktop/Python/Personal/engine/trading_loop.py](/Users/liamrodgers/Desktop/Python/Personal/engine/trading_loop.py): continuous training-loop orchestration
- [/Users/liamrodgers/Desktop/Python/Personal/execution/training_executor.py](/Users/liamrodgers/Desktop/Python/Personal/execution/training_executor.py): demo-only execution layer
- [/Users/liamrodgers/Desktop/Python/Personal/analytics/pnl_engine.py](/Users/liamrodgers/Desktop/Python/Personal/analytics/pnl_engine.py): bankroll and P&L calculations
- [/Users/liamrodgers/Desktop/Python/Personal/storage/session_store.py](/Users/liamrodgers/Desktop/Python/Personal/storage/session_store.py): SQLite persistence layer
- [/Users/liamrodgers/Desktop/Python/Personal/app/dashboard.py](/Users/liamrodgers/Desktop/Python/Personal/app/dashboard.py): Streamlit monitoring dashboard
- [/Users/liamrodgers/Desktop/Python/Personal/run_backtest.py](/Users/liamrodgers/Desktop/Python/Personal/run_backtest.py): command-line backtest runner

## Getting Started

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local environment file:

```bash
cp .env.example .env
```

Run the dashboard:

```bash
streamlit run dashboard.py
```

## Backtest Usage

Example backtest run:

```bash
python run_backtest.py \
  --years 2 \
  --min-abs-gap 25 \
  --min-pf-confidence 0.01 \
  --market-up-price 0.50 \
  --market-down-price 0.50 \
  --output-xlsx pf_distance_backtest.xlsx
```

Useful CLI parameters:

- `--min-abs-gap`: minimum absolute PF distance required to trade
- `--min-pf-confidence`: minimum PF confidence required to trade
- `--market-up-price`: assumed entry price for `UP` contracts
- `--market-down-price`: assumed entry price for `DOWN` contracts
- `--contracts-per-trade`: contracts used per simulated trade
- `--candles-15m-csv`: backtest from a local candle file instead of fetching data

## Live Configuration

The live training stack reads settings from environment variables. The most relevant PF controls are:

```bash
PF_MIN_GAP_DOLLARS=25
PF_MIN_CONFIDENCE=0.01
```

Additional runtime and bankroll settings are defined in [/Users/liamrodgers/Desktop/Python/Personal/config/settings.py](/Users/liamrodgers/Desktop/Python/Personal/config/settings.py).

## Dashboard and Monitoring

The Streamlit dashboard surfaces:

- current bankroll and training P&L
- recent trades and fills
- live BTC reference pricing
- particle-filter fair value, PF gap, and PF confidence
- configured PF confidence and PF gap thresholds
- execution and runtime status

This makes the repository usable both as a research environment and as a lightweight demo-trading operations console.

## Research Notes

This project is designed as an iterative quant research workflow rather than a finished production trading system.

Current emphasis:

- isolate the standalone predictive value of the particle-filter fair-value signal
- measure sensitivity to PF gap and confidence thresholds
- evaluate how a short post-loss cooldown affects robustness
- preserve modularity so regime filters, Kelly sizing, and other controls can be reintroduced only after the PF-only signal is properly understood

## License

This repository is for research, education, and demo-environment experimentation. Validate all assumptions independently before using any component for real-money trading.
