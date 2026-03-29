from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KellyResult:
    # fraction is the post-risk-controls bankroll fraction to allocate.
    fraction: float
    # raw_kelly is the theoretical Kelly fraction before fractional/max caps.
    raw_kelly: float
    # edge is expected value per share after fees: p_win - effective_share_price.
    edge: float
    expected_log_growth: float
    # In a binary share market, break-even probability equals all-in share cost.
    break_even_prob: float
    effective_share_price: float
    # net_odds is reported for inspection, but the contract is still modeled as a
    # binary share that settles to 1.0 or 0.0 rather than an odds-style bet.
    net_odds: float


def kelly_fraction_binary(
    p_win: float,
    share_price: float,
    fee_rate: float = 0.0,
    fractional_kelly: float = 0.5,
    max_fraction: float = 0.20,
) -> KellyResult:
    """Size a binary prediction-market share with Kelly.

    Share mechanics:
    - Buy one share at market price x.
    - The share settles to 1.0 if the prediction is correct and 0.0 otherwise.
    - Gross per-share win profit is 1 - x.
    - Gross per-share loss is x.

    After fees we use an effective all-in share cost x_eff, so:
    - break-even probability = x_eff
    - expected value per share = p_win - x_eff

    Kelly for this binary share can be written either as:
    - f* = (b * p - q) / b, where b = (1 - x_eff) / x_eff and q = 1 - p
    - f* = (p_win - x_eff) / (1 - x_eff)

    The second form is used directly below because it makes the share pricing
    interpretation explicit.
    """
    if not 0 < share_price < 1:
        raise ValueError("share_price must be between 0 and 1")
    if not 0 <= p_win <= 1:
        raise ValueError("p_win must be between 0 and 1")

    # Fees are folded into the share price so x_eff is the all-in cost of
    # acquiring one share. That makes the no-trade rule intuitive:
    # only trade when estimated p_win exceeds x_eff.
    effective_share_price = float(np.clip(share_price * (1.0 + fee_rate), 1e-6, 0.999999))
    net_odds = (1.0 - effective_share_price) / effective_share_price

    # Kelly under binary shares:
    # f* = (p_win - x_eff) / (1 - x_eff)
    # This is algebraically identical to the odds form, but makes clear that
    # x_eff is a share cost and p_win - x_eff is the per-share edge.
    raw_kelly = (float(p_win) - effective_share_price) / (1.0 - effective_share_price)
    raw_kelly = float(max(0.0, raw_kelly))

    # fractional_kelly and max_fraction are risk-policy controls, not
    # statistical calibration. They intentionally shrink the theoretical bet.
    fraction = raw_kelly * fractional_kelly
    fraction = float(np.clip(fraction, 0.0, max_fraction))

    growth = 0.0
    if fraction > 0:
        # If fraction f of bankroll is used to buy binary shares at x_eff, then:
        # - wealth multiplier on a win is 1 + f * ((1 - x_eff) / x_eff)
        # - wealth multiplier on a loss is 1 - f
        growth = float(
            p_win * np.log1p(fraction * net_odds)
            + (1.0 - p_win) * np.log1p(-fraction)
        )

    return KellyResult(
        fraction=fraction,
        raw_kelly=raw_kelly,
        edge=float(p_win - effective_share_price),
        expected_log_growth=growth,
        break_even_prob=effective_share_price,
        effective_share_price=effective_share_price,
        net_odds=float(net_odds),
    )
