import unittest

from btc15m.math.kelly import kelly_fraction_binary
from btc15m.math.pf_kelly import compute_kelly_from_pf


class BinaryShareKellyTests(unittest.TestCase):
    def test_binary_share_examples_match_requested_values(self) -> None:
        for p_win, share_price, expected_fraction in (
            (0.55, 0.50, 0.10),
            (0.53, 0.50, 0.06),
            (0.60, 0.50, 0.20),
        ):
            result = kelly_fraction_binary(
                p_win=p_win,
                share_price=share_price,
                fee_rate=0.0,
                fractional_kelly=1.0,
                max_fraction=1.0,
            )
            self.assertAlmostEqual(result.raw_kelly, expected_fraction, places=12)
            self.assertAlmostEqual(result.fraction, expected_fraction, places=12)
            self.assertAlmostEqual(result.break_even_prob, share_price, places=12)

    def test_compute_kelly_from_pf_rejects_negative_edge_after_fees(self) -> None:
        result = compute_kelly_from_pf(
            live_price=100.0,
            fair_price_pf=100.1,
            pf_uncertainty=1000.0,
            pf_confidence=1.0,
            regime_label="bull",
            regime_confidence=1.0,
            market_share_price=0.50,
            fee_rate=0.02,
            alpha=1.0,
            min_gap_scale=0.0,
            fractional_kelly=1.0,
            max_fraction=1.0,
            use_confidence_shrink=False,
        )
        self.assertEqual(result.no_trade_reason, "negative_edge_after_fees")
        self.assertEqual(result.kelly_fraction, 0.0)
        self.assertGreater(result.break_even_prob, result.p_final)


if __name__ == "__main__":
    unittest.main()
