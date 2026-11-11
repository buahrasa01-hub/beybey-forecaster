#!/usr/bin/env python3
"""Offline self-tests for bot.py — no network, no secrets."""
import bot

FAKE_BINARY = """(a) Base rate: about half of such scheduled events happen.
(b) Evidence for: official schedule confirms the date.
(c) Evidence against: any slip resolves No.
(d) The community is roughly where I am.
Probability: 64%"""

FAKE_MC = """(a) Base rates discussed.
(b) Evidence discussed.
(c) Residual probability kept.
Option_A: 10%
Option_B: 30%
Option_C: 25%
Option_D: 20%
Option_E: 15%"""

FAKE_NUMERIC = """(a) Status quo anchor.
(b) Trend continuation.
(c) Low and high scenarios.
Percentile 10: 120
Percentile 20: 135
Percentile 40: 148
Percentile 60: 155
Percentile 80: 170
Percentile 90: 185"""


def q_binary():
    return {"question_id": 1, "post_id": 1, "type": "binary", "title": "T?",
            "description": "d", "resolution_criteria": "r", "fine_print": "",
            "options": [], "scheduled_close_time": None,
            "community_prediction": 0.7, "scaling": {},
            "open_upper_bound": None, "open_lower_bound": None, "unit": None}


def q_mc():
    q = q_binary(); q["type"] = "multiple_choice"
    q["options"] = ["A", "B", "C", "D", "E"]; q["community_prediction"] = None
    return q


def q_numeric():
    q = q_binary(); q["type"] = "numeric"; q["community_prediction"] = None
    q["scaling"] = {"range_min": 100.0, "range_max": 200.0}
    return q


def main():
    # fallback paths
    p, r = bot.fallback_binary(q_binary())
    assert abs(p - (0.7 * 0.7 + 0.3 * 0.5)) < 1e-9, p
    probs, _ = bot.fallback_multiple_choice(q_mc())
    assert abs(sum(probs.values()) - 1.0) < 1e-9

    # LLM parsing via monkeypatched transport
    bot.call_openrouter = lambda prompt, k, m: FAKE_BINARY
    p, _ = bot.llm_binary(q_binary(), "k", "m")
    assert p == 0.64, p

    bot.call_openrouter = lambda prompt, k, m: FAKE_MC
    probs, _ = bot.llm_multiple_choice(q_mc(), "k", "m")
    assert abs(sum(probs.values()) - 1.0) < 1e-6, probs
    assert abs(probs["B"] - 0.30) < 0.02, probs

    bot.call_openrouter = lambda prompt, k, m: FAKE_NUMERIC
    pct, _ = bot.llm_numeric(q_numeric(), "k", "m")
    assert pct[10] == 120 and pct[90] == 185, pct

    # full numeric pipeline -> CDF validity
    res = bot.forecast_question(q_numeric(), "k", "m")
    cdf = res["payload"]["continuous_cdf"]
    assert len(cdf) == 201
    assert all(b >= a for a, b in zip(cdf, cdf[1:])), "CDF not monotonic"
    assert all(0.0 <= v <= 1.0 for v in cdf)
    assert max(b - a for a, b in zip(cdf, cdf[1:])) <= 0.200001, "step too big"

    # binary payload shape matches the official template
    res = bot.forecast_question(q_binary(), None, "m")
    assert res["payload"]["probability_yes"] is not None
    assert res["payload"]["continuous_cdf"] is None

    # LLM failure -> fallback engages
    def boom(prompt, k, m):
        raise RuntimeError("simulated outage")
    bot.call_openrouter = boom
    res = bot.forecast_question(q_binary(), "k", "m")
    assert res["method"] == "fallback", res["method"]

    test_cdf_validator_rules()

    print("ALL TESTS PASSED")


def test_cdf_validator_rules():
    import bot
    cases = [
        ({10: 0.0, 25: 10.0, 50: 50.0, 75: 90.0, 90: 100.0}, True, True),
        ({10: 50.0, 25: 50.0, 50: 50.0, 75: 50.0, 90: 50.0}, True, True),   # flat
        ({10: 99.0, 25: 99.5, 50: 99.9, 75: 100.0, 90: 100.0}, True, True), # piled at top
        ({10: 0.0, 25: 0.1, 50: 0.2, 75: 0.3, 90: 0.4}, True, True),        # piled at bottom
        ({10: 0.0, 25: 10.0, 50: 50.0, 75: 90.0, 90: 100.0}, False, False), # closed bounds
        ({10: 4.3, 20: 5.58, 40: 6.86, 60: 8.14, 80: 9.42, 90: 10.7}, True, True),  # discrete-like

    ]
    for pct, o_lo, o_hi in cases:
        cdf = bot.percentiles_to_cdf(pct, 0.0, 100.0, open_lo=o_lo, open_hi=o_hi)
        assert len(cdf) == 201
        steps = [cdf[i + 1] - cdf[i] for i in range(200)]
        assert min(steps) >= 5e-05 - 1e-12, (pct, min(steps))
        if o_lo:
            assert cdf[0] >= 0.001, cdf[0]
        else:
            assert cdf[0] == 0.0, cdf[0]
        if o_hi:
            assert cdf[-1] <= 0.999, cdf[-1]
        else:
            assert cdf[-1] == 1.0, cdf[-1]
    cdf16 = bot.percentiles_to_cdf({10: 4.3, 20: 5.58, 40: 6.86, 60: 8.14, 80: 9.42, 90: 10.7},
                                   0.0, 15.0, size=16)
    steps16 = [cdf16[i + 1] - cdf16[i] for i in range(15)]
    assert min(steps16) >= 0.000625, min(steps16)


if __name__ == "__main__":
    main()
