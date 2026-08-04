"""
Black-Scholes pricing and Greeks for European-style index options
(NIFTY/BANKNIFTY options are European, so this is a reasonable fit).

No external dependencies beyond the standard library.
"""

import math


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """
    S: spot price, K: strike, T: time to expiry in YEARS, r: risk-free rate (annualized),
    sigma: implied volatility (annualized), option_type: 'CE' or 'PE'
    """
    if T <= 0:
        if option_type == "CE":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "CE":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:  # PE
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    if T <= 0:
        if option_type == "CE":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if option_type == "CE":
        return _norm_cdf(d1)
    else:
        return _norm_cdf(d1) - 1.0


def find_strike_for_delta(S: float, T: float, r: float, sigma: float,
                           option_type: str, target_delta: float,
                           strike_step: float = 50.0) -> float:
    """
    Search a strike grid (in strike_step increments, e.g. 50 for NIFTY)
    for the strike whose |delta| is closest to target_delta.
    """
    best_strike, best_diff = None, float("inf")
    lo = round((S * 0.85) / strike_step) * strike_step
    hi = round((S * 1.15) / strike_step) * strike_step
    k = lo
    while k <= hi:
        d = abs(bs_delta(S, k, T, r, sigma, option_type))
        diff = abs(d - target_delta)
        if diff < best_diff:
            best_diff, best_strike = diff, k
        k += strike_step
    return best_strike
