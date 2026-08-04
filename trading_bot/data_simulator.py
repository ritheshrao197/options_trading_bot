"""
Underlying Price Data Feeds for Options Trading Bot.

Provides:
- SyntheticUnderlyingFeed: Geometric Brownian Motion generator.
- HistoricalCSVFeed: Loads real historical spot prices from a local CSV file.
- HistoricalYFinanceFeed: Downloads real historical index/stock prices via Yahoo Finance (e.g. ^NSEI).
- HistoricalUnderlyingFeed: Unified feed wrapper conforming to .generate() -> np.ndarray.
"""

import os
import numpy as np


class SyntheticUnderlyingFeed:
    """Generates a synthetic price path via Geometric Brownian Motion."""
    def __init__(self, start_price: float = 24500.0, annual_vol: float = 0.13,
                 annual_drift: float = 0.10, trading_days: int = 252, seed: int = 42):
        self.start_price = start_price
        self.annual_vol = annual_vol
        self.annual_drift = annual_drift
        self.trading_days = trading_days
        self.seed = seed

    def generate(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        dt = 1.0 / 252.0
        n = self.trading_days
        z = rng.standard_normal(n)
        drift = (self.annual_drift - 0.5 * self.annual_vol ** 2) * dt
        diffusion = self.annual_vol * np.sqrt(dt) * z
        log_returns = drift + diffusion
        prices = self.start_price * np.exp(np.cumsum(log_returns))
        return np.concatenate([[self.start_price], prices])


class HistoricalCSVFeed:
    """Loads historical prices from a CSV file containing Date and Close price columns."""
    def __init__(self, csv_path: str, price_col: str = "Close", trading_days: int = None):
        self.csv_path = csv_path
        self.price_col = price_col
        self.trading_days = trading_days

    def generate(self) -> np.ndarray:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Historical CSV file not found: {self.csv_path}")

        import pandas as pd
        df = pd.read_csv(self.csv_path)

        # Match column case-insensitively
        matched_col = None
        for col in df.columns:
            if col.strip().lower() == self.price_col.lower():
                matched_col = col
                break

        if matched_col is None:
            raise ValueError(f"Column '{self.price_col}' not found in CSV headers: {list(df.columns)}")

        prices = df[matched_col].dropna().values.astype(float)

        if self.trading_days and len(prices) > self.trading_days:
            prices = prices[-self.trading_days:]

        return prices


class HistoricalYFinanceFeed:
    """Fetches real historical spot prices for an index or ticker (e.g. '^NSEI' for NIFTY 50)."""
    def __init__(self, symbol: str = "^NSEI", period: str = "2y", trading_days: int = 252):
        self.symbol = symbol
        self.period = period
        self.trading_days = trading_days

    def generate(self) -> np.ndarray:
        try:
            import yfinance as yf
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period=self.period)

            if df.empty or "Close" not in df.columns:
                raise ValueError(f"No price data returned from Yahoo Finance for symbol '{self.symbol}'.")

            prices = df["Close"].dropna().values.astype(float)
            if self.trading_days and len(prices) > self.trading_days:
                prices = prices[-self.trading_days:]
            return prices

        except Exception as e:
            print(f"[Warning] Failed to fetch yfinance data for {self.symbol}: {e}. Falling back to synthetic feed.")
            fallback = SyntheticUnderlyingFeed(trading_days=self.trading_days or 252)
            return fallback.generate()


class HistoricalUnderlyingFeed:
    """
    Unified feed factory.
    Supports feed_type: 'synthetic', 'yfinance', or 'csv'.
    Exposes .generate() -> np.ndarray interface.
    """
    def __init__(self, feed_type: str = "synthetic", symbol: str = "^NSEI",
                 csv_path: str = None, trading_days: int = 252, seed: int = 42,
                 start_price: float = 24500.0):
        self.feed_type = feed_type.lower()
        self.symbol = symbol
        self.csv_path = csv_path
        self.trading_days = trading_days
        self.seed = seed
        self.start_price = start_price

    def generate(self) -> np.ndarray:
        if self.feed_type == "csv" and self.csv_path:
            loader = HistoricalCSVFeed(csv_path=self.csv_path, trading_days=self.trading_days)
            return loader.generate()
        elif self.feed_type == "yfinance":
            loader = HistoricalYFinanceFeed(symbol=self.symbol, trading_days=self.trading_days)
            return loader.generate()
        else:
            loader = SyntheticUnderlyingFeed(
                start_price=self.start_price,
                trading_days=self.trading_days,
                seed=self.seed,
            )
            return loader.generate()
