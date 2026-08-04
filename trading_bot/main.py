"""
Single entry point for the whole project.

  python3 main.py run                 -> backtest with current live params, print report + chart
  python3 main.py optimize            -> run the self-improvement cycle
  python3 main.py history             -> show the optimization audit log

Self-improvement, in one sentence: every `optimize` run walk-forward
tests a grid of parameters against data the grid search didn't see, and
only overwrites strategy_config.json if the new parameters actually beat
the current ones out-of-sample by a real margin. `run` always reads
whatever is currently in strategy_config.json, so improvements persist
automatically across runs without touching code.
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_simulator import SyntheticUnderlyingFeed, HistoricalUnderlyingFeed
from backtest import run_backtest, StrategyParams
from metrics import compute_metrics
from optimizer import walk_forward_optimize, evaluate_params_on_windows, _make_windows
from backtest import DAYS_PER_EXPIRY
import config_store as cfg

STARTING_CAPITAL = 200_000.0
IV_ESTIMATE = 0.14
MIN_IMPROVEMENT = 0.15   # required average-OOS-Sharpe margin to adopt new params


def get_prices(trading_days: int, seed: int = 7, source: str = "synthetic", symbol: str = "^NSEI", csv_path: str = None):
    feed = HistoricalUnderlyingFeed(
        feed_type=source,
        symbol=symbol,
        csv_path=csv_path,
        trading_days=trading_days,
        seed=seed,
        start_price=24500,
    )
    return feed.generate()


def cmd_run(args):
    state = cfg.load_state()
    params = cfg.get_live_params(state)
    prices = get_prices(
        trading_days=args.days,
        seed=args.seed,
        source=args.source,
        symbol=args.symbol,
        csv_path=args.csv,
    )

    result = run_backtest(prices, params=params, starting_capital=STARTING_CAPITAL,
                           iv_estimate=IV_ESTIMATE, strategy_type=getattr(args, 'strategy', 'short_strangle'))
    report = compute_metrics(result)

    print("=" * 55)
    print(f"BACKTEST REPORT [{getattr(args, 'strategy', 'short_strangle').upper()}] ({args.source.upper()} DATA -- {args.symbol if args.source == 'yfinance' else (args.csv or 'Synthetic')})")
    print(f"Live params: {params}")
    print("=" * 55)
    for k, v in report.items():
        print(f"{k:25s}: {v}")

    plt.figure(figsize=(10, 5))
    plt.plot(result.equity_curve, marker="o", markersize=3)
    plt.title(f"Equity Curve (per trade) -- Weekly Short Strangle, {args.source.capitalize()} Data")
    plt.xlabel("Trade number")
    plt.ylabel("Capital (Rs)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("equity_curve.png", dpi=150)
    print("\nSaved chart -> equity_curve.png")


def cmd_optimize(args):
    state = cfg.load_state()
    current_params = cfg.get_live_params(state)
    prices = get_prices(
        trading_days=args.days,
        seed=args.seed,
        source=args.source,
        symbol=args.symbol,
        csv_path=args.csv,
    )

    print(f"Running walk-forward optimization over {len(prices)} trading days [{args.source.upper()}]...")
    candidate, candidate_oos_score, window_logs = walk_forward_optimize(
        prices, train_weeks=args.train_weeks, test_weeks=args.test_weeks,
        step_weeks=args.step_weeks, starting_capital=STARTING_CAPITAL, iv_estimate=IV_ESTIMATE,
    )

    if candidate is None:
        print("No parameter combination cleared the minimum-trades / max-drawdown "
              "guardrails on this data. Config left unchanged.")
        cfg.record_optimization_run(state, current_params, float("nan"), float("nan"),
                                     improved=False, min_improvement=MIN_IMPROVEMENT,
                                     notes="No valid candidate found.")
        return

    windows = _make_windows((len(prices) - 1) // DAYS_PER_EXPIRY,
                             args.train_weeks, args.test_weeks, args.step_weeks)
    baseline_scores = evaluate_params_on_windows(prices, current_params, windows,
                                                  STARTING_CAPITAL, IV_ESTIMATE)
    valid_baseline = [s for s in baseline_scores if s != float("-inf")]
    baseline_oos_score = sum(valid_baseline) / len(valid_baseline) if valid_baseline else float("-inf")

    improved = bool(candidate.key() != current_params.key() and
                    candidate_oos_score > baseline_oos_score + args.min_improvement)

    print("-" * 55)
    print(f"Current live params : {current_params}")
    print(f"  -> avg OOS score   : {baseline_oos_score:.3f}")
    print(f"Candidate params     : {candidate}")
    print(f"  -> avg OOS score   : {candidate_oos_score:.3f}")
    print(f"Required improvement : +{args.min_improvement} (Sharpe)")
    print(f"Decision             : {'ADOPT new params' if improved else 'KEEP current params'}")
    print("-" * 55)

    notes = (f"{len(window_logs)} walk-forward windows evaluated [{args.source.upper()}] "
             f"(train={args.train_weeks}w, test={args.test_weeks}w, step={args.step_weeks}w).")
    cfg.record_optimization_run(state, candidate, baseline_oos_score, candidate_oos_score,
                                 improved=improved, min_improvement=args.min_improvement, notes=notes)

    if improved:
        print("strategy_config.json updated -- future `run` calls use the new params.")
    else:
        print("strategy_config.json unchanged.")


def cmd_history(args):
    state = cfg.load_state()
    if not state["history"]:
        print("No optimization runs recorded yet. Run: python3 main.py optimize")
        return
    print(json.dumps(state["history"], indent=2))


def cmd_ui(args):
    import webbrowser
    from server import run_server

    url = f"http://localhost:{args.port}"
    print(f"Launching Options Trading Bot Terminal UI on {url}...")
    if not args.no_browser:
        webbrowser.open(url)
    run_server(port=args.port)


def cmd_paper(args):
    from broker_interface import PaperBroker, AngelOneBroker
    from paper_trader import PaperTrader

    state = cfg.load_state()
    params = cfg.get_live_params(state)

    if args.source == "angel" and args.api_key and args.client_id and args.password and args.totp_secret:
        try:
            angel = AngelOneBroker(api_key=args.api_key, client_id=args.client_id,
                                  password=args.password, totp_secret=args.totp_secret)
            broker = PaperBroker(angel_instance=angel)
            print("[PAPER TRADING] Connected to FREE Angel One SmartAPI for live quote streaming.")
        except Exception as e:
            print(f"[PAPER TRADING] Could not connect to Angel One SmartAPI ({e}). Falling back to yfinance live quotes.")
            broker = PaperBroker(default_source="yfinance")
    elif args.source == "kite" and args.api_key and args.access_token:
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=args.api_key)
            kite.set_access_token(args.access_token)
            broker = PaperBroker(kite_instance=kite)
            print("[PAPER TRADING] Connected to Zerodha KiteConnect for live quote streaming.")
        except Exception as e:
            print(f"[PAPER TRADING] Could not connect to KiteConnect ({e}). Falling back to yfinance live quotes.")
            broker = PaperBroker(default_source="yfinance")
    else:
        broker = PaperBroker(default_source="yfinance")
        print("[PAPER TRADING] Initialized PaperBroker against 100% FREE live market quote stream (yfinance). Zero financial risk.")

    trader = PaperTrader(broker=broker, params=params, symbol=args.symbol)
    trader.run_live_loop(poll_interval_sec=args.interval, max_ticks=args.ticks)


def main():
    parser = argparse.ArgumentParser(description="Options trading bot -- backtest, paper-trade & self-improve")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Backtest with current live params")
    p_run.add_argument("--days", type=int, default=252)
    p_run.add_argument("--seed", type=int, default=7)
    p_run.add_argument("--source", type=str, choices=["synthetic", "yfinance", "csv"], default="synthetic", help="Data source: synthetic, yfinance, or csv")
    p_run.add_argument("--symbol", type=str, default="^NSEI", help="Ticker symbol for yfinance (e.g. ^NSEI, ^NSEBANK, SPY)")
    p_run.add_argument("--csv", type=str, default=None, help="Path to local historical CSV file")
    p_run.add_argument("--strategy", type=str, choices=["short_strangle", "short_straddle", "iron_condor", "iron_butterfly", "bull_put_spread", "bear_call_spread", "long_call", "long_put"], default="short_strangle", help="Option strategy choice")
    p_run.set_defaults(func=cmd_run)

    p_opt = sub.add_parser("optimize", help="Run a self-improvement (walk-forward optimization) cycle")
    p_opt.add_argument("--days", type=int, default=756)  # ~3 years, enough weeks for several windows
    p_opt.add_argument("--seed", type=int, default=11)
    p_opt.add_argument("--source", type=str, choices=["synthetic", "yfinance", "csv"], default="synthetic", help="Data source: synthetic, yfinance, or csv")
    p_opt.add_argument("--symbol", type=str, default="^NSEI", help="Ticker symbol for yfinance (e.g. ^NSEI, ^NSEBANK, SPY)")
    p_opt.add_argument("--csv", type=str, default=None, help="Path to local historical CSV file")
    p_opt.add_argument("--train-weeks", type=int, default=20, dest="train_weeks")
    p_opt.add_argument("--test-weeks", type=int, default=10, dest="test_weeks")
    p_opt.add_argument("--step-weeks", type=int, default=10, dest="step_weeks")
    p_opt.add_argument("--min-improvement", type=float, default=MIN_IMPROVEMENT, dest="min_improvement")
    p_opt.set_defaults(func=cmd_optimize)

    p_hist = sub.add_parser("history", help="Show the optimization audit log")
    p_hist.set_defaults(func=cmd_history)

    p_ui = sub.add_parser("ui", help="Launch the Web UI Dashboard")
    p_ui.add_argument("--port", type=int, default=8000, help="Port to run the web server on")
    p_ui.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser")
    p_ui.set_defaults(func=cmd_ui)

    p_paper = sub.add_parser("paper", help="Run live paper-trading strategy loop against real quotes")
    p_paper.add_argument("--symbol", type=str, default="^NSEI", help="Symbol to paper trade (e.g. ^NSEI, ^NSEBANK)")
    p_paper.add_argument("--source", type=str, choices=["yfinance", "angel", "kite"], default="yfinance", help="Live quote source: yfinance (FREE), angel (FREE), or kite")
    p_paper.add_argument("--api-key", type=str, default=None, help="API key")
    p_paper.add_argument("--client-id", type=str, default=None, help="Angel One Client ID (optional)")
    p_paper.add_argument("--password", type=str, default=None, help="Angel One Password (optional)")
    p_paper.add_argument("--totp-secret", type=str, default=None, help="Angel One TOTP secret (optional)")
    p_paper.add_argument("--access-token", type=str, default=None, help="Kite Connect Access Token (optional)")
    p_paper.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds")
    p_paper.add_argument("--ticks", type=int, default=10, help="Number of ticks to run (omit for infinite live loop)")
    p_paper.set_defaults(func=cmd_paper)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


