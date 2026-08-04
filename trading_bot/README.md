# Options Trading Bot -- Self-Improving Starter Kit

One project, one entry point (`main.py`), three stages: **backtest → paper
trade → live trade**, plus a **self-improvement loop** that re-tunes the
strategy's parameters over time and remembers what it learned.

## Project layout

```
trading_bot/
  main.py               <- single entry point: run / optimize / history
  options_pricing.py    <- Black-Scholes pricing, Greeks, strike selection
  data_simulator.py     <- SYNTHETIC price generator (placeholder, see warning)
  backtest.py           <- strategy + backtest engine, StrategyParams dataclass
  metrics.py            <- CAGR, drawdown, win rate, Sharpe
  optimizer.py          <- walk-forward optimizer (the self-improvement engine)
  config_store.py        <- persistent state: live params + audit log
  broker_interface.py   <- PaperBroker (simulated) + KiteBroker (real orders)
  strategy_config.json  <- auto-created: current live params + full history
  requirements.txt
```

```bash
pip install -r requirements.txt
python3 main.py run          # backtest with current live params
python3 main.py optimize     # run a self-improvement cycle
python3 main.py history      # see every optimization decision ever made
```

## How the self-improvement loop actually works

Every `optimize` run:

1. Generates/loads a price series and splits it into rolling **(train,
   test)** windows.
2. Grid-searches strategy parameters (target delta, stop-loss multiple,
   profit-target %) on each window's **train** slice only.
3. Evaluates the winning combo on that window's **test** slice — data it
   never saw during the search.
4. Averages that out-of-sample (OOS) score across all windows, and does
   the same for the *current* live parameters as a baseline.
5. Only overwrites `strategy_config.json` if the candidate beats the
   baseline by more than `--min-improvement` (default: +0.15 Sharpe) —
   a deliberate margin so a difference that's probably just noise
   doesn't cause the config to churn every run.
6. Logs the decision either way to the `history` list in
   `strategy_config.json` — timestamp, both parameter sets, both scores,
   and whether it adopted the change.

Because `run` always reads whatever is currently in
`strategy_config.json`, an improvement found by `optimize` automatically
applies to every future `run` (and eventually paper/live trading) — no
code edits needed. That persistence is what makes this "self-improving"
rather than "a script I re-run and manually decide about."

**What's deliberately *not* tunable:** position size (`LOTS` in
`backtest.py`). An optimizer that's allowed to size positions will
"improve" backtest returns purely by adding leverage, which is not an
improvement — it's a great backtest and a account-blowing live account.
Position sizing stays a human risk decision.

## A real gotcha you'll hit immediately — read this

Run `optimize` on the bundled synthetic data and it'll likely find a
parameter set with an OOS Sharpe around **30**, which is absurd — real
strategies don't have Sharpe ratios anywhere near that. This isn't a bug
in the optimizer; it's the walk-forward guardrail correctly finding the
best fit to a **synthetic price path that's smoother than any real
market**. The synthetic generator (`data_simulator.py`) is pure
Geometric Brownian Motion — no overnight gaps, no volatility spikes, no
Friday-expiry chaos, no Budget Day moves. A tight stop-loss looks
flawless when the underlying never gaps past it.

This is the single most important lesson this starter kit can teach
you: **walk-forward validation protects you from overfitting to noise
in your data, but it cannot protect you from data that doesn't resemble
reality.** Garbage in, garbage out — no amount of clever validation
fixes that. Once you plug in real historical data (see below), re-run
`optimize` and expect much more modest, believable numbers. If you ever
see a Sharpe above single digits on real option data, be suspicious of
your own pipeline before you get excited.

## Stage 1: Backtest (you're here)

Works today against synthetic data. Swap `data_simulator.py`'s
`SyntheticUnderlyingFeed` for a real loader once you have:
- Real historical spot/futures prices (your broker's historical API, or
  NSE's own historical data)
- Real historical implied volatility / option chain data (the hard
  part — vendors like Sensibull or Opstra, or a market data provider,
  usually have this; most free sources don't)

Keep the same `.generate() -> np.ndarray` interface and nothing else in
the project needs to change.

## Stage 2: Paper trading

1. Implement `PaperBroker.get_ltp()` against a live quote source (most
   broker APIs expose free LTP/quote endpoints even before you're
   approved for order placement).
2. Run the strategy loop against live quotes with `PaperBroker` to test
   your *code path* (order sequencing, stop-loss triggers, position
   tracking) under real market timing, with zero financial risk.
3. Let it run for weeks, across different market conditions, before
   trusting it. Keep running `optimize` periodically against the
   growing real trade history — that's the loop continuing to work once
   you're live.

## Stage 3: Going live

- Pick a broker with a retail API (Zerodha Kite Connect, Upstox, Angel
  One SmartAPI, Fyers, Dhan are the common ones in India).
- Register your strategy under SEBI's post-April-2026 retail algo
  framework (Algo-ID / strategy registration, IP whitelisting). The
  exact process is broker-specific and has been changing — confirm
  current steps directly with your broker's API team.
- Swap `PaperBroker` for `KiteBroker` (or your broker's equivalent) only
  after Stage 2 has held up.
- Keep the stop-loss mandatory — naked short strangles have unbounded
  downside without one.

## Testing checklist before risking real capital

- [ ] Backtest run against **real** historical option data, not synthetic
- [ ] Optimizer re-run on real data and the resulting Sharpe looks
      believable (single digits, not double), not a repeat of the
      synthetic-data artifact above
- [ ] Tested across at least one high-volatility period (a budget day, a
      major election result, a global risk-off event) — strategies that
      only look good in calm markets often blow up in vol spikes
- [ ] Paper-traded live for several weeks with no code-path surprises
- [ ] Stop-loss and max-daily-loss limits confirmed to actually trigger
      (deliberately force a losing scenario in paper mode and check)
- [ ] Position sizing sized to capital you can genuinely afford to lose
- [ ] Broker's algo registration / Algo-ID requirements completed
- [ ] `history` log reviewed — does the optimizer's parameter choice
      actually make intuitive sense, or does it look like it's gaming
      something in the data?
