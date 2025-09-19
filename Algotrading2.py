# streamlit_app.py
# Simplified UI: fixed entry = MA crossover (short MA > long MA).
# Dark UI with enforced white labels. Full replacement — overwrite your app file and redeploy.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import io, traceback

# optional ta import
try:
    import ta
except Exception:
    ta = None

# ---------------- Page config & CSS ----------------
st.set_page_config(page_title="Backtest Studio (Simplified)", layout="wide", page_icon="📈")
st.markdown(
    """
    <style>
    html, body, .stApp, .block-container { background: #000000 !important; color: #FFFFFF !important; }
    .stTextInput>div>label, .stTextInput>label, .stNumberInput>div>label, .stDateInput>div>label,
    .stSelectbox>div>label, .stSlider>div>label, label { color: #FFFFFF !important; }
    input, textarea, select { color: #FFFFFF !important; background: #0b0b0b !important; border:1px solid #222 !important; }
    ::placeholder { color: #BFC7CC !important; opacity: 1; }
    .section-title { font-size:18px; font-weight:600; color:#FFFFFF; margin-bottom:6px; }
    .muted { color: #BFC7CC; font-size:13px; }
    .card { background:#0b0b0b; padding:12px; border-radius:8px; border:1px solid #222; color:#FFFFFF; }
    .stButton>button, .stDownloadButton>button { background-color:#1f2937 !important; color:#FFFFFF !important; border-radius:6px !important; }
    .stDataFrame table thead th { color: #FFFFFF !important; background: #0f1720 !important; }
    pre, code { color: #E6E6E6 !important; background: #0b0b0b !important; }
    </style>
    """,
    unsafe_allow_html=True
)
plt.style.use('dark_background')

st.title("📈 Backtest Studio — Simplified (MA crossover)")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Run")
    run = st.button("🚀 Run Backtest")
    st.write("---")
    if st.button("Env Test"):
        modules = ["streamlit","pandas","numpy","yfinance","matplotlib"]
        env = {}
        for m in modules:
            try:
                __import__(m)
                env[m] = "ok"
            except Exception as e:
                env[m] = f"error: {e}"
        st.json(env)
    st.markdown("<div class='muted'>This simplified build uses MA crossover (Short MA > Long MA) as the only entry rule.</div>", unsafe_allow_html=True)

# ---------------- Main inputs ----------------
col_left, col_right = st.columns([3,1])
with col_left:
    st.markdown('<div class="section-title">General</div>', unsafe_allow_html=True)
    raw_ticker = st.text_input("Ticker / Symbol (e.g. ACC, RELIANCE, NIFTY)", value="ACC")
    start_date = st.date_input("Start Date", value=dt.date(2021,9,30))
    end_date = st.date_input("End Date", value=dt.date.today())
    initial_capital = st.number_input("Initial Capital", value=100000.0, format="%.2f")
    capital_alloc_pct = st.slider("Capital Allocation per Trade (%)", 1, 100, 10) / 100.0
    max_open_positions = st.number_input("Max Open Positions", value=3, min_value=1)
    max_hold_days = st.number_input("Max Hold Days", value=10, min_value=1)

    st.markdown('<div class="section-title">Entry (fixed: MA crossover)</div>', unsafe_allow_html=True)
    ma_short = st.number_input("Short MA window", value=20, min_value=1)
    ma_long = st.number_input("Long MA window", value=50, min_value=1)
    st.markdown('<div class="muted">Entry rule: buy when Short MA &gt; Long MA at daily open.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Exit & Risk</div>', unsafe_allow_html=True)
    profit_target = st.number_input("Profit Target (%)", value=5.0) / 100.0
    per_trade_stop = st.number_input("Per-Trade Stop Loss (%)", value=2.0) / 100.0
    monthly_target_pct = st.number_input("Monthly Profit Target (%)", value=5.0) / 100.0
    max_overall_drawdown_pct = st.number_input("Max Overall Drawdown (%)", value=5.0) / 100.0
    max_consecutive_losses = st.number_input("Max Consecutive Losses", value=5, min_value=1)

    st.markdown('<div class="section-title">Sizing & Costs</div>', unsafe_allow_html=True)
    sizing_method = st.selectbox("Sizing Method", ['% of Capital', 'Fixed shares', 'Volatility (ATR)'])
    fixed_shares = st.number_input("Fixed shares (if used)", value=0, min_value=0)
    brokerage = st.number_input("Brokerage per trade (flat)", value=0.0, format="%.2f")
    slippage_pct = st.number_input("Slippage (%)", value=0.05) / 100.0
    exchange_fee_pct = st.number_input("Exchange fee (%)", value=0.01) / 100.0
    taxes_pct = st.number_input("Taxes (%)", value=0.0) / 100.0

with col_right:
    st.markdown('<div class="section-title">Quick Info</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">Simplified: MA crossover entry only. Set parameters and click Run.</div>', unsafe_allow_html=True)
    preview = st.empty()

# ---------------- Helper functions ----------------
def normalize_ticker(t: str) -> str:
    if not t or not str(t).strip():
        raise ValueError("Ticker required")
    s = t.strip().upper()
    if s in ("NIFTY","NIFTY50"): return "^NSEI"
    if s in ("SENSEX","BSE","BSESN"): return "^BSESN"
    if "." in t or t.startswith("^"): return t.strip()
    return s + ".NS"

@st.cache_data(ttl=3600)
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        raise RuntimeError("No data returned for ticker/date range.")
    if 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    if 'Open' not in df.columns:
        raise RuntimeError("Downloaded data missing 'Open' column.")
    df.index = pd.to_datetime(df.index)
    return df

def compute_max_drawdown(series: pd.Series) -> float:
    if series.empty: return 0.0
    peak = series.cummax()
    dd = (peak - series) / peak
    return float(dd.max())

def to_scalar(x):
    if x is None:
        return np.nan
    if isinstance(x, (pd.Series, pd.Index)):
        if len(x) == 0:
            return np.nan
        try:
            return float(x.iloc[-1])
        except Exception:
            return float(x.values[-1])
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return np.nan
        return float(x.ravel()[-1])
    try:
        if pd.isna(x):
            return np.nan
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return np.nan

# ---------------- Backtest engine (MA crossover only) ----------------
def run_backtest_engine(cfg):
    try:
        df = fetch_data(cfg['ticker_norm'], cfg['start'], cfg['end'])
    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}, None, None

    df = df.copy().dropna()
    # compute MAs
    df['MA_short'] = df['Close'].rolling(int(cfg['ma_short'])).mean()
    df['MA_long'] = df['Close'].rolling(int(cfg['ma_long'])).mean()
    # ATR
    if ta is not None and set(['High', 'Low']).issubset(df.columns):
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    else:
        df['ATR'] = np.nan

    # state
    cash = float(cfg['initial_capital'])
    positions = []
    equity_points = []
    trades = []
    monthly_realized = {}
    month_start_equity = {}
    consecutive_losses = 0
    allowed_min = cfg['initial_capital'] * (1.0 - cfg['max_overall_drawdown_pct'])

    for idx, row in df.iterrows():
        today = idx.date()
        month_key = str(idx.to_period('M'))

        if month_key not in monthly_realized:
            monthly_realized[month_key] = 0.0
            month_start_equity[month_key] = cash + sum([p['shares'] * row['Close'] for p in positions])

        # intraday low/high
        intr_low = to_scalar(row.get('Low', np.nan))
        if np.isnan(intr_low):
            intr_low = min(to_scalar(row.get('Open', np.nan)), to_scalar(row.get('Close', np.nan)))
        intr_high = to_scalar(row.get('High', np.nan))
        if np.isnan(intr_high):
            intr_high = max(to_scalar(row.get('Open', np.nan)), to_scalar(row.get('Close', np.nan)))

        # check existing positions
        remaining = []
        for pos in positions:
            stop_price = to_scalar(pos.get('stop_price', np.nan))
            target_price = to_scalar(pos.get('target_price', np.nan))
            entry_price = to_scalar(pos.get('entry_price', np.nan))
            shares = int(pos.get('shares', 0))

            exit_price = None; reason = None
            if (not np.isnan(stop_price)) and (not np.isnan(intr_low)) and (intr_low <= stop_price):
                exit_price = stop_price; reason = 'stop'
            elif (not np.isnan(target_price)) and (not np.isnan(intr_high)) and (intr_high >= target_price):
                exit_price = target_price; reason = 'target'
            else:
                hold_days = (today - pos['entry_date']).days
                if hold_days >= cfg['max_hold_days']:
                    exit_price = to_scalar(row.get('Close', np.nan)); reason = 'time_exit'

            if exit_price is not None:
                pnl = (exit_price - entry_price) * shares
                cost_exit = cfg['brokerage'] + abs(exit_price * shares) * (cfg['slippage_pct'] + cfg['exchange_fee_pct'] + cfg['taxes_pct'])
                pnl_net = pnl - cost_exit
                cash += shares * exit_price - cost_exit
                trades.append({'entry_date': pos['entry_date'].isoformat(),
                               'exit_date': today.isoformat(),
                               'entry_price': entry_price,
                               'exit_price': exit_price,
                               'shares': shares,
                               'pnl': pnl_net,
                               'reason': reason})
                monthly_realized[month_key] += pnl_net
                consecutive_losses = consecutive_losses + 1 if pnl_net < 0 else 0
            else:
                remaining.append(pos)
        positions = remaining

        # equity snapshot
        equity_now = cash + sum([p['shares'] * row['Close'] for p in positions]) if len(positions) > 0 else cash
        equity_points.append({'date': idx, 'equity': equity_now})

        # monthly pause
        paused = monthly_realized[month_key] >= (month_start_equity[month_key] * cfg['monthly_target_pct'])

        # emergency close
        worst_equity = cash + sum([ (p['stop_price'] if (p['stop_price'] is not None and p['stop_price'] < intr_low) else to_scalar(row['Close'])) * p['shares'] for p in positions ]) if len(positions)>0 else cash
        if worst_equity < allowed_min:
            for pos in positions:
                exit_price = to_scalar(row.get('Close', np.nan))
                entry_price = to_scalar(pos['entry_price'])
                shares = int(pos['shares'])
                pnl = (exit_price - entry_price) * shares
                cost_exit = cfg['brokerage'] + abs(exit_price * shares) * (cfg['slippage_pct'] + cfg['exchange_fee_pct'] + cfg['taxes_pct'])
                pnl_net = pnl - cost_exit
                cash += shares * exit_price - cost_exit
                trades.append({'entry_date': pos['entry_date'].isoformat(),
                               'exit_date': today.isoformat(),
                               'entry_price': entry_price,
                               'exit_price': exit_price,
                               'shares': shares,
                               'pnl': pnl_net,
                               'reason': 'emergency_close'})
            positions = []
            equity_now = cash
            if equity_now < allowed_min:
                break

        # ENTRY: MA crossover only
        if (not paused) and len(positions) < cfg['max_open_positions'] and consecutive_losses < cfg['max_consecutive_losses']:
            ms = to_scalar(row.get('MA_short', np.nan))
            ml = to_scalar(row.get('MA_long', np.nan))
            if (not np.isnan(ms)) and (not np.isnan(ml)) and (ms > ml):
                entry_price = to_scalar(row.get('Open', np.nan))
                if not np.isnan(entry_price) and entry_price > 0:
                    # sizing
                    if cfg['sizing_method'] == '% of Capital':
                        cap_for_trade = cfg['capital_alloc_pct'] * (cash if not cfg['reinvest_profits'] else (cash + sum([p['shares']*row['Close'] for p in positions])))
                        stop_price = entry_price * (1.0 - cfg['per_trade_stop'])
                        shares_calc = cap_for_trade // entry_price if entry_price>0 else 0
                        shares = int(to_scalar(shares_calc))
                    elif cfg['sizing_method'] == 'Fixed shares':
                        shares = int(to_scalar(cfg['fixed_shares'])); stop_price = entry_price * (1.0 - cfg['per_trade_stop'])
                    elif cfg['sizing_method'] == 'Volatility (ATR)':
                        atr_val = to_scalar(row.get('ATR', np.nan))
                        stop_price = entry_price - max(atr_val if not np.isnan(atr_val) else 0.0, entry_price*cfg['per_trade_stop'])
                        cap_for_trade = cfg['capital_alloc_pct'] * (cash if not cfg['reinvest_profits'] else (cash + sum([p['shares']*row['Close'] for p in positions])))
                        shares_calc = cap_for_trade // entry_price if entry_price>0 else 0
                        shares = int(to_scalar(shares_calc))
                    else:
                        shares = 0; stop_price = entry_price * (1.0 - cfg['per_trade_stop'])

                    shares_s = int(to_scalar(shares)) if not np.isnan(to_scalar(shares)) else 0
                    entry_price_s = to_scalar(entry_price)
                    stop_price_s = to_scalar(stop_price)

                    if shares_s > 0 and (entry_price_s * shares_s) <= float(cash):
                        hypothetical_cash = float(cash) - (shares_s * entry_price_s)
                        worst_if_stop = hypothetical_cash + (shares_s * stop_price_s) + sum([p['shares'] * row['Close'] for p in positions]) if len(positions)>0 else hypothetical_cash + (shares_s * stop_price_s)
                        if worst_if_stop < allowed_min:
                            trades.append({'entry_date': today.isoformat(), 'exit_date': None, 'entry_price': entry_price_s, 'exit_price': None, 'shares': 0, 'pnl': 0.0, 'reason': 'entry_skipped_breach_drawdown'})
                        else:
                            cost_entry = cfg['brokerage'] + abs(entry_price_s * shares_s) * (cfg['slippage_pct'] + cfg['exchange_fee_pct'] + cfg['taxes_pct'])
                            cash -= (shares_s * entry_price_s) + cost_entry
                            positions.append({'entry_date': today,
                                              'entry_price': float(entry_price_s),
                                              'shares': int(shares_s),
                                              'stop_price': float(stop_price_s),
                                              'target_price': float(entry_price_s * (1 + cfg['profit_target']))})
                            trades.append({'entry_date': today.isoformat(), 'exit_date': None, 'entry_price': float(entry_price_s), 'exit_price': None, 'shares': int(shares_s), 'pnl': None, 'reason': 'entry'})

    # finalize
    eq_df = pd.DataFrame(equity_points)
    if not eq_df.empty:
        eq_df = eq_df.set_index('date')
        equity_series = eq_df['equity']
    else:
        equity_series = pd.Series([cfg['initial_capital']])

    final_equity = float(equity_series.iloc[-1])
    total_return = (final_equity / cfg['initial_capital'] - 1.0) * 100.0
    max_dd = compute_max_drawdown(equity_series)
    try:
        days = (pd.to_datetime(cfg['end']) - pd.to_datetime(cfg['start'])).days or 1
        years = days / 365.25
        cagr_val = ((equity_series.iloc[-1] / equity_series.iloc[0]) ** (1 / years) - 1.0) * 100.0 if years > 0 else 0.0
    except Exception:
        cagr_val = 0.0
    daily_ret = equity_series.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std(ddof=1) * np.sqrt(252)) if not daily_ret.empty and daily_ret.std(ddof=1) != 0 else 0.0

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df['pnl'] > 0] if not trades_df.empty else pd.DataFrame()
    losses = trades_df[trades_df['pnl'] < 0] if not trades_df.empty else pd.DataFrame()
    win_pct = (len(wins) / len(trades_df) * 100.0) if len(trades_df) > 0 else 0.0
    avg_win = wins['pnl'].mean() if not wins.empty else 0.0
    avg_loss = losses['pnl'].mean() if not losses.empty else 0.0
    profit_factor = (wins['pnl'].sum() / abs(losses['pnl'].sum())) if (not losses.empty and losses['pnl'].sum() != 0) else np.nan

    report = {
        'final_equity': final_equity,
        'total_return_pct': total_return,
        'max_drawdown_pct': max_dd,
        'cagr_pct': cagr_val,
        'sharpe': sharpe,
        'win_pct': win_pct,
        'avg_win': float(avg_win) if not pd.isna(avg_win) else 0.0,
        'avg_loss': float(avg_loss) if not pd.isna(avg_loss) else 0.0,
        'profit_factor': float(profit_factor) if not pd.isna(profit_factor) else None,
        'equity_df': eq_df.reset_index() if eq_df is not None else pd.DataFrame(),
        'trades_df': trades_df
    }
    return report, report['equity_df'], report['trades_df']

# ---------------- Run handler ----------------
if run:
    try:
        ticker_norm = normalize_ticker(raw_ticker)
    except Exception as e:
        st.error(f"Ticker normalization error: {e}")
        st.stop()

    cfg = {
        'ticker_norm': ticker_norm,
        'start': str(start_date),
        'end': str(end_date),
        'initial_capital': float(initial_capital),
        'capital_alloc_pct': float(capital_alloc_pct),
        'max_open_positions': int(max_open_positions),
        'max_hold_days': int(max_hold_days),
        # MA crossover params
        'ma_short': int(ma_short),
        'ma_long': int(ma_long),
        # risk & exit
        'profit_target': float(profit_target),
        'per_trade_stop': float(per_trade_stop),
        'monthly_target_pct': float(monthly_target_pct),
        'max_overall_drawdown_pct': float(max_overall_drawdown_pct),
        'max_consecutive_losses': int(max_consecutive_losses),
        # sizing & costs
        'sizing_method': sizing_method,
        'fixed_shares': int(fixed_shares),
        'brokerage': float(brokerage),
        'slippage_pct': float(slippage_pct),
        'exchange_fee_pct': float(exchange_fee_pct),
        'taxes_pct': float(taxes_pct),
        'reinvest_profits': True,
        'capital_alloc_pct': float(capital_alloc_pct),
    }

    st.success(f"Running MA crossover backtest for {ticker_norm} ...")
    with st.spinner("Running..."):
        result, eq_df, trades_df = run_backtest_engine(cfg)

    if isinstance(result, dict) and result.get('error'):
        st.error("Backtest error (details):")
        st.text(result.get('error'))
        st.text(result.get('traceback'))
    else:
        st.markdown("### Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final equity", f"{result['final_equity']:.2f}")
        c2.metric("Total return %", f"{result['total_return_pct']:.2f}%")
        c3.metric("Max drawdown", f"{result['max_drawdown_pct']:.2%}")
        c4.metric("CAGR %", f"{result['cagr_pct']:.2f}%")
        st.write(f"Sharpe: {result['sharpe']:.3f}  Win%: {result['win_pct']:.2f}  Profit Factor: {result['profit_factor']}")

        if eq_df is not None and not eq_df.empty:
            fig, ax = plt.subplots(figsize=(10,4))
            ax.plot(pd.to_datetime(eq_df['date']), eq_df['equity'], label='Equity', linewidth=2, color='#00E5FF')
            ax.axhline(cfg['initial_capital']*(1-cfg['max_overall_drawdown_pct']), color='red', linestyle='--', label='Allowed min equity')
            ax.set_title(f"Equity Curve: {ticker_norm}", color='#FFFFFF')
            ax.set_xlabel("Date", color='#FFFFFF'); ax.set_ylabel("Equity", color='#FFFFFF')
            ax.tick_params(colors='#FFFFFF')
            ax.legend(facecolor='#0b0b0b', edgecolor='#222', labelcolor='#FFFFFF')
            st.pyplot(fig)

        st.markdown("### Trades")
        if trades_df is None or trades_df.empty:
            st.write("No trades executed.")
        else:
            st.dataframe(trades_df)
            buf = io.StringIO()
            trades_df.to_csv(buf, index=False)
            st.download_button("Download trades.csv", buf.getvalue().encode(), file_name="trades.csv")

        if eq_df is not None and not eq_df.empty:
            buf2 = io.StringIO()
            eq_df.to_csv(buf2, index=False)
            st.download_button("Download equity_curve.csv", buf2.getvalue().encode(), file_name="equity_curve.csv")

    preview.write(f"Ticker: **{ticker_norm}** | rows: {len(eq_df) if eq_df is not None else 0}")



