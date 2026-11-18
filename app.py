import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from datetime import datetime
import requests
import re

st.set_page_config(
    layout="wide",
    page_title="APEX // 1-Click Option Signals",
    page_icon="🚦",
    initial_sidebar_state="expanded"
)

# --- CSS DEFINITIVO: MODO OSCURO FORZADO Y CERO LÍNEAS BLANCAS ---
st.markdown("""
<style>
    /* Forzar fondo oscuro y colores en Streamlit Cloud */
    [data-testid="stAppViewContainer"], .stApp {
        background: radial-gradient(circle at top right, #111827, #0a0e17 65%) !important;
        color: #f8fafc !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    [data-testid="stDecoration"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #0d1321 !important;
        border-right: 1px solid #1e293b;
    }
    .terminal-header {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.12) 0%, rgba(59, 130, 246, 0.12) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 14px;
        padding: 16px 24px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .terminal-title {
        font-size: 1.55rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .terminal-sub {
        color: #94a3b8;
        font-size: 0.88rem;
        margin-top: 2px;
    }

    /* Giant Traffic-Light Hero Signal Banners */
    .signal-hero-call {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.22) 0%, rgba(6, 78, 59, 0.45) 100%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.25);
    }
    .signal-hero-put {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.22) 0%, rgba(136, 19, 55, 0.45) 100%);
        border: 2px solid #f43f5e;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 0 35px rgba(244, 63, 94, 0.25);
    }
    .signal-hero-wait {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.18) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 2px solid #f59e0b;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 0 25px rgba(245, 158, 11, 0.15);
    }
    .hero-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.82rem;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .hero-headline {
        font-size: 1.9rem;
        font-weight: 900;
        color: #ffffff !important;
        margin: 0 0 6px 0;
        letter-spacing: 0.3px;
    }
    .hero-subtext {
        font-size: 1.02rem;
        color: #e2e8f0 !important;
        margin-bottom: 18px;
    }

    /* 3-Checkmark Alignment Boxes */
    .check-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-top: 14px;
    }
    .check-box {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 14px 16px;
    }
    .check-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .check-status {
        font-size: 1.05rem;
        font-weight: 800;
        margin-top: 4px;
        color: #f8fafc !important;
    }
    .check-desc {
        font-size: 0.82rem;
        color: #cbd5e1 !important;
        margin-top: 3px;
    }

    /* Simple Action Plan Box */
    .action-plan-box {
        background: rgba(2, 6, 23, 0.82);
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 18px;
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
    }
    .action-step-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 700;
    }
    .action-step-val {
        font-size: 1.2rem;
        font-weight: 900;
        color: #ffffff !important;
        margin-top: 2px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #131c2e, #0f172a);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 12px 16px;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)

# --- BLACK-SCHOLES GREEKS FALLBACK ---
def calc_bs_greeks(S, K, T, r, sigma, opt_type):
    T = np.maximum(T, 1 / 365.0)
    sigma = np.maximum(sigma, 0.01)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if opt_type == 'CALL':
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return delta, gamma

# --- INTRADAY 5-MIN TREND DETECTOR ---
@st.cache_data(ttl=45, show_spinner=False)
def fetch_intraday_trend(symbol):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="2d", interval="5m")
        if hist.empty or len(hist) < 5:
            return "NEUTRAL", 0.0
        closes = hist['Close']
        last_price = float(closes.iloc[-1])
        ema_9 = float(closes.ewm(span=9, adjust=False).mean().iloc[-1])
        ema_21 = float(closes.ewm(span=21, adjust=False).mean().iloc[-1])
        day_open = float(hist['Open'].iloc[-min(len(hist), 20)])
        pct_move = ((last_price - day_open) / day_open) * 100.0

        if last_price > ema_9 and ema_9 >= ema_21:
            return "BULLISH", pct_move
        elif last_price < ema_9 and ema_9 <= ema_21:
            return "BEARISH", pct_move
        return "NEUTRAL", pct_move
    except Exception:
        return "NEUTRAL", 0.0

# --- DATA ENGINE 1: CBOE DIRECT EXCHANGE JSON ---
@st.cache_data(ttl=45, show_spinner=False)
def fetch_cboe_chains(symbol, min_dte, max_dte):
    cboe_sym = f"_{symbol}" if symbol in ["SPX", "VIX", "NDX", "RUT"] else symbol
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{cboe_sym}.json"
    headers = {"User-Agent": "Mozilla/5.0"}

    resp = requests.get(url, headers=headers, timeout=10)
    if resp.status_code != 200:
        return None, None, None, f"CBOE Feed Status {resp.status_code} for {symbol}."

    payload = resp.json().get("data", {})
    spot = float(payload.get("current_price", 0.0) or 0.0)
    options_list = payload.get("options", [])
    if not options_list or spot <= 0:
        return None, None, None, f"No CBOE options data returned for {symbol}."

    today = datetime.now().date()
    occ_pattern = re.compile(r'^([A-Z_]+)(\d{6})([CP])(\d{8})$')
    rows = []

    for item in options_list:
        opt_sym = str(item.get("option", ""))
        m = occ_pattern.match(opt_sym)
        if not m:
            continue
        _, yymmdd, cp_flag, strike_raw = m.groups()
        try:
            exp_dt = datetime.strptime(yymmdd, "%y%m%d").date()
        except Exception:
            continue

        dte = (exp_dt - today).days
        if dte < min_dte or (max_dte > 0 and dte > max_dte):
            continue

        strike_val = float(strike_raw) / 1000.0
        opt_type = "CALL" if cp_flag == "C" else "PUT"

        rows.append({
            "strike": strike_val,
            "Type": opt_type,
            "expiration": exp_dt.strftime("%Y-%m-%d"),
            "DTE": max(dte, 1),
            "bid": float(item.get("bid", 0.0) or 0.0),
            "ask": float(item.get("ask", 0.0) or 0.0),
            "lastPrice": float(item.get("last_trade_price", 0.0) or 0.0),
            "impliedVolatility": float(item.get("iv", 0.0) or 0.0),
            "openInterest": float(item.get("open_interest", 0.0) or 0.0),
            "volume": float(item.get("volume", 0.0) or 0.0),
            "native_delta": float(item.get("delta", np.nan)),
            "native_gamma": float(item.get("gamma", np.nan))
        })

    if not rows:
        return spot, "CBOE Direct", pd.DataFrame(), "No CBOE expirations matched your DTE filter."

    return spot, "CBOE Direct", pd.DataFrame(rows), None

# --- DATA ENGINE 2: YAHOO FINANCE FALLBACK ---
@st.cache_data(ttl=60, show_spinner=False)
def fetch_yfinance_chains(symbol, min_dte, max_dte):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="5d")
    if hist.empty:
        return None, None, None, f"No price data found for {symbol}."
    spot = float(hist['Close'].iloc[-1])

    expirations = ticker.options
    if not expirations:
        return spot, "Yahoo Finance", pd.DataFrame(), f"No option chains available for {symbol}."

    today = datetime.now().date()
    all_rows = []

    for exp in expirations:
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte < min_dte or (max_dte > 0 and dte > max_dte):
            continue
        try:
            chain = ticker.option_chain(exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls['Type'] = 'CALL'
            puts['Type'] = 'PUT'
            df_exp = pd.concat([calls, puts], ignore_index=True)
            df_exp['expiration'] = exp
            df_exp['DTE'] = max(dte, 1)
            all_rows.append(df_exp)
        except Exception:
            continue

    if not all_rows:
        return spot, "Yahoo Finance", pd.DataFrame(), "No expirations matched your DTE filter."

    return spot, "Yahoo Finance", pd.concat(all_rows, ignore_index=True), None

def enrich_chain_metrics(df, spot, r=0.045):
    df = df.copy()
    for col in ['bid', 'ask', 'impliedVolatility', 'openInterest', 'volume', 'lastPrice']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    df['mid'] = np.where((df['bid'] > 0) & (df['ask'] > 0), (df['bid'] + df['ask']) / 2.0, df['lastPrice'])
    df['spread'] = np.maximum(0.0, df['ask'] - df['bid'])
    df['spread_pct'] = np.where(df['mid'] > 0, (df['spread'] / df['mid']) * 100.0, 0.0)
    df['vol_oi_ratio'] = np.where(df['openInterest'] > 0, df['volume'] / df['openInterest'], 0.0)
    df['T'] = df['DTE'] / 365.0

    deltas, gammas = np.zeros(len(df)), np.zeros(len(df))
    for opt_t in ['CALL', 'PUT']:
        mask = df['Type'] == opt_t
        if mask.any():
            d, g = calc_bs_greeks(
                spot,
                df.loc[mask, 'strike'].values,
                df.loc[mask, 'T'].values,
                r,
                df.loc[mask, 'impliedVolatility'].values,
                opt_t
            )
            deltas[mask] = d
            gammas[mask] = g

    if 'native_delta' in df.columns:
        df['delta'] = np.where(df['native_delta'].notna() & (df['native_delta'] != 0), df['native_delta'], deltas)
    else:
        df['delta'] = deltas

    if 'native_gamma' in df.columns:
        df['gamma'] = np.where(df['native_gamma'].notna() & (df['native_gamma'] != 0), df['native_gamma'], gammas)
    else:
        df['gamma'] = gammas

    df['abs_delta'] = np.abs(df['delta']) * 100.0

    df['GEX'] = np.where(
        df['Type'] == 'CALL',
        df['openInterest'] * df['gamma'] * 100.0 * (spot ** 2) * 0.01,
        -df['openInterest'] * df['gamma'] * 100.0 * (spot ** 2) * 0.01
    )

    df['IV_pct'] = df['impliedVolatility'] * 100.0
    df['fitted_IV_pct'] = df['IV_pct']

    for (exp, opt_t), group in df.groupby(['expiration', 'Type']):
        valid = group[(group['IV_pct'] > 1.0) & (group['strike'] >= spot * 0.65) & (group['strike'] <= spot * 1.35)]
        if len(valid) >= 4:
            try:
                coeffs = np.polyfit(valid['strike'], valid['IV_pct'], deg=2)
                fitted = np.polyval(coeffs, group['strike'])
                df.loc[group.index, 'fitted_IV_pct'] = np.maximum(1.0, fitted)
            except Exception:
                pass

    df['IV_pp'] = df['IV_pct'] - df['fitted_IV_pct']

    capital_base = np.where(df['Type'] == 'CALL', spot, df['strike'])
    df['Annual_Pct'] = np.where(
        capital_base > 0,
        (df['mid'] / capital_base) * (365.0 / df['DTE']) * 100.0,
        0.0
    )
    return df

def compute_gex_key_levels(df_exp, spot):
    sub = df_exp[(df_exp['strike'] >= spot * 0.85) & (df_exp['strike'] <= spot * 1.15)].copy()
    if sub.empty:
        return spot * 1.02, spot * 0.98, spot

    calls = sub[sub['Type'] == 'CALL']
    puts = sub[sub['Type'] == 'PUT']

    call_wall = float(calls.loc[calls['GEX'].idxmax(), 'strike']) if not calls.empty else spot * 1.02
    put_wall = float(puts.loc[puts['GEX'].idxmin(), 'strike']) if not puts.empty else spot * 0.98

    gex_by_strike = sub.groupby('strike')['GEX'].sum().reset_index().sort_values('strike')
    signs = np.sign(gex_by_strike['GEX'].values)
    strikes = gex_by_strike['strike'].values
    gex_vals = gex_by_strike['GEX'].values
    zero_gamma = spot
    min_dist = float('inf')

    for i in range(len(strikes) - 1):
        if signs[i] != signs[i + 1] and (gex_vals[i + 1] - gex_vals[i]) != 0:
            flip = strikes[i] - gex_vals[i] * (strikes[i + 1] - strikes[i]) / (gex_vals[i + 1] - gex_vals[i])
            if abs(flip - spot) < min_dist:
                min_dist = abs(flip - spot)
                zero_gamma = float(flip)

    return call_wall, put_wall, zero_gamma

def build_vertical_spreads(df_exp, spot, call_wall, put_wall, spread_width=5.0):
    spreads = []
    for opt_t in ['PUT', 'CALL']:
        if opt_t == 'PUT':
            sub = df_exp[(df_exp['Type'] == 'PUT') & (df_exp['strike'] <= spot) & (df_exp['bid'] > 0.05)].sort_values('strike')
        else:
            sub = df_exp[(df_exp['Type'] == 'CALL') & (df_exp['strike'] >= spot) & (df_exp['bid'] > 0.05)].sort_values('strike')

        if sub.empty:
            continue

        candidates = sub[(sub['abs_delta'] >= 14) & (sub['abs_delta'] <= 32)]
        for _, short_row in candidates.iterrows():
            s_strike = float(short_row['strike'])
            target_long_strike = (s_strike - spread_width) if opt_t == 'PUT' else (s_strike + spread_width)

            long_matches = sub[np.isclose(sub['strike'], target_long_strike, atol=0.5)]
            if long_matches.empty:
                continue
            long_row = long_matches.iloc[0]
            l_strike = float(long_row['strike'])
            actual_width = abs(s_strike - l_strike)
            if actual_width <= 0:
                continue

            short_mid = float(short_row['mid'])
            long_mid = float(long_row['mid'])
            net_credit = short_mid - long_mid
            if net_credit <= 0.20 or net_credit >= actual_width * 0.85:
                continue

            max_risk = (actual_width - net_credit) * 100.0
            credit_dollars = net_credit * 100.0
            ror_pct = (credit_dollars / max_risk) * 100.0 if max_risk > 0 else 0.0
            pop_pct = 100.0 - float(short_row['abs_delta'])

            if opt_t == 'PUT':
                wall_status = "✅ Protected Below Floor" if s_strike <= put_wall else "⚠️ Above Floor"
            else:
                wall_status = "✅ Protected Above Ceiling" if s_strike >= call_wall else "⚠️ Below Ceiling"

            spreads.append({
                'Strategy': 'Bull Put Spread' if opt_t == 'PUT' else 'Bear Call Spread',
                'Option Type': opt_t,
                'Expiration': short_row['expiration'],
                'DTE': int(short_row['DTE']),
                'Sell Strike': s_strike,
                'Sell Leg Mid ($)': short_mid,
                'Buy Strike': l_strike,
                'Buy Leg Mid ($)': long_mid,
                'Net Spread Credit ($)': credit_dollars,
                'Max Risk ($)': max_risk,
                'Return on Risk %': ror_pct,
                'Win Prob (POP %)': pop_pct,
                'Short IV +pp': float(short_row['IV_pp']),
                'GEX Alignment': wall_status,
                '50% Profit Exit ($)': round(net_credit * 0.50, 2),
                'Stop Loss Trigger ($)': round(net_credit * 2.00, 2)
            })

    if not spreads:
        return pd.DataFrame()
    sp_df = pd.DataFrame(spreads)
    sp_df['Edge_Score'] = sp_df['Return on Risk %'] + (sp_df['Short IV +pp'] * 4.0)
    return sp_df.sort_values(by='Edge_Score', ascending=False).drop(columns=['Edge_Score']).reset_index(drop=True)

def evaluate_traffic_light_signal(play_full_df, spot, call_wall, put_wall, zero_gamma, trend_dir):
    scalps = play_full_df[
        (play_full_df['abs_delta'] >= 32) &
        (play_full_df['abs_delta'] <= 58) &
        (play_full_df['mid'] >= 0.25) &
        (play_full_df['openInterest'] >= 80) &
        (play_full_df['spread_pct'] <= 15.0)
    ].copy()

    if scalps.empty:
        scalps = play_full_df[(play_full_df['abs_delta'] >= 25) & (play_full_df['abs_delta'] <= 65) & (play_full_df['mid'] > 0.15)].copy()

    scalps['Flow_Score'] = (scalps['vol_oi_ratio'] * 3.0) - (scalps['IV_pp'] * 0.4) - (scalps['spread_pct'] * 0.25)

    calls = scalps[scalps['Type'] == 'CALL'].sort_values('Flow_Score', ascending=False)
    puts = scalps[scalps['Type'] == 'PUT'].sort_values('Flow_Score', ascending=False)

    best_call = calls.iloc[0] if not calls.empty else None
    best_put = puts.iloc[0] if not puts.empty else None

    call_flow_val = float(best_call['vol_oi_ratio']) if best_call is not None else 0.0
    put_flow_val = float(best_put['vol_oi_ratio']) if best_put is not None else 0.0

    bull_1_zone = spot >= zero_gamma
    bull_2_room = (call_wall - spot) >= (spot * 0.008)
    bull_3_flow = (call_flow_val >= put_flow_val) and (call_flow_val >= 0.35 or trend_dir == "BULLISH")
    bull_score = int(bull_1_zone) + int(bull_2_room) + int(bull_3_flow)

    bear_1_zone = spot < zero_gamma
    bear_2_room = (spot - put_wall) >= (spot * 0.008)
    bear_3_flow = (put_flow_val > call_flow_val) or (trend_dir == "BEARISH")
    bear_score = int(bear_1_zone) + int(bear_2_room) + int(bear_3_flow)

    if bull_score == 3 and best_call is not None:
        return "BUY_CALL", 3, best_call, [
            ("✅ 1. Bullish Zone", f"Price (${spot:.2f}) is ABOVE Trigger Line (${zero_gamma:.2f})"),
            ("✅ 2. Room to Ceiling", f"Clear upside room to ${call_wall:.2f} Target Ceiling"),
            ("✅ 3. Call Buying Flow", f"Whale Call Activity ({call_flow_val:.2f}x Vol/OI) leads Puts")
        ]
    elif bear_score == 3 and best_put is not None:
        return "BUY_PUT", 3, best_put, [
            ("✅ 1. Bearish Breakdown", f"Price (${spot:.2f}) is BELOW Trigger Line (${zero_gamma:.2f})"),
            ("✅ 2. Room to Floor", f"Open drop room down to ${put_wall:.2f} Support Floor"),
            ("✅ 3. Put Buying Flow", f"Heavy Put Flow ({put_flow_val:.2f}x Vol/OI) & Bearish Momentum")
        ]
    else:
        if bull_score >= bear_score and best_call is not None:
            checks = [
                (f"{'✅' if bull_1_zone else '⏳'} 1. Price vs Trigger", f"{'Above' if bull_1_zone else f'Wait for break above'} ${zero_gamma:.2f}"),
                (f"{'✅' if bull_2_room else '⚠️'} 2. Room to Ceiling", f"Target Ceiling is ${call_wall:.2f} ({((call_wall-spot)/spot)*100:+.1f}%)"),
                (f"{'✅' if bull_3_flow else '⏳'} 3. Whale Option Flow", f"Call Vol/OI is {call_flow_val:.2f}x vs Put {put_flow_val:.2f}x")
            ]
            return "WAIT_BULL", bull_score, best_call, checks
        else:
            checks = [
                (f"{'✅' if bear_1_zone else '⏳'} 1. Price vs Trigger", f"{'Below' if bear_1_zone else f'Wait for break below'} ${zero_gamma:.2f}"),
                (f"{'✅' if bear_2_room else '⚠️'} 2. Room to Floor", f"Target Floor is ${put_wall:.2f} ({((put_wall-spot)/spot)*100:+.1f}%)"),
                (f"{'✅' if bear_3_flow else '⏳'} 3. Whale Option Flow", f"Put Vol/OI is {put_flow_val:.2f}x vs Call {call_flow_val:.2f}x")
            ]
            return "WAIT_BEAR", bear_score, best_put, checks

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("## 🚦 APEX // SIMPLE MODE")
symbol = st.sidebar.text_input("1. Enter Stock Ticker", value="NVDA").upper().strip()

trade_style = st.sidebar.radio(
    "2. Pick Your Trading Timeframe",
    ["⚡ Fast Weekly Scalps (0 - 14 Days)", "🛡️ Monthly Swing Trades (15 - 45 Days)"],
    index=0
)

min_dte, max_dte = (0, 14) if trade_style.startswith("⚡") else (15, 45)
spread_width = st.sidebar.selectbox("3. Spread Protection Width ($)", [2.5, 5.0, 10.0], index=1)

with st.sidebar.expander("⚙️ Advanced Filters (Optional)", expanded=False):
    data_source = st.radio("Data Feed", ["CBOE Direct Exchange (Free)", "Yahoo Finance Backup"])
    min_oi = st.number_input("Min Open Interest", min_value=0, value=50, step=25)
    delta_range = st.slider("Delta Range (|Δ| %)", 5, 95, (10, 75))

if st.sidebar.button("🔄 Refresh Signal Now", use_container_width=True, type="primary"):
    st.cache_data.clear()

with st.spinner(f"Analyzing {symbol} signals, trend & whale flow..."):
    trend_dir, intraday_pct = fetch_intraday_trend(symbol)
    if data_source.startswith("CBOE"):
        spot, feed_lbl, raw_df, err = fetch_cboe_chains(symbol, min_dte, max_dte)
        if err or raw_df is None or raw_df.empty:
            spot, feed_lbl, raw_df, err = fetch_yfinance_chains(symbol, min_dte, max_dte)
    else:
        spot, feed_lbl, raw_df, err = fetch_yfinance_chains(symbol, min_dte, max_dte)

if err or raw_df is None or raw_df.empty:
    st.error(err or "No option data found.")
    st.stop()

full_df = enrich_chain_metrics(raw_df, spot)
available_exps = sorted(full_df['expiration'].unique())

st.markdown(f"""
<div class="terminal-header">
    <div>
        <p class="terminal-title">🚦 APEX // 1-CLICK OPTIONS SIGNAL ASSISTANT</p>
        <p class="terminal-sub">No complicated math — Wait for 3 Green Checkmarks to align, then follow the exact Trade Plan.</p>
    </div>
    <div style="background: rgba(16,185,129,0.15); border: 1px solid #10b981; color:#34d399; padding:6px 14px; border-radius:999px; font-size:0.8rem; font-weight:700;">
        ● {feed_lbl.upper()} // {symbol} ${spot:.2f}
    </div>
</div>
""", unsafe_allow_html=True)

selected_exp = st.selectbox("📅 Choose Option Expiration Date:", available_exps, index=0)
exp_df = full_df[full_df['expiration'] == selected_exp].copy()

call_wall, put_wall, zero_gamma = compute_gex_key_levels(exp_df, spot)
sig_state, sig_score, best_opt, check_items = evaluate_traffic_light_signal(
    exp_df, spot, call_wall, put_wall, zero_gamma, trend_dir
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Current Stock Price", f"${spot:.2f}", f"Intraday: {intraday_pct:+.2f}% ({trend_dir})")
k2.metric("Bull / Bear Trigger Line", f"${zero_gamma:.2f}", "Above = Calls | Below = Puts")
k3.metric("Upside Target Ceiling", f"${call_wall:.2f}", f"{((call_wall - spot)/spot)*100:+.1f}% from current price")
k4.metric("Downside Support Floor", f"${put_wall:.2f}", f"{((put_wall - spot)/spot)*100:+.1f}% from current price")

st.markdown("<br>", unsafe_allow_html=True)

if sig_state == "BUY_CALL" and best_opt is not None:
    entry_p = float(best_opt['mid'])
    tp_p = entry_p * 1.25
    sl_p = entry_p * 0.75
    st.markdown(f"""
    <div class="signal-hero-call">
        <span class="hero-badge" style="background:#10b981; color:#022c22;">🟢 3 / 3 BULLISH CHECKS ALIGNED — GO CALL</span>
        <p class="hero-headline">BUY {symbol} {selected_exp} ${best_opt['strike']:.2f} CALL</p>
        <p class="hero-subtext">
            <strong>Why now:</strong> {symbol} is trading above its <strong>${zero_gamma:.2f}</strong> Bull Trigger Line with active Call buying and clear room toward the <strong>${call_wall:.2f}</strong> Ceiling.
        </p>
        <div class="check-grid">
            <div class="check-box"><div class="check-title">CONDITION 1</div><div class="check-status">{check_items[0][0]}</div><div class="check-desc">{check_items[0][1]}</div></div>
            <div class="check-box"><div class="check-title">CONDITION 2</div><div class="check-status">{check_items[1][0]}</div><div class="check-desc">{check_items[1][1]}</div></div>
            <div class="check-box"><div class="check-title">CONDITION 3</div><div class="check-status">{check_items[2][0]}</div><div class="check-desc">{check_items[2][1]}</div></div>
        </div>
        <div class="action-plan-box">
            <div><div class="action-step-label">1. CONTRACT TO BUY</div><div class="action-step-val" style="color:#34d399;">${best_opt['strike']:.2f} CALL</div></div>
            <div><div class="action-step-label">2. PAY UP TO (MID PRICE)</div><div class="action-step-val">${entry_p:.2f} <span style="font-size:0.85rem; color:#94a3b8;">(${entry_p*100:.0f})</span></div></div>
            <div><div class="action-step-label">3. TAKE PROFIT (+25% GOAL)</div><div class="action-step-val" style="color:#38bdf8;">Sell @ ${tp_p:.2f}</div></div>
            <div><div class="action-step-label">4. HARD STOP-LOSS (-25%)</div><div class="action-step-val" style="color:#fb7185;">Exit @ ${sl_p:.2f}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif sig_state == "BUY_PUT" and best_opt is not None:
    entry_p = float(best_opt['mid'])
    tp_p = entry_p * 1.25
    sl_p = entry_p * 0.75
    st.markdown(f"""
    <div class="signal-hero-put">
        <span class="hero-badge" style="background:#f43f5e; color:#ffffff;">🔴 3 / 3 BEARISH CHECKS ALIGNED — GO PUT</span>
        <p class="hero-headline">BUY {symbol} {selected_exp} ${best_opt['strike']:.2f} PUT</p>
        <p class="hero-subtext">
            <strong>Why now:</strong> {symbol} broke below its <strong>${zero_gamma:.2f}</strong> Bear Trigger Line with heavy Put flow, opening a drop toward the <strong>${put_wall:.2f}</strong> Support Floor.
        </p>
        <div class="check-grid">
            <div class="check-box"><div class="check-title">CONDITION 1</div><div class="check-status">{check_items[0][0]}</div><div class="check-desc">{check_items[0][1]}</div></div>
            <div class="check-box"><div class="check-title">CONDITION 2</div><div class="check-status">{check_items[1][0]}</div><div class="check-desc">{check_items[1][1]}</div></div>
            <div class="check-box"><div class="check-title">CONDITION 3</div><div class="check-status">{check_items[2][0]}</div><div class="check-desc">{check_items[2][1]}</div></div>
        </div>
        <div class="action-plan-box">
            <div><div class="action-step-label">1. CONTRACT TO BUY</div><div class="action-step-val" style="color:#fb7185;">${best_opt['strike']:.2f} PUT</div></div>
            <div><div class="action-step-label">2. PAY UP TO (MID PRICE)</div><div class="action-step-val">${entry_p:.2f} <span style="font-size:0.85rem; color:#94a3b8;">(${entry_p*100:.0f})</span></div></div>
            <div><div class="action-step-label">3. TAKE PROFIT (+25% GOAL)</div><div class="action-step-val" style="color:#34d399;">Sell @ ${tp_p:.2f}</div></div>
            <div><div class="action-step-label">4. HARD STOP-LOSS (-25%)</div><div class="action-step-val" style="color:#f43f5e;">Exit @ ${sl_p:.2f}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    wait_contract = f"${best_opt['strike']:.2f} {best_opt['Type']}" if best_opt is not None else "N/A"
    entry_p = float(best_opt['mid']) if best_opt is not None else 0.0
    st.markdown(f"""
    <div class="signal-hero-wait">
        <span class="hero-badge" style="background:#f59e0b; color:#0f172a;">🟡 {sig_score} / 3 CHECKS ALIGNED — WAIT FOR BREAKOUT (OR SELL SPREAD BELOW)</span>
        <p class="hero-headline">HOLD OFF ON SINGLE CALL/PUT SCALP — {symbol} IS IN A CHOP ZONE</p>
        <p class="hero-subtext">
            <strong>What to wait for:</strong> Buy the <strong>{selected_exp} {wait_contract}</strong> (~${entry_p:.2f}) ONLY if all 3 boxes below turn green, OR collect safe sideways income using the <strong>High-Win-Rate Credit Spread</strong> card below.
        </p>
        <div class="check-grid">
            <div class="check-box"><div class="check-title">CONDITION 1</div><div class="check-status">{check_items[0][0]}</div><div class="check-desc">{check_items[0][1]}</div></div>
            <div class="check-box"><div class="check-title">CONDITION 2</div><div class="check-status">{check_items[1][0]}</div><div class="check-desc">{check_items[1][1]}</div></div>
            <div class="check-box"><div class="check-title">CONDITION 3</div><div class="check-status">{check_items[2][0]}</div><div class="check-desc">{check_items[2][1]}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

tab_simple, tab_pro_charts = st.tabs([
    "🎯 Best Trade Tickets (Simple View)",
    "📊 Pro Charts & Full Option Chain (Under the Hood)"
])

with tab_simple:
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("⚡ Top 5 Single-Contract Scalps (Buy Call / Buy Put)")
        st.caption("Ranked by Smart-Money Volume Flow & tight spreads. Buy when 3/3 checks above turn green.")
        scalp_df = exp_df[
            (exp_df['abs_delta'] >= 30) &
            (exp_df['abs_delta'] <= 60) &
            (exp_df['mid'] >= 0.20) &
            (exp_df['openInterest'] >= 50)
        ].copy()
        if not scalp_df.empty:
            scalp_df['Score'] = (scalp_df['vol_oi_ratio'] * 3.0) - (scalp_df['IV_pp'] * 0.4) - (scalp_df['spread_pct'] * 0.25)
            scalp_df['Contract'] = scalp_df['strike'].apply(lambda x: f"${x:.2f}") + " " + scalp_df['Type']
            scalp_df['Buy Price ($)'] = scalp_df['mid']
            scalp_df['Take Profit (+25%)'] = scalp_df['mid'] * 1.25
            scalp_df['Stop Loss (-25%)'] = scalp_df['mid'] * 0.75
            scalp_df['Win Chance %'] = scalp_df['abs_delta']
            scalp_df['Whale Flow (Vol/OI)'] = scalp_df['vol_oi_ratio']

            show_sc = scalp_df.sort_values('Score', ascending=False)[
                ['Contract', 'Buy Price ($)', 'Take Profit (+25%)', 'Stop Loss (-25%)', 'Win Chance %', 'Whale Flow (Vol/OI)']
            ].head(6).reset_index(drop=True)

            st.dataframe(
                show_sc.style.format({
                    'Buy Price ($)': '${:.2f}',
                    'Take Profit (+25%)': '${:.2f}',
                    'Stop Loss (-25%)': '${:.2f}',
                    'Win Chance %': '{:.0f}%',
                    'Whale Flow (Vol/OI)': '{:.2f}x'
                }),
                use_container_width=True
            )

    with col_r:
        st.subheader(f"🛡️ Top 5 High-Win-Rate Credit Spreads (${spread_width:.2f} Wide)")
        st.caption("Best when the stock is chopping sideways. You get paid upfront if the stock stays away from the Sell Strike.")
        spreads_df = build_vertical_spreads(exp_df, spot, call_wall, put_wall, spread_width=float(spread_width))
        if spreads_df.empty:
            st.info("No credit spreads matched this width. Switch Spread Protection Width in the sidebar to $2.5 or $10.0.")
        else:
            top_sp = spreads_df.iloc[0]
            st.success(
                f"**Easiest High-Probability Trade ({top_sp['Win Prob (POP %)']:.0f}% Win Rate):** "
                f"**Sell ${top_sp['Sell Strike']:.2f} {top_sp['Option Type']}** & **Buy ${top_sp['Buy Strike']:.2f} {top_sp['Option Type']}** → "
                f"Collect **${top_sp['Net Spread Credit ($)']:.0f}** upfront | Close at **${top_sp['50% Profit Exit ($)']:.2f}** for a 50% win."
            )
            show_sp = spreads_df[[
                'Strategy', 'Sell Strike', 'Buy Strike', 'Net Spread Credit ($)',
                'Max Risk ($)', 'Win Prob (POP %)', '50% Profit Exit ($)', 'GEX Alignment'
            ]].head(6)
            st.dataframe(
                show_sp.style.format({
                    'Sell Strike': '${:.2f}',
                    'Buy Strike': '${:.2f}',
                    'Net Spread Credit ($)': '${:.0f}',
                    'Max Risk ($)': '${:.0f}',
                    'Win Prob (POP %)': '{:.1f}%',
                    '50% Profit Exit ($)': '${:.2f}'
                }),
                use_container_width=True
            )

with tab_pro_charts:
    c1, c2 = st.columns(2)
    gex_sub = exp_df[(exp_df['strike'] >= spot * 0.88) & (exp_df['strike'] <= spot * 1.12)]
    gex_sum = gex_sub.groupby('strike')['GEX'].sum().reset_index()

    with c1:
        fig_gex = go.Figure()
        fig_gex.add_trace(go.Bar(
            x=gex_sum['strike'],
            y=gex_sum['GEX'],
            marker_color=np.where(gex_sum['GEX'] >= 0, '#10b981', '#f43f5e')
        ))
        fig_gex.add_vline(x=spot, line_width=2, line_dash="dot", line_color="#38bdf8", annotation_text="Spot")
        fig_gex.add_vline(x=zero_gamma, line_width=2, line_dash="dash", line_color="#f59e0b", annotation_text="Trigger")
        fig_gex.update_layout(
            title=f"Support & Resistance Walls (Net GEX) — {symbol}",
            xaxis_title="Strike Price ($)",
            yaxis_title="Dealer Gamma ($)",
            template="plotly_dark",
            height=420
        )
        st.plotly_chart(fig_gex, use_container_width=True)

    with c2:
        fig_iv = go.Figure()
        for t_name, col_c in [('CALL', '#10b981'), ('PUT', '#f43f5e')]:
            sub_t = gex_sub[gex_sub['Type'] == t_name].sort_values('strike')
            if not sub_t.empty:
                fig_iv.add_trace(go.Scatter(x=sub_t['strike'], y=sub_t['IV_pct'], mode='markers', name=f"{t_name} Actual IV", marker=dict(color=col_c, size=9)))
                fig_iv.add_trace(go.Scatter(x=sub_t['strike'], y=sub_t['fitted_IV_pct'], mode='lines', name=f"{t_name} Fair IV Curve", line=dict(dash='dash', color=col_c)))
        fig_iv.add_vline(x=spot, line_width=2, line_dash="dot", line_color="#38bdf8")
        fig_iv.update_layout(
            title=f"Volatility Pricing Curve (Dots Above Line = Rich Premium) — {selected_exp}",
            xaxis_title="Strike Price ($)",
            yaxis_title="Implied Volatility (%)",
            template="plotly_dark",
            height=420
        )
        st.plotly_chart(fig_iv, use_container_width=True)
