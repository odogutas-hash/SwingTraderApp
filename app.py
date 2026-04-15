# -*- coding: utf-8 -*-
# SwingTrader Screener — v3.0
# Skor: RSI+StochRSI(20) + HacimYön(25) + Fib+Squeeze(15) + MACD(15) + RS_SPY(15) + ADX(10) = 100

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io, requests, json, os, datetime

st.set_page_config(page_title="SwingTrader Screener", page_icon="🎯",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""<style>
div[data-testid="stDataFrame"] div[role="gridcell"]         { text-align: center !important; justify-content: center !important; }
div[data-testid="stDataFrame"] div[role="columnheader"] div { text-align: center !important; justify-content: center !important; }
@media (max-width: 768px) {
    .block-container { padding: 1rem 0.5rem !important; }
    div[data-testid="stMetric"] { font-size: 0.75rem !important; }
}
</style>""", unsafe_allow_html=True)

# ── Dosya yolları ─────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
WATCHLIST_FILE = os.path.join(BASE_DIR, 'watchlist.json')
HISTORY_FILE   = os.path.join(BASE_DIR, 'score_history.csv')

# ── Watchlist ─────────────────────────────────────────────────────────────────
def load_watchlist():
    try:
        with open(WATCHLIST_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return []

def save_watchlist(wl):
    with open(WATCHLIST_FILE, 'w') as f:
        json.dump(sorted(set(wl)), f)

def wl_remove(ticker):
    wl = [x for x in st.session_state['watchlist'] if x != ticker]
    st.session_state['watchlist'] = wl
    save_watchlist(wl)

# ── Skor geçmişi ──────────────────────────────────────────────────────────────
def save_score_history(results):
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    df_new = pd.DataFrame([{
        'Tarih': now, 'Hisse': r['Hisse'], 'Skor': r['Skor'],
        'RSI': r['RSI'], 'RS_SPY': r['RS vs SPY'],
    } for r in results])
    if os.path.exists(HISTORY_FILE):
        try:
            df_new = pd.concat([pd.read_csv(HISTORY_FILE), df_new], ignore_index=True)
        except Exception:
            pass
    df_new['Tarih'] = pd.to_datetime(df_new['Tarih'])
    cutoff = datetime.datetime.now() - datetime.timedelta(days=30)
    df_new[df_new['Tarih'] >= cutoff].to_csv(HISTORY_FILE, index=False)

def get_ticker_history(ticker):
    try:
        df = pd.read_csv(HISTORY_FILE)
        df['Tarih'] = pd.to_datetime(df['Tarih'])
        return df[df['Hisse'] == ticker].sort_values('Tarih').tail(30)
    except Exception:
        return pd.DataFrame()

# ── Session state ─────────────────────────────────────────────────────────────
if 'filtered'  not in st.session_state: st.session_state['filtered']  = []
if 'watchlist' not in st.session_state: st.session_state['watchlist'] = load_watchlist()


# ── Katman 1: Veri ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_sp_list(choice):
    urls = {
        "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "S&P 400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        "S&P 600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
    }
    response = requests.get(urls[choice], headers={'User-Agent': 'Mozilla/5.0'})
    df       = pd.read_html(io.StringIO(response.text))[0]
    t_col = next((c for c in ['Symbol', 'Ticker symbol', 'Ticker'] if c in df.columns), None)
    n_col = 'Security' if 'Security' in df.columns else 'Company'
    s_col = next((c for c in ['GICS Sector', 'Sector'] if c in df.columns), None)
    df[t_col] = df[t_col].astype(str).str.replace('.', '-', regex=False)
    sector_tr = {
        "Communication Services": "İletişim",   "Consumer Discretionary": "Tüketim (Lüks)",
        "Consumer Staples": "Temel Tüketim",     "Energy": "Enerji",
        "Financials": "Finans",                  "Health Care": "Sağlık",
        "Industrials": "Sanayi",                 "Information Technology": "Teknoloji",
        "Materials": "Hammadde",                 "Real Estate": "Gayrimenkul",
        "Utilities": "Kamu Hizmetleri",
    }
    return {
        row[t_col]: {
            'Security': row[n_col],
            'Sector':   sector_tr.get(row.get(s_col, ''), row.get(s_col, '') or 'Bilinmiyor'),
        }
        for _, row in df.iterrows()
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(tickers, period, interval):
    dl_iv = "1h" if interval == "4h" else interval
    raw   = yf.download(list(tickers), period=period, interval=dl_iv,
                        threads=True, progress=False, auto_adjust=True)
    if raw.empty:
        return raw
    if interval == "4h":
        raw = pd.concat({
            'Open':   raw['Open'].resample("4h").first(),
            'High':   raw['High'].resample("4h").max(),
            'Low':    raw['Low'].resample("4h").min(),
            'Close':  raw['Close'].resample("4h").last(),
            'Volume': raw['Volume'].resample("4h").sum(),
        }, axis=1).dropna(how='all')
    return raw


@st.cache_data(ttl=300, show_spinner=False)
def get_spy_return(period):
    spy = yf.download('SPY', period=period, interval='1d', progress=False, auto_adjust=True)
    if spy.empty or len(spy) < 20:
        return 0.0
    close = spy['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return float((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] * 100)


# ── Katman 2+3: İndikatörler & Skor ──────────────────────────────────────────
def compute_indicators(df):
    # True Range
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low']  - df['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    df['ATR']     = tr.rolling(14).mean()
    df['ATR_pct'] = df['ATR'] / df['Close'] * 100

    # RSI(14)
    delta       = df['Close'].diff()
    gain        = delta.where(delta > 0, 0).rolling(14).mean()
    loss        = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI']   = 100 - (100 / (1 + gain / loss))

    # Stochastic RSI(14,3)
    rsi_min          = df['RSI'].rolling(14).min()
    rsi_max          = df['RSI'].rolling(14).max()
    df['StochRSI_K'] = (df['RSI'] - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    df['StochRSI_D'] = df['StochRSI_K'].rolling(3).mean()

    # MACD(12,26,9)
    df['MACD']   = (df['Close'].ewm(span=12, adjust=False).mean() -
                    df['Close'].ewm(span=26, adjust=False).mean())
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist']   = df['MACD'] - df['Signal']

    # Bollinger Bands(20,2)
    df['BB_Mid']   = df['Close'].rolling(20).mean()
    df['BB_Upper'] = df['BB_Mid'] + df['Close'].rolling(20).std() * 2
    df['BB_Lower'] = df['BB_Mid'] - df['Close'].rolling(20).std() * 2

    # Keltner Channel(20, ATR×1.5) — Squeeze tespiti için
    df['KC_Mid']   = df['Close'].ewm(span=20, adjust=False).mean()
    df['KC_Upper'] = df['KC_Mid'] + df['ATR'] * 1.5
    df['KC_Lower'] = df['KC_Mid'] - df['ATR'] * 1.5
    df['Squeeze']  = (df['BB_Upper'] <= df['KC_Upper']) & (df['BB_Lower'] >= df['KC_Lower'])

    # ADX(14) + DI+/DI-
    dm_plus  = df['High'].diff().clip(lower=0)
    dm_minus = (-df['Low'].diff()).clip(lower=0)
    dm_plus  = dm_plus.where(dm_plus >= dm_minus, 0)
    dm_minus = dm_minus.where(dm_minus > dm_plus, 0)
    tr14 = tr.rolling(14).sum()
    df['DI_plus']  = dm_plus.rolling(14).sum() / (tr14 + 1e-10) * 100
    df['DI_minus'] = dm_minus.rolling(14).sum() / (tr14 + 1e-10) * 100
    dx = ((df['DI_plus'] - df['DI_minus']).abs() /
          (df['DI_plus'] + df['DI_minus'] + 1e-10)) * 100
    df['ADX'] = dx.rolling(14).mean()

    # OBV (On-Balance Volume)
    direction    = df['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df['OBV']    = (df['Volume'] * direction).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(20).mean()

    df['SMA50']    = df['Close'].rolling(50).mean()
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    return df


def compute_score(df, last, prev, rs_vs_spy=0.0):
    # 1. RSI + Stochastic RSI (0-20)
    rsi     = last['RSI']
    stoch_k = last['StochRSI_K'] if pd.notna(last['StochRSI_K']) else 50
    if   rsi < 30 and stoch_k < 20: rsi_s = 20   # çift onay: aşırı satım
    elif rsi < 30:                   rsi_s = 15
    elif rsi < 40 and stoch_k < 30: rsi_s = 12
    elif rsi < 40:                   rsi_s = 8
    elif rsi < 50:                   rsi_s = 4
    else:                            rsi_s = 0

    # 2. Hacim Yönü (0-25) — yeşil mum + spike = güçlü alım sinyali
    vol_ratio      = last['Volume'] / last['Vol_MA20'] if last['Vol_MA20'] > 0 else 0
    bullish_candle = last['Close'] > last['Open']
    if   bullish_candle and vol_ratio >= 3.0: vol_s = 25
    elif bullish_candle and vol_ratio >= 2.0: vol_s = 18
    elif bullish_candle and vol_ratio >= 1.5: vol_s = 12
    elif vol_ratio >= 2.0:                    vol_s = 8
    elif vol_ratio >= 1.5:                    vol_s = 4
    else:                                     vol_s = 0

    # 3. Fibonacci + Squeeze (0-15)
    rh, rl      = df['High'].max(), df['Low'].min()
    fib_levels  = [rh - (rh - rl) * r for r in [0.236, 0.382, 0.500, 0.618, 0.786]]
    min_fib_pct = min(abs(last['Close'] - f) / f for f in fib_levels) * 100
    squeeze_on  = bool(last['Squeeze']) if pd.notna(last['Squeeze']) else False
    if   min_fib_pct < 1.0 and squeeze_on: fib_s = 15
    elif min_fib_pct < 1.0:                fib_s = 12
    elif min_fib_pct < 2.0:                fib_s = 8
    elif min_fib_pct < 3.0:                fib_s = 5
    elif min_fib_pct < 5.0:                fib_s = 2
    else:                                  fib_s = 0

    # 4. MACD Momentum (0-15)
    macd_s = 15 if (last['Hist'] > 0 and last['Hist'] > prev['Hist']) else 10 if last['Hist'] > prev['Hist'] else 0

    # 5. RS vs SPY (0-15) — piyasadan güçlü hisseler bonus alır
    if   rs_vs_spy > 5.0: rs_s = 15
    elif rs_vs_spy > 2.0: rs_s = 10
    elif rs_vs_spy > 0.0: rs_s = 5
    else:                 rs_s = 0

    # 6. ADX Trend Gücü (0-10) — güçlü trend yönünü doğrular
    adx      = last['ADX']      if pd.notna(last['ADX'])      else 0
    di_plus  = last['DI_plus']  if pd.notna(last['DI_plus'])  else 0
    di_minus = last['DI_minus'] if pd.notna(last['DI_minus']) else 0
    if   adx > 30 and di_plus > di_minus: adx_s = 10
    elif adx > 20:                        adx_s = 5
    else:                                 adx_s = 0

    detail = {'rsi': rsi_s, 'vol': vol_s, 'fib': fib_s,
              'macd': macd_s, 'rs': rs_s, 'adx': adx_s}
    return rsi_s + vol_s + fib_s + macd_s + rs_s + adx_s, vol_ratio, detail


@st.cache_data(ttl=300, show_spinner=False)
def run_screen(_raw, tickers, _sp_info, rsi_lim, spy_ret_20d, data_shape, sma_filter, sector_filter):
    results, valid = [], 0
    sector_set = set(sector_filter)

    for t in tickers:
        try:
            if sector_set and _sp_info[t]['Sector'] not in sector_set:
                continue

            df = pd.DataFrame({
                'Open': _raw['Open'][t], 'High': _raw['High'][t],
                'Low':  _raw['Low'][t],  'Close': _raw['Close'][t],
                'Volume': _raw['Volume'][t],
            }).dropna()
            if len(df) < 60:
                continue
            valid += 1

            df = compute_indicators(df).dropna()
            if len(df) < 2:
                continue

            last, prev = df.iloc[-1], df.iloc[-2]

            if last['RSI'] > rsi_lim:                        continue
            if sma_filter and last['Close'] < last['SMA50']: continue
            if last['Volume'] < 50_000:                      continue

            pct       = (last['Close'] - prev['Close']) / prev['Close'] * 100
            ret_20d   = (last['Close'] - df['Close'].iloc[-20]) / df['Close'].iloc[-20] * 100
            rs_vs_spy = ret_20d - spy_ret_20d
            dist_52w  = (last['Close'] - df['High'].tail(252).max()) / df['High'].tail(252).max() * 100

            score, vol_ratio, detail = compute_score(df, last, prev, rs_vs_spy)
            pot = "🔥 Çok Yüksek" if score >= 75 else "⬆️ Yüksek" if score >= 55 else "➡️ Orta" if score >= 35 else "⬇️ Düşük"

            results.append({
                'Hisse':        t,
                'Şirket':       _sp_info[t]['Security'],
                'Sektör':       _sp_info[t]['Sector'],
                'Fiyat ($)':    round(last['Close'], 2),
                'Değişim (%)':  round(pct, 2),
                'RSI':          round(last['RSI'], 1),
                'StochRSI':     round(last['StochRSI_K'], 1) if pd.notna(last['StochRSI_K']) else 0,
                'Hacim (×avg)': round(vol_ratio, 1),
                'RS vs SPY':    round(rs_vs_spy, 1),
                '52H Uzaklık':  round(dist_52w, 1),
                'ADX':          round(last['ADX'], 1) if pd.notna(last['ADX']) else 0,
                'Squeeze':      '🟡' if last['Squeeze'] else '—',
                'Skor':         score,
                'Potansiyel':   pot,
                '_detail':      detail,
            })
        except Exception:
            continue

    results.sort(key=lambda x: x['Skor'], reverse=True)
    return results[:20], valid


# ── Katman 4: Gösterim ────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎯 SwingTrader")
    st.caption("Patlama Potansiyeli Screener")
    st.divider()

    pool     = st.selectbox("📊 Hisse Havuzu", ["S&P 500", "S&P 400", "S&P 600"], index=0)
    interval = st.selectbox("⏱ Zaman Dilimi", ["1d", "1h", "4h"], index=0)

    st.divider()
    st.subheader("🔧 Filtre Ayarları")
    rsi_lim    = st.slider("Maks. RSI", 20, 80, 55)
    sma_filter = st.toggle("📈 Yalnız SMA50 üzeri", value=False,
                           help="Kapalı = her piyasa koşulunda tarama yapar")
    auto_ref   = st.selectbox("⏰ Otomatik Yenileme",
                              ["Kapalı", "1 dk", "3 dk", "5 dk", "10 dk"], index=3)
    ref_secs = {"Kapalı": 0, "1 dk": 60, "3 dk": 180, "5 dk": 300, "10 dk": 600}[auto_ref]
    if ref_secs > 0:
        st.markdown(f'<meta http-equiv="refresh" content="{ref_secs}">', unsafe_allow_html=True)

    period = {"1d": "1y", "1h": "3mo", "4h": "6mo"}[interval]

    # Sektör filtresi — sp_info yüklendikten sonra doldurulacak
    st.divider()
    st.caption(f"Veri: **{period}** · Cache: **5 dk**")
    st.caption("Skor: RSI+StochRSI(20)+HacimYön(25)+Fib+Squeeze(15)+MACD(15)+RS_SPY(15)+ADX(10)")

    if st.session_state['watchlist']:
        st.divider()
        st.subheader("⭐ Watchlist")
        for w in list(st.session_state['watchlist']):
            c1, c2 = st.columns([4, 1])
            c1.write(f"**{w}**")
            c2.button("✕", key=f"rm_{w}", on_click=wl_remove, args=(w,))


# ── Ana Alan ──────────────────────────────────────────────────────────────────
st.title("🎯 SwingTrader Screener")
st.caption("Gün içi patlama potansiyeli taşıyan hisseleri bulur — S&P evreninden, sabit sinyal sistemi ile.")

sp_info = get_sp_list(pool)
tickers = tuple(sp_info.keys())

# Sektör filtresi (sp_info yüklendikten sonra)
all_sectors = sorted(set(v['Sector'] for v in sp_info.values()))
with st.sidebar:
    sector_filter = st.multiselect("🏭 Sektör Filtresi", all_sectors,
                                   placeholder="Tümü (filtre yok)")

with st.status(f"📡 {pool} verisi yükleniyor ({len(tickers)} hisse)...", expanded=True) as status:
    st.write("⬇️ Fiyat verisi indiriliyor — ilk açılışta 20-30 saniye sürebilir...")
    raw = fetch_data(tickers, period, interval)
    if raw.empty:
        status.update(label="❌ Veri çekilemedi", state="error")
        st.stop()
    st.write("📈 SPY referansı alınıyor...")
    spy_ret = get_spy_return(period)
    st.write(f"🔍 {len(tickers)} hisse taranıyor ve puanlanıyor...")
    results, valid_count = run_screen(
        raw, tickers, sp_info, rsi_lim, spy_ret,
        raw.shape, sma_filter, tuple(sector_filter)
    )
    status.update(label=f"✅ Tamamlandı — {len(results)} sinyal bulundu", state="complete", expanded=False)

if results:
    try:
        save_score_history(results)
    except Exception:
        pass

# ── Metrikler ─────────────────────────────────────────────────────────────────
st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Havuz",   f"{len(tickers)} hisse")
m2.metric("Taranan", f"{valid_count} geçerli")
m3.metric("Sinyal",  f"{len(results)} aday")
m4, m5, m6 = st.columns(3)
m4.metric("SPY 20g",    f"{spy_ret:+.1f}%")
m5.metric("RSI Limiti", f"< {rsi_lim}")
m6.metric("Güncelleme", datetime.datetime.now().strftime("%H:%M:%S"))

# ── Tablo ─────────────────────────────────────────────────────────────────────
st.subheader("📊 Top 20 — En Yüksek Patlama Potansiyeli")

if results:
    df_res = pd.DataFrame([{k: v for k, v in r.items() if k != '_detail'} for r in results])
    df_res.index = range(1, len(df_res) + 1)
    df_res.insert(1, '⭐', df_res['Hisse'].apply(
        lambda h: '⭐' if h in st.session_state['watchlist'] else ''))

    styled = (
        df_res.style
        .format({'Fiyat ($)': '{:g}', 'Değişim (%)': '{:+g}', 'RSI': '{:g}',
                 'StochRSI': '{:g}', 'Hacim (×avg)': '{:g}',
                 'RS vs SPY': '{:+g}', '52H Uzaklık': '{:g}%',
                 'ADX': '{:g}', 'Skor': '{:g}'})
        .map(lambda v: f'color:{"#00ff00" if v > 0 else "#ff4b4b"};font-weight:bold',
             subset=['Değişim (%)', 'RS vs SPY'])
        .map(lambda v: (
            'color:#ff4500;font-weight:bold' if v >= 75 else
            'color:#00ff00;font-weight:bold' if v >= 55 else
            'color:#ffcc00;font-weight:bold' if v >= 35 else
            'color:#888888'), subset=['Skor'])
        .set_properties(**{'text-align': 'center'})
    )
    st.dataframe(styled, width="stretch")
    st.download_button("📥 CSV Olarak İndir",
                       df_res.to_csv(index=True).encode('utf-8-sig'),
                       f"swingtrader_{datetime.date.today()}.csv", "text/csv")
    st.session_state['filtered'] = df_res['Hisse'].tolist()
else:
    st.warning("Sinyal bulunamadı. RSI limitini artırın veya SMA50 filtresini kapatın.")
    st.session_state['filtered'] = []


# ── Grafik ────────────────────────────────────────────────────────────────────
if not st.session_state.get('filtered'):
    st.stop()

available = [t for t in st.session_state['filtered'] if t in raw['Close'].columns]
if not available:
    st.warning("Güncel veri yok. Sayfayı yenileyin.")
    st.stop()

st.divider()
sel = st.selectbox("🎯 Detaylı Analiz:", available)

df_c = pd.DataFrame({
    'Open': raw['Open'][sel], 'High': raw['High'][sel],
    'Low':  raw['Low'][sel],  'Close': raw['Close'][sel],
    'Volume': raw['Volume'][sel],
}).dropna()
df_c = compute_indicators(df_c)
df_c['Pct']   = df_c['Close'].pct_change().mul(100).fillna(0)
df_c['Color'] = (df_c['Close'] >= df_c['Open']).map({True: '#26a69a', False: '#ef5350'})

gc1, gc2, gc3, gc4, gc5 = st.columns(5)
ma1_p      = gc1.number_input("Kısa MA",  1, 100, 10)
ma2_p      = gc2.number_input("Uzun MA",  1, 200, 50)
show_bb    = gc3.checkbox("Bollinger", value=True)
show_fib   = gc4.checkbox("Fibonacci", value=True)
panel3_opt = gc5.selectbox("Alt Panel", ["MACD", "Stoch RSI", "OBV"], index=0)

df_c['MA1'] = df_c['Close'].rolling(int(ma1_p)).mean()
df_c['MA2'] = df_c['Close'].rolling(int(ma2_p)).mean()

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                    row_heights=[0.60, 0.20, 0.20],
                    subplot_titles=('', 'Hacim', panel3_opt))

# Panel 1: Mum grafik
fig.add_trace(go.Candlestick(
    x=df_c.index, open=df_c['Open'], high=df_c['High'],
    low=df_c['Low'], close=df_c['Close'], name='Fiyat',
    increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
    customdata=df_c[['Pct', 'Color']].values,
    hovertemplate=(
        "<span style='font-size:14px;font-family:Arial Black;color:%{customdata[1]}'>"
        "A:%{open:.2f} | Y:%{high:.2f} | D:%{low:.2f} | K:%{close:.2f}"
        " | %{customdata[0]:.2f}%</span><extra></extra>"
    ),
), row=1, col=1)

fig.add_trace(go.Scatter(x=df_c.index, y=df_c['MA1'],
    line=dict(color='#00d4ff', width=1.5), name=f'MA{int(ma1_p)}'), row=1, col=1)
fig.add_trace(go.Scatter(x=df_c.index, y=df_c['MA2'],
    line=dict(color='#ff9800', width=1.5), name=f'MA{int(ma2_p)}'), row=1, col=1)

if show_bb:
    fig.add_trace(go.Scatter(x=df_c.index, y=df_c['BB_Upper'],
        line=dict(color='rgba(100,180,255,0.4)', width=1, dash='dot'),
        showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_c.index, y=df_c['BB_Lower'],
        line=dict(color='rgba(100,180,255,0.4)', width=1, dash='dot'),
        fill='tonexty', fillcolor='rgba(100,180,255,0.05)',
        showlegend=False), row=1, col=1)

if show_fib:
    rh, rl = df_c['High'].max(), df_c['Low'].min()
    for ratio, col_hex in zip(
        [0.236, 0.382, 0.500, 0.618, 0.786],
        ['#b2b2ff', '#2962ff', '#ffffff', '#d50000', '#ff6d00']
    ):
        f_v = rh - (rh - rl) * ratio
        fig.add_hline(y=f_v, line_dash="dot", line_color=col_hex, opacity=0.45,
                      row=1, col=1,
                      annotation_text=f"Fib {ratio:.3f}  ${f_v:.2f}",
                      annotation_font_color=col_hex, annotation_position="left")

# Squeeze göstergesi (grafiğin altında küçük noktalar)
squeeze_y = df_c['Low'] * 0.997
sq_colors = ['#ffcc00' if s else 'rgba(255,255,255,0.1)' for s in df_c['Squeeze'].fillna(False)]
fig.add_trace(go.Scatter(
    x=df_c.index, y=squeeze_y, mode='markers',
    marker=dict(color=sq_colors, size=3, symbol='circle'),
    name='Squeeze', hoverinfo='none', showlegend=False,
), row=1, col=1)

# Panel 2: Hacim
fig.add_trace(go.Bar(x=df_c.index, y=df_c['Volume'],
    marker_color=df_c['Color'].tolist(), opacity=0.6, showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=df_c.index, y=df_c['Vol_MA20'],
    line=dict(color='#ffcc00', width=1.2), name='Vol MA20'), row=2, col=1)

# Panel 3: Seçime göre
if panel3_opt == "MACD":
    macd_colors = (df_c['Hist'] >= 0).map({True: '#26a69a', False: '#ef5350'})
    fig.add_trace(go.Bar(x=df_c.index, y=df_c['Hist'],
        marker_color=macd_colors.tolist(), opacity=0.8, showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_c.index, y=df_c['MACD'],
        line=dict(color='#00d4ff', width=1.2), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_c.index, y=df_c['Signal'],
        line=dict(color='#ff9800', width=1.2), name='Sinyal'), row=3, col=1)

elif panel3_opt == "Stoch RSI":
    fig.add_trace(go.Scatter(x=df_c.index, y=df_c['StochRSI_K'],
        line=dict(color='#00d4ff', width=1.5), name='StochRSI K'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_c.index, y=df_c['StochRSI_D'],
        line=dict(color='#ff9800', width=1.2), name='StochRSI D'), row=3, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="rgba(255,100,100,0.4)", row=3, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="rgba(100,255,100,0.4)", row=3, col=1)

else:  # OBV
    fig.add_trace(go.Scatter(x=df_c.index, y=df_c['OBV'],
        line=dict(color='#b2b2ff', width=1.5), name='OBV'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_c.index, y=df_c['OBV_MA'],
        line=dict(color='#ff9800', width=1.2), name='OBV MA20'), row=3, col=1)

fig.update_layout(
    template="plotly_dark", height=850, hovermode="x unified",
    xaxis_rangeslider_visible=False,
    hoverlabel=dict(bgcolor="rgba(10,10,10,0.95)", font_size=13),
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    showlegend=False,
    margin=dict(l=50, r=20, t=20, b=20),
)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.07)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.07)')

breaks = [dict(bounds=["sat", "mon"])]
if interval != "1d":
    breaks.append(dict(bounds=[16, 9.5], pattern="hour"))
fig.update_xaxes(rangebreaks=breaks)

st.plotly_chart(fig, width="stretch")

# ── Skor Detayı & Geçmiş ──────────────────────────────────────────────────────
with st.expander(f"📋 {sel} — Skor Detayı & Geçmiş"):
    match = next((r for r in results if r['Hisse'] == sel), None)
    if match:
        d = match['_detail']
        # 3+3 layout
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("RSI + StochRSI", f"{d['rsi']} / 20",  help="Çift onaylı aşırı satım")
        dc2.metric("Hacim Yönü",     f"{d['vol']} / 25",  help="Yeşil mum + hacim spike")
        dc3.metric("Fib + Squeeze",  f"{d['fib']} / 15",  help="Fib seviyesi + BB sıkışması")
        dc4, dc5, dc6 = st.columns(3)
        dc4.metric("MACD",           f"{d['macd']} / 15", help="Histogram yönü")
        dc5.metric("RS vs SPY",      f"{d['rs']} / 15",   help="Piyasaya göre güç")
        dc6.metric("ADX Trend",      f"{d['adx']} / 10",  help="Trend gücü ve yönü")
        st.info(f"**Toplam Skor: {match['Skor']} / 100** — {match['Potansiyel']}")

        # Ek göstergeler
        adx_val = match['ADX']
        sq_val  = match['Squeeze']
        st.caption(f"ADX: **{adx_val}** {'(Güçlü trend)' if adx_val > 30 else '(Zayıf trend)' if adx_val < 20 else '(Orta trend)'}  |  Squeeze: **{sq_val}**")

    hist_df = get_ticker_history(sel)
    if len(hist_df) > 1:
        st.markdown("**📈 Skor Trendi (Son 30 Gün)**")
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=hist_df['Tarih'], y=hist_df['Skor'],
            mode='lines+markers', line=dict(color='#ff9800', width=2), marker=dict(size=6)))
        fig_h.add_hrect(y0=75, y1=100, fillcolor="rgba(255,69,0,0.1)",   line_width=0)
        fig_h.add_hrect(y0=55, y1=75,  fillcolor="rgba(0,255,0,0.07)",   line_width=0)
        fig_h.add_hrect(y0=35, y1=55,  fillcolor="rgba(255,204,0,0.07)", line_width=0)
        fig_h.update_layout(template="plotly_dark", height=200, showlegend=False,
                            margin=dict(l=40, r=20, t=10, b=30), yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_h, width="stretch")
    else:
        st.caption("Skor trendi için birden fazla tarama gerekiyor.")
