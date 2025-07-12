import streamlit as st
import requests
import pandas as pd
import time

# Ambil data koin per halaman (maks 250 per page)
@st.cache_data(ttl=120)
def get_all_coins(vs_currency='usd', total_pages=2, delay=1):
    all_data = []
    for page in range(1, total_pages + 1):
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": 250,
            "page": page,
            "sparkline": False,
            "price_change_percentage": "24h"
        }
        res = requests.get(url, params=params)
        if res.status_code == 200:
            all_data.extend(res.json())
        else:
            st.warning(f"Error fetching page {page}: {res.status_code}")
            break
        time.sleep(delay)
    return pd.DataFrame(all_data)

# Ambil koin trending (ID saja)
@st.cache_data(ttl=120)
def get_trending_coins():
    url = "https://api.coingecko.com/api/v3/search/trending"
    res = requests.get(url).json()
    coins = res['coins']
    trending = [{
        'id': c['item']['id'],
        'name': c['item']['name'],
        'symbol': c['item']['symbol'],
        'rank': c['item']['market_cap_rank']
    } for c in coins]
    return pd.DataFrame(trending)

# Ambil detail koin by ID (untuk ambil ikon)
def get_coin_image(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    res = requests.get(url)
    if res.status_code == 200:
        data = res.json()
        return data['image']['thumb']
    return ""

# =======================
# Dashboard Coinsight
# =======================
with st.spinner("Loading..."):
    df_all = get_all_coins(total_pages=2)  # hingga 1000 coin
    trending_df = get_trending_coins()

# Konversi persentase ke float
df_all['price_change_percentage_24h'] = pd.to_numeric(
    df_all['price_change_percentage_24h'], errors='coerce'
)

# Ambil Top 5 Coin
top5 = df_all[['name', 'symbol', 'current_price', 'image']].head(3)

# Ambil Top Gainers
gainers = df_all.sort_values(by='price_change_percentage_24h', ascending=False).head(3)

# =======================
# TAMPILAN LAYOUT 3 KOLOM
# =======================
st.title("Welcome to CoinSight 👋🏻")
st.markdown("The global cryptocurrency market situation on last 24 hours:")

col1, col2, col3 = st.columns(3)

# TOP COINS
with col1:
    with st.container(border=True):
        st.subheader("💰 Top Coins")
        for _, row in top5.iterrows():
            col_left, col_right = st.columns([3, 2])
            with col_left:
                st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:10px">
                        <img src="{row['image']}" width="25" style="border-radius:50%; border:0px solid #ccc;">
                        <span><b>{row['name']}</b></span>
                    </div>
                """, unsafe_allow_html=True)
            with col_right:
                st.markdown(f"<p style='text-align:right'><b>${row['current_price']:,.2f}</b></p>", unsafe_allow_html=True)

# TOP GAINERS
with col2:
    with st.container(border=True):
        st.subheader("🚀 Top Gainers (24h)")
        for _, row in gainers.iterrows():
            col_left, col_right = st.columns([3, 2])
            with col_left:
                st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:10px">
                        <img src="{row['image']}" width="25" style="border-radius:50%; border:0px solid #ccc;">
                        <span><b>{row['name']}</b></span>
                    </div>
                """, unsafe_allow_html=True)
            with col_right:
                st.markdown(f"<p style='text-align:right; color:limegreen'><b>+{row['price_change_percentage_24h']:.2f}%</b></p>", unsafe_allow_html=True)

# TRENDING
with col3:
    with st.container(border=True):
        st.subheader("🔥 Trending")
        if not trending_df.empty:
            for _, row in trending_df.iloc[:3].iterrows():
                image_url = get_coin_image(row['id'])
                col_left, col_right = st.columns([3, 2])
                with col_left:
                    st.markdown(f"""
                        <div style="display:flex; align-items:center; gap:10px">
                            <img src="{image_url}" width="25" style="border-radius:50%; border:0px solid #ccc;">
                            <span><b>{row['name']}</b></span>
                        </div>
                    """, unsafe_allow_html=True)
                with col_right:
                    st.markdown(f"<p style='text-align:right'>Rank #{row['rank']}</p>", unsafe_allow_html=True)
        else:
            st.write("Tidak dapat memuat data trending.")


# =======================
# RINGKASAN HARGA UNTUK 10 KOIN TERPILIH DENGAN HEADER
# =======================
st.write("")
st.markdown("""These are list of coins that can be predicted on CoinSight: """, unsafe_allow_html=True)

with st.container(border=True):

    # Header kolom
    st.markdown("""
        <div style="display:flex; justify-content:space-between; font-weight:600; padding:6px 0; border-bottom:1px solid #444;">
            <div style="width:5%">#</div>
            <div style="width:35%; text-align:left;">Name</div>
            <div style="width:20%; text-align:right;">Price</div>
            <div style="width:20%; text-align:right;">24h %</div>
            <div style="width:20%; text-align:right;">Volume</div>
        </div>
    """, unsafe_allow_html=True)

    target_id = ['cardano', 'binancecoin', 'bitcoin', 'hedera-hashgraph', 'dogecoin', 'ethereum', 'chainlink', 'tron', 'stellar', 'ripple']
    df_filtered = df_all[df_all['id'].isin(target_id)].reset_index(drop=True)

    if df_filtered.empty:
        st.warning("⚠️ Data koin tidak ditemukan.")
    else:
        for i, row in df_filtered.iterrows():
            price = row['current_price']
            volume = row['total_volume']
            change = row['price_change_percentage_24h']
            color = "limegreen" if change >= 0 else "red"
            symbol = row['symbol'].upper()
            name = row['name']
            image = row['image']

            st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid #222;">
                    <div style="width:5%;">{i+1}</div>
                    <div style="width:35%; display:flex; align-items:center; gap:10px;">
                        <img src="{image}" width="28" style="border-radius:50%; border:0px solid #ccc;">
                        <span><b>{name}</b> ({symbol}-USD)</span>
                    </div>
                    <div style="width:20%; text-align:right;"><b>${price:,.4f}</b></div>
                    <div style="width:20%; text-align:right; color:{color};"><b>{change:+.2f}%</b></div>
                    <div style="width:20%; text-align:right; color:white;">${volume/1_000_000_000:.2f}B</div>
                </div>
            """, unsafe_allow_html=True)