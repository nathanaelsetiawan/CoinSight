import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
from plotly.subplots import make_subplots
import joblib
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler


st.title("Crypto Price History & Prediction")
# Inisialisasi tabs
price_history, forecast = st.tabs(['Price History', 'Prediction'])

with price_history:
    # Inisialisai ticker
    tickers = ('ADA-USD', 'BNB-USD', 'BTC-USD', 'DOGE-USD', 'ETH-USD', 'HBAR-USD', 'LINK-USD', 'TRX-USD', 'XLM-USD', 'XRP-USD')

    # Pilihan periode prediksi
    period_options = {
        '1D':'1d',
        '1W':'1wk',
        '1M':'1mo',
        '1Y':'1y',
        'MAX':'max'
    }

    # Pilih koin
    ticker = st.selectbox('Select Coin:', tickers, key='ticker_select')

    # Tambahkan blok reset ini setelah selectbox
    if 'prev_ticker' not in st.session_state:
        st.session_state.prev_ticker = None
    if 'show_graph' not in st.session_state:
        st.session_state.show_graph = False

    if st.session_state.prev_ticker != st.session_state.get('ticker_select'):
        st.session_state.show_graph = False
        st.session_state.prev_ticker = st.session_state.get('ticker_select')

    # Tombol untuk menampilkan grafik
    if st.button('Show Graph', key='grafik_historis'):
        st.session_state.show_graph = True

    if st.session_state.get('show_graph', False):

    # Pilih periode dan interval
        col1, col2 = st.columns([5, 2])
        with col2:
            period_label = st.segmented_control('Select Period:', 
                                        options=period_options.keys(), 
                                        selection_mode='single',
                                        default='1M')

    # Ambil value yang dipilih
        if period_label is None:
            st.warning("Silakan pilih periode terlebih dahulu.")
            st.stop()
        elif period_label not in period_options:
            st.error("Periode yang dipilih tidak valid.")
            st.stop()
        else:
            selected_period = period_options[period_label]

        @st.cache_data
        def fetch_weekly_price_history(ticker, period, interval):
            data = yf.Ticker(ticker).history(period=period, interval=interval)
            return data
        
        with st.spinner("Loading..."): # ini spinner
            df = fetch_weekly_price_history(ticker, selected_period, interval='1d')
            
        if df.empty:
            st.warning("Data tidak tersedia untuk kombinasi ini.")
        else:
            df = df.rename_axis("Date").reset_index()

            # Buat subplot: candlestick + volume
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True,
                row_heights=[0.8, 0.2],
                vertical_spacing=0.05
            )

            fig.add_trace(go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                increasing_line_color='limegreen',
                increasing_fillcolor='limegreen',
                decreasing_line_color='red',
                decreasing_fillcolor='red',
                line_width=1,
                name='Price',
                showlegend=True
                
            ), row=1, col=1)

            volume_colors = ['green' if c >= p else 'red'
                for c, p in zip(df['Close'][1:], df['Close'][:-1])]
            volume_colors.insert(0, 'green')

            # --- Bar Volume ---
            fig.add_trace(go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name='Volume',
                marker_color=volume_colors,
                marker=dict(line=dict(width=0)),
                opacity=0.4,
                showlegend=True,
            ), row=2, col=1)

            # --- Atur layout ---
            fig.update_layout(
                xaxis=dict(
                    rangeslider_visible=False
                ),
                xaxis2=dict(
                    showgrid=False,
                    showticklabels=True,
                    tickformat="%Y-%m-%d",
                    tickmode="auto",
                    nticks=10,
                    type="date",
                    rangeslider_visible=False,
                ),
                yaxis=dict(
                    title = "Price (USD)",
                    side = 'right',
                    tickformat=".2f",
                    tickmode="auto",
                    nticks=8,
                    showgrid=True,
                    gridcolor="rgba(220,220,220,0.2)",
                    gridwidth=0.1,   
                ),
                yaxis2=dict(
                    side='left',
                    tickformat=".2f",
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False,
                ),
                legend=dict(
                    x=0.5, 
                    y=1.25, 
                    xanchor='center', 
                    yanchor='top',
                    orientation='h',
                    bgcolor='rgba(0,0,0,0)'
                ),
            )

            st.plotly_chart(fig, use_container_width=True)


###################### MULTI TABS ##########################

with forecast:
    # --- Load Scalers ---
    @st.cache_resource
    def load_scaler():
        return joblib.load('models/grouped_scalers.pkl')

    grouped_scalers = load_scaler()

    # --- UI ---
    ticker = ['ADA-USD', 'BNB-USD', 'BTC-USD', 'DOGE-USD', 'ETH-USD', 'HBAR-USD', 'LINK-USD', 'TRX-USD', 'XLM-USD', 'XRP-USD']
    ticker = st.selectbox("Select Coin:", list(grouped_scalers.keys()), key='ticker_select_forecast')
    features = ['Close', 'Volume', 'MA_21', 'MA_100', 'RSI_28']
    window_size = 60

    forecast_options = {
        "1 Week": 7,
        "1 Month": 30,
        "3 Month": 90,
        "6 Month": 180,
        "1 Year": 365
    }
    forecast_choice = st.selectbox("Select Prediction Time:", list(forecast_options.keys()))
    n_steps = forecast_options[forecast_choice]

    # --- Load Model ---
    @st.cache_resource
    def load_model_by_ticker(ticker_name):
        model_path = f'models/{ticker_name}_model.h5'
        return load_model(model_path)

    model = load_model_by_ticker(ticker)

    # Atur durasi historis berdasarkan prediksi
    if forecast_choice in ["1 Week", "1 Month"]:
        period_hist = '180d'
    else:
        period_hist = '1y'

    if st.button("Show Graph", key="grafik_prediksi"):
        with st.spinner("Loading..."):
            # --- Ambil data historis ---
            df = yf.Ticker(ticker).history(period=period_hist)
            df['MA_21'] = df['Close'].rolling(window=21).mean()
            df['MA_100'] = df['Close'].rolling(window=100).mean()

            # Hitung RSI 28 hari
            delta = df['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(28).mean()
            avg_loss = loss.rolling(28).mean()
            rs = avg_gain / avg_loss
            df['RSI_28'] = 100 - (100 / (1 + rs))

            df.dropna(inplace=True)

            if len(df) < window_size:
                st.warning(f"Data tidak cukup. Minimal {window_size} data setelah indikator dihitung.")
            else:
                try:
                    df_window = df[features].iloc[-window_size:]  # Ambil data terakhir
                    predictions = []
                    window_data = df_window.copy()

                    for i in range(n_steps):
                        scaled_window = []

                        for feat in features:
                            # Penyesuaian nama kolom dan nama scaler
                            if feat in ['Close', 'Volume']:
                                feat_name = f"{feat}_{ticker}"
                                col_name = feat  # nama kolom di df_window
                            else:
                                feat_name = f"{ticker}_{feat}"
                                col_name = feat

                            # Pastikan scaler tersedia
                            if feat_name not in grouped_scalers[ticker]:
                                st.error(f"❌ Scaler tidak ditemukan: {feat_name}")
                                st.stop()

                            scaler = grouped_scalers[ticker][feat_name]
                            values = window_data[col_name].values.reshape(-1, 1)
                            values_df = pd.DataFrame(values, columns=[feat_name]) # ini tambahan
                            scaled = scaler.transform(values_df)
                            scaled_window.append(scaled.flatten())

                        # Format input LSTM
                        X_scaled = np.array(scaled_window).T
                        X_input = X_scaled.reshape(1, window_size, len(features))
                        next_pred = model.predict(X_input)[0][0]
                        predictions.append(next_pred)

                        # Inverse transform hasil prediksi 'Close'
                        close_feat_name = f"Close_{ticker}"
                        close_real = grouped_scalers[ticker][close_feat_name].inverse_transform([[next_pred]])[0][0]

                        # Susun baris baru untuk ditambahkan ke window
                        new_row = {'Close': close_real}
                        for feat in ['Volume', 'MA_21', 'MA_100', 'RSI_28']:
                            new_row[feat] = window_data.iloc[-1][feat]

                        window_data = pd.concat([window_data, pd.DataFrame([new_row])], ignore_index=True)
                        if len(window_data) > window_size:
                            window_data = window_data.iloc[-window_size:]


                    # --- Siapkan tanggal dan hasil prediksi (inverse ke harga asli) ---
                    feat_name = f'Close_{ticker}'
                    predicted_closes = grouped_scalers[ticker][feat_name].inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

                    future_dates = [df.index[-1] + pd.Timedelta(days=i + 1) for i in range(n_steps)]

                    # --- Buat candlestick simulasi prediksi (OHLC) ---
                    pred_ohlc = []
                    for i, close in enumerate(predicted_closes):
                        open_price = predicted_closes[i - 1] if i > 0 else df['Close'].iloc[-1]
                        high = max(open_price, close) * 1.01
                        low = min(open_price, close) * 0.99
                        pred_ohlc.append((future_dates[i], open_price, high, low, close))

                    df_pred = pd.DataFrame(pred_ohlc, columns=["Date", "Open", "High", "Low", "Close"])
                    df_pred["Date"] = pd.to_datetime(df_pred["Date"])

                    # --- Plotting ---
                    fig = make_subplots(
                            rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.8, 0.2], 
                            vertical_spacing=0.05
                        )

                    # Historis
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='Price History',
                        increasing_line_color='limegreen',
                        increasing_fillcolor='limegreen',
                        decreasing_line_color='red',
                        decreasing_fillcolor='red',
                        showlegend=True
                    ), row=1, col=1)

                    # Prediksi
                    fig.add_trace(go.Candlestick(
                        x=df_pred["Date"],
                        open=df_pred["Open"],
                        high=df_pred["High"],
                        low=df_pred["Low"],
                        close=df_pred["Close"],
                        name='Prediction',
                        increasing_line_color='blue',
                        increasing_fillcolor='blue',
                        decreasing_line_color='blue',
                        decreasing_fillcolor='blue',
                        showlegend=True
                    ), row=1, col=1)

                    # Volume bar warna hijau/merah
                    volume_colors = ['green' if c >= p else 'red' for c, p in zip(df['Close'][1:], df['Close'][:-1])]
                    volume_colors.insert(0, 'green')

                    fig.add_trace(go.Bar(
                        x=df.index,
                        y=df['Volume'],
                        name='Volume',
                        marker_color=volume_colors,
                        marker=dict(line=dict(width=0)),
                        opacity=0.4,
                        showlegend=True,
                        
                    ), row=2, col=1)

                    # Layout
                    fig.update_layout(
                        xaxis=dict(
                            rangeslider_visible=False,
                        ),
                        xaxis2=dict(
                            showgrid=False, 
                            showticklabels=True,
                            tickformat="%Y-%m-%d",
                            rangeslider_visible=False,
                            type="date"
                        ),       
                        yaxis=dict(
                            title="Price (USD)", 
                            side='right',
                            tickformat=".2f", 
                            gridcolor="rgba(220,220,220,0.2)",
                            nticks=8
                        ),
                        yaxis2=dict(
                            showgrid=False, 
                            showticklabels=False
                        ),
                        legend=dict(
                            x=0.5, 
                            y=1.25, 
                            xanchor='center',
                            orientation='h', 
                            bgcolor='rgba(0,0,0,0)'
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True)
            
                except Exception as e:
                        st.error(f"❌ Terjadi kesalahan saat scaling atau prediksi: {e}")