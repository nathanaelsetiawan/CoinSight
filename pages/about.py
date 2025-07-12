import streamlit as st

# Header
st.markdown("<h1 style='text-align: center;'>Welcome to CoinSight</h1>", unsafe_allow_html=True)
st.markdown("---")

# Introduction
st.markdown("""
<div style='font-size:16px;'>
In the fast-moving world of cryptocurrency, having reliable and predictive insights is essential. CoinSight offers a smart and data-driven platform to help you make informed investment decisions with confidence.
</div>
""", unsafe_allow_html=True)

# Section 1: AI Forecast
st.markdown("### 🔍 AI-Powered Forecasting")
st.markdown("""
CoinSight uses advanced machine learning—especially Long Short-Term Memory (LSTM) networks—to accurately predict crypto price movements over time.

<b>We train our models using key market indicators:</b>
- **Closing Price** – the final price at market close
- **Volume** – measures trade activity
- **MA-21** – short-term trend direction
- **MA-100** – medium-term momentum
- **RSI-28** – helps identify overbought or oversold conditions
""", unsafe_allow_html=True)

# Section 2: Forecast Options
st.markdown("### 📈 Flexible Forecast Periods")
st.markdown("""
Different investors need different perspectives. CoinSight provides prediction ranges to match your goals:

- 7 Days – for short-term trades
- 1 Month – track medium trends
- 3 Months – for strategic decisions
- 6 Months – for long-term insights
- 1 Year – macro investment planning
""")

# Section 3: Why Crypto
st.markdown("### 🌍 Why Focus on Cryptocurrency?")
st.markdown("""
Cryptocurrency is transforming finance globally—but volatility is high. CoinSight helps you manage that risk with insight into key drivers:

- Adoption & Regulation
- Blockchain Innovation
- Market Sentiment & News
- Economic Conditions
- Liquidity & Trading Volume
""")

# Call to Action
st.markdown("### ✅ Start with CoinSight Today")
st.markdown("""
Whether you're just starting out or already experienced, CoinSight empowers you to invest smarter in the crypto market using intelligent forecasts and clean visualizations.
""")