import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta
import requests

# Load models and scalers
xgb_model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
lstm_model = load_model('lstm_model.h5')

# Streamlit UI
st.set_page_config(page_title="Hybrid Crypto Analyzer", layout="wide")
st.title("🚀 Hybrid Cryptocurrency Analyzer")
st.markdown("""
Welcome! This app lets you:
- 📊 **Categorize** any cryptocurrency by market cap
- 📉 **Predict** its future price using deep learning

Type a coin ID below (e.g., `bitcoin`, `ethereum`) and choose an option:
""")

# --- Crypto Data Fetching ---
@st.cache_data
def fetch_crypto_data(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": coin_id,
        "order": "market_cap_desc",
        "per_page": 1,
        "page": 1,
        "sparkline": False
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return None

@st.cache_data
def fetch_historical_prices(coin_id, days=180):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        prices = response.json()['prices']
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        df.drop(columns=['timestamp'], inplace=True)
        return df
    else:
        return None

# --- Categorization ---
def get_classification_reason(market_cap):
    if market_cap >= 10e9:
        return f"because its market cap of ${market_cap:,.0f} exceeds the $10B threshold for Large Cap."
    elif market_cap >= 1e9:
        return f"because its market cap of ${market_cap:,.0f} falls between $1B and $10B, placing it in the Mid Cap category."
    else:
        return f"because its market cap of ${market_cap:,.0f} is below $1B, making it a Small Cap coin."

def predict_category(coin_id):
    data = fetch_crypto_data(coin_id)
    if not data:
        return "❌ Cryptocurrency not found."

    coin = data[0]
    features = np.array([
        coin.get("market_cap", 0),
        coin.get("total_volume", 0),
        coin.get("circulating_supply", 0),
        coin.get("max_supply", 0),
        coin.get("high_24h", 0),
        coin.get("low_24h", 0)
    ]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = xgb_model.predict(features_scaled)[0]
    mapping = {0: "Small Cap", 1: "Mid Cap", 2: "Large Cap"}
    category = mapping[prediction]

    reason = get_classification_reason(coin.get("market_cap", 0))

    with st.expander("🔍 Market Data Used"):
        st.markdown(f"""
        - Market Cap: **${coin['market_cap']:,.0f}**
        - Volume (24h): **${coin['total_volume']:,.0f}**
        - Circulating Supply: **{coin['circulating_supply']:,.0f}**
        - Max Supply: **{coin['max_supply'] if coin['max_supply'] else 'Unknown'}**
        - 24h High: **${coin['high_24h']:,.2f}**
        - 24h Low: **${coin['low_24h']:,.2f}**
        """)

    return f"✅ The cryptocurrency **{coin_id.capitalize()}** is categorized as **{category}**, {reason}"

# --- Price Prediction using real historical data ---
def prepare_lstm_input(price_series, sequence_length=60):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(price_series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(sequence_length, len(scaled) - 1):
        X.append(scaled[i - sequence_length:i])
        y.append(scaled[i + 1])  # Predict the price at t+1
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler, scaled

def predict_price(coin_id):
    df = fetch_historical_prices(coin_id, days=180)
    if df is None or len(df) < 61:
        return "❌ Not enough historical data to make prediction."

    X, y, price_scaler, scaled = prepare_lstm_input(df['price'])
    last_sequence = scaled[-60:].reshape(1, 60, 1)
    predicted_scaled = lstm_model.predict(last_sequence)[0][0]
    predicted_price = price_scaler.inverse_transform([[predicted_scaled]])[0][0]

    latest_actual_price = df['price'].iloc[-1]
    diff = predicted_price - latest_actual_price
    diff_percent = (diff / latest_actual_price) * 100
    accuracy_comment = "✅ Close to actual" if abs(diff_percent) <= 3 else "⚠️ Noticeable deviation"

    st.subheader("📉 Price Chart")
    fig, ax = plt.subplots(figsize=(10, 5))
    df['price'].plot(ax=ax, label='Historical Price', color='blue')
    future_date = df.index[-1] + pd.Timedelta(days=1)
    ax.plot(future_date, predicted_price, 'ro', label='Predicted Next Price')
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.set_title(f"Price Prediction for {coin_id.capitalize()}")
    st.pyplot(fig)

    with st.expander("📘 How this prediction was made"):
        st.markdown(f"""
        - Based on the **past 60 days** of daily closing prices for `{coin_id}`
        - Data fetched from **CoinGecko API** in real-time
        - LSTM model trained on historical crypto trends
        - ⚠️ Model does *not* include market sentiment or news
        """)

    with st.expander("🕰 Recent Price Trend"):
        st.dataframe(df['price'].tail(5).rename("Closing Price (USD)"))

    st.markdown("""
    ### 🔍 Prediction vs Actual
    - 📈 **Predicted Price:** ${:,.2f}
    - 💰 **Actual Latest Price:** ${:,.2f}
    - 🧮 **Difference:** ${:+,.2f} ({:+.2f}%)
    - {}
    """.format(predicted_price, latest_actual_price, diff, diff_percent, accuracy_comment))

    st.balloons()
    return f"📈 Predicted future price for **{coin_id.capitalize()}**: **${predicted_price:,.2f}**"

# --- User Input ---
crypto_name = st.text_input("🔍 Enter cryptocurrency ID (e.g., bitcoin, ethereum)")

if crypto_name:
    option = st.radio("What would you like to do?", ["📊 Categorize Cryptocurrency", "📉 Predict Future Price"])

    if option == "📊 Categorize Cryptocurrency":
        result = predict_category(crypto_name.lower())
        st.success(result)

    elif option == "📉 Predict Future Price":
        result = predict_price(crypto_name.lower())
        st.success(result)

st.markdown("---")
st.caption("Built with ❤️ using XGBoost + LSTM + Streamlit")
