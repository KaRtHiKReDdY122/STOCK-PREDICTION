import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 Stock Price Movement Prediction App", layout="wide")
st.title("📊 Stock Price Movement Prediction App")
st.markdown("Choose a stock and predict whether its price will go UP or DOWN tomorrow.")

st.sidebar.title("Settings")
use_offline = st.sidebar.checkbox("📁 Use Offline Mode")

# Stock selection
st.sidebar.markdown("### Select Stock")
stock_list = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS",
    "ITC.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "BPCL.NS"
]
ticker = st.selectbox("Choose a stock to predict", stock_list)

# Load offline data if needed
if use_offline:
    try:
        data = pd.read_csv(f"data/{ticker.replace('.NS', '')}_data.csv", index_col=0, parse_dates=True)
        st.sidebar.success("✅ Using offline data")
    except:
        st.sidebar.error("❌ Offline data not found. Please disable offline mode.")
        st.stop()
else:
    st.sidebar.info(f"📡 Fetching data for {ticker}...")
    try:
        data = yf.download(ticker, start="2019-01-01", end="2024-12-31")
    except:
        st.error("❌ Failed to fetch data from Yahoo Finance.")
        st.stop()

# Feature engineering
if data.empty or "Close" not in data.columns:
    st.error("❌ Insufficient data to make prediction.")
    st.stop()

# Create binary target: 1 if tomorrow's Close > today's Close, else 0
data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
data = data.dropna()

# Features
features = ["Open", "High", "Low", "Close", "Volume"]
X = data[features]
y = data["Target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# UI layout
col1, col2 = st.columns(2)
with col1:
    st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
with col2:
    latest_data = X.tail(1)
    tomorrow_prediction = model.predict(latest_data)[0]
    direction = "📈 UP" if tomorrow_prediction == 1 else "📉 DOWN"
    st.metric("Tomorrow's Prediction", direction)

# Last 5 days predictions
st.subheader("📅 Last 5 Days Predictions")
predict_df = data.tail(6).copy()
predict_df["Prediction"] = model.predict(predict_df[features])
predict_df["Prediction"] = predict_df["Prediction"].replace({1: "UP", 0: "DOWN"})
st.dataframe(predict_df[["Close", "Prediction"]].tail(5))

# Plot
st.subheader("📊 Stock Closing Price Trend")
fig, ax = plt.subplots()
data["Close"].plot(ax=ax, title=f"{ticker} Closing Price")
st.pyplot(fig)

st.success("✅ Prediction Complete!")
