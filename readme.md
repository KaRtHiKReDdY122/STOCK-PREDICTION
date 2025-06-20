# 📊 Stock Price Movement Prediction App

**Author:** Karthik Reddy  
**Project Type:** Web App (Deployed using Streamlit Cloud)  
**Tech Stack:** Python, Streamlit, scikit-learn, yfinance, pandas, matplotlib

---

## 🚀 Overview

This application predicts whether a stock's closing price will go **UP or DOWN** tomorrow using a Machine Learning model trained on historical stock data.

🔮 It supports both **live data (online)** and **pre-downloaded CSVs (offline mode)**.

---

## 📌 Features

- Choose from top Indian stocks (like RELIANCE.NS, TCS.NS, INFY.NS, etc.)
- Train a Random Forest model on real stock market data
- View:
  - Prediction results (UP/DOWN)
  - Model accuracy
  - Last 5-day predictions
  - Stock closing price chart
- Offline mode support
- Simple UI built with Streamlit

---

## 🧠 How It Works

1. Downloads historical data using `yfinance`
2. Creates a target label:
   - `1` if next day's closing price is higher
   - `0` if not
3. Trains a `RandomForestClassifier` on Open, High, Low, Close, and Volume
4. Makes and displays predictions for recent days

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/stock-predictor.git
cd stock-predictor
pip install -r requirements.txt
streamlit run stock_dashboard.py
📁 stock-predictor/
├── stock_dashboard.py       # Main app file
├── requirements.txt         # Python dependencies
