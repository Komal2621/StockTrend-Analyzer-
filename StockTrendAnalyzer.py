# 📦 --- Import Libraries ---
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import streamlit as st

# 📥 --- Streamlit Title ---
st.title("📈 Stock Price Forecasting Dashboard (ARIMA | SARIMA | Prophet)")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload your Stock Data CSV file", type=['csv'])

if uploaded_file is not None:
    # --- Load and Prepare Data ---
    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip()  # remove extra spaces if any
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values('Date', inplace=True)   # sort oldest → latest
    data.set_index('Date', inplace=True)

    st.subheader("📊 Original Data (Close Prices)")
    st.line_chart(data['Close'])

    # -----------------------------
    # --- ARIMA Forecasting ---
    # -----------------------------
    st.subheader("🔮 ARIMA Forecast")

    try:
        model_arima = ARIMA(data['Close'], order=(5, 1, 0))
        result_arima = model_arima.fit()

        forecast_arima = result_arima.forecast(steps=30)
        forecast_arima = pd.Series(
            forecast_arima,
            index=pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
        )

        # ✅ Show ARIMA Graph
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(data['Close'], label='Actual')
        ax1.plot(forecast_arima, label='ARIMA Forecast', color='red')
        ax1.set_title('ARIMA Forecast')
        ax1.legend()
        st.pyplot(fig1)

    except Exception as e:
        st.error(f"ARIMA Error: {e}")

    # -----------------------------
    # --- SARIMA Forecasting ---
    # -----------------------------
    st.subheader("🌦 SARIMA Forecast")

    try:
        model_sarima = SARIMAX(
            data['Close'],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 0, 12)
        )
        result_sarima = model_sarima.fit(disp=False)

        forecast_sarima = result_sarima.get_forecast(steps=30)
        forecast_mean = forecast_sarima.predicted_mean
        forecast_ci = forecast_sarima.conf_int()
        forecast_mean.index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')
        forecast_ci.index = forecast_mean.index

        # ✅ Show SARIMA Graph
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(data['Close'], label='Actual')
        ax2.plot(forecast_mean, label='SARIMA Forecast', color='green')
        ax2.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                         color='gray', alpha=0.3)
        ax2.set_title('SARIMA Forecast')
        ax2.legend()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"SARIMA Error: {e}")

    # -----------------------------
    # --- Prophet Forecasting ---
    # -----------------------------
    st.subheader("🔯 Prophet Forecast")

    try:
        prophet_df = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        model_prophet = Prophet()
        model_prophet.fit(prophet_df)

        future = model_prophet.make_future_dataframe(periods=30)
        forecast_prophet = model_prophet.predict(future)

        fig3 = model_prophet.plot(forecast_prophet)
        st.pyplot(fig3)
    except Exception as e:
        st.error(f"Prophet Error: {e}")

    st.success("✅ All forecasts generated successfully!")

else:
    st.info("👆 Please upload a CSV file (must include 'Date' and 'Close' columns).")
