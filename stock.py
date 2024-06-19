import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained model
model = load_model("stockpredict.keras")

def predict_and_suggest_action(data_test_scale, scaler):
    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i - 100:i])
        y.append(data_test_scale[i, 0])

    x, y = np.array(x), np.array(y)

    predict = model.predict(x)

    scale = 1 / scaler.scale_

    predict = predict * scale
    y = y * scale

    return predict, y

def suggest_action(predicted_price, current_price):
    if predicted_price > current_price:
        return "Buy"
    else:
        return "Sell"

def main():
    # Define header content
    st.title('Stock Market Dashboard')
    st.markdown("---")

    # Date range selector for selecting the data range to predict
    st.subheader("Select Data Range to Predict")
    start_date = st.date_input("Start Date", pd.to_datetime('2012-01-01'))
    end_date = st.date_input("End Date", pd.to_datetime('2022-12-31'))

    # Dropdown for selecting stock symbol
    selected_stock = st.selectbox('Select Stock Symbol', ['AAPL', 'GOOG', 'MSFT', 'AMZN'])

    data = yf.download(selected_stock, start=start_date, end=end_date)

    st.write(data)

    data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

    scaler = MinMaxScaler(feature_range=(0, 1))

    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    # Predict and suggest action
    predict, y = predict_and_suggest_action(data_test_scale, scaler)
    suggested_action = suggest_action(predict[-1], data.Close.iloc[-1])

   # Additional analysis
    st.subheader('Analysis') 
    
    # Plot predicted vs actual prices
    st.subheader('Original Price vs Predicted Price')
    fig, ax = plt.subplots()
    ax.plot(y, 'r', label='Original Price')
    ax.plot(predict, 'g', label='Predicted Price')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

    # Display predicted and actual values
    st.subheader('Predicted vs Actual Values')
    results = pd.DataFrame({'Predicted': predict.flatten(), 'Actual': y.flatten()})
    st.write(results)

    # Plot historical closing prices
    st.subheader('Historical Closing Prices')
    fig_close, ax_close = plt.subplots()
    ax_close.plot(data['Close'], label='Closing Price')
    ax_close.set_xlabel('Date')
    ax_close.set_ylabel('Price')
    ax_close.legend()
    st.pyplot(fig_close)

    # Plot moving averages
    st.subheader('Moving Averages')
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    fig_ma, ax_ma = plt.subplots()
    ax_ma.plot(data['Close'], label='Closing Price')
    ax_ma.plot(data['MA50'], label='50-Day MA')
    ax_ma.plot(data['MA200'], label='200-Day MA')
    ax_ma.set_xlabel('Date')
    ax_ma.set_ylabel('Price')
    ax_ma.legend()
    st.pyplot(fig_ma)

    # Plot volatility
    st.subheader('Volatility')
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=50).std() * np.sqrt(50)
    fig_vol, ax_vol = plt.subplots()
    ax_vol.plot(data['Volatility'], label='Volatility')
    ax_vol.set_xlabel('Date')
    ax_vol.set_ylabel('Volatility')
    ax_vol.legend()
    st.pyplot(fig_vol)

    # Suggested action
    st.subheader('Suggested Action')
    if suggested_action == "Buy":
        st.success(f"Suggested Action: {suggested_action}")
    else:
        st.error(f"Suggested Action: {suggested_action}")

    # Display MarketWatch queries
    st.markdown("---")
    st.markdown("## Queries")
    st.markdown("Contact Us:")
    st.markdown("- main site: https://tangerine-kangaroo-71ab69.netlify.app/")
    st.markdown("- Email: support@gmail.com")

    # Horizontal scrolling disclaimer text
    st.markdown("---")
    st.write(
        """
        <div style="overflow-x: auto; white-space: nowrap;">
            <marquee behavior="scroll" direction="left" scrollamount="5">
                Intraday Data provided by FACTSET and subject to terms of use. 
                Historical and current end-of-day data provided by FACTSET. 
                All quotes are in local exchange time. Real-time last sale data for U.S. 
                stock quotes reflect trades reported through Nasdaq only. 
                Intraday data delayed at least 15 minutes or per exchange requirements.
            </marquee>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
