import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

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

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))

    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    # Predict and suggest action
    predict, y = predict_and_suggest_action(data_test_scale, scaler)
    suggested_action = suggest_action(predict[-1], data.Close.iloc[-1])

    # Plot predicted vs actual prices
    st.subheader('Original Price vs Predicted Price')
    fig, ax = plt.subplots()
    ax.plot(predict, 'r', label='Original Price')
    ax.plot(y, 'g', label='Predicted Price')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

    # Display suggested action in colored boxes
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
