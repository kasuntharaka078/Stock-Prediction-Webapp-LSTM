import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

st.title('Stock Trend Prediction')

symbol = st.text_input('Enter Stock Ticker', 'BTC-USD')
period='10y'
ticker = yf.Ticker(symbol)
df = ticker.history(period)

#Describing Data
st.subheader('Data from 10 years')
st.write(df.describe())

#visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12 ,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12 ,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12 ,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

# Spliting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0 : int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))

#Load my model
model = load_model('keras_model.h5')

#Testing Part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

#Making Predictions
y_predicted = model.predict(x_test)

y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
y_predicted = scaler.inverse_transform(y_predicted)

#FInal Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize = (12, 6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
