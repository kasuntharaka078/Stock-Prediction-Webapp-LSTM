# Stock Trend Prediction

This Streamlit web application is designed to predict stock trends based on historical data using a trained Keras model.

## Description

The application retrieves historical stock data using Yahoo Finance API and presents various visualizations such as closing price charts, moving averages, and predictions versus original prices using a pre-trained Keras model.

## Installation

1. Ensure Python is installed.
2. Clone this repository:

    ```bash
    git clone https://github.com/your-username/stock-trend-prediction.git
    ```

3. Navigate to the project directory and install the necessary dependencies:

    ```bash
    cd stock-trend-prediction
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

2. Access the application through your web browser at [http://localhost:8501](http://localhost:8501).

## How to Use

- **Enter Stock Ticker**: Input the stock symbol (e.g., AAPL, MSFT) to fetch the historical data for prediction.
- **Data Description**: Displays descriptive statistics of the retrieved 10-year historical stock data.
- **Closing Price vs Time Chart**: Shows the closing price of the stock over time.
- **Closing Price vs Time Chart with Moving Averages**: Displays the closing price chart with 100-day moving averages.
- **Closing Price vs Time Chart with Multiple Moving Averages**: Displays the closing price chart with 100-day and 200-day moving averages.
- **Predictions vs Original Chart**: Presents the predicted stock prices compared to the actual stock prices.

## Technologies Used

- Streamlit
- Pandas
- NumPy
- yfinance
- Matplotlib
- Keras

## Credits

This application utilizes the following libraries and tools:
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [yfinance](https://pypi.org/project/yfinance/)
- [Matplotlib](https://matplotlib.org/)
- [Keras](https://keras.io/)

## Webapp

[stockpredictionkeras](https://stockpredictionkeras.streamlit.app/)
