import requests
import pandas as pd
from textblob import TextBlob
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class AutoTrader:
    def __init__(self):
        self.stock_prices = pd.DataFrame()
        self.news_data = pd.DataFrame()
        self.social_media_data = pd.DataFrame()
        self.data = pd.DataFrame()

    def collect_stock_prices(self, symbol, start_date, end_date):
        url = f"https://api.example.com/stock_prices?symbol={symbol}&start_date={start_date}&end_date={end_date}"
        response = requests.get(url)
        if response.status_code == 200:
            self.stock_prices = pd.DataFrame(response.json())
        else:
            raise Exception(
                f"Failed to collect stock prices. Error code: {response.status_code}")

    def collect_news_data(self, symbol, start_date, end_date):
        url = f"https://api.example.com/news_data?symbol={symbol}&start_date={start_date}&end_date={end_date}"
        response = requests.get(url)
        if response.status_code == 200:
            self.news_data = pd.DataFrame(response.json())
        else:
            raise Exception(
                f"Failed to collect news data. Error code: {response.status_code}")

    def collect_social_media_data(self, symbol, start_date, end_date):
        url = f"https://api.example.com/social_media_data?symbol={symbol}&start_date={start_date}&end_date={end_date}"
        response = requests.get(url)
        if response.status_code == 200:
            self.social_media_data = pd.DataFrame(response.json())
        else:
            raise Exception(
                f"Failed to collect social media data. Error code: {response.status_code}")

    def preprocess_data(self):
        # Clean and preprocess collected data
        self.stock_prices['date'] = pd.to_datetime(self.stock_prices['date'])
        self.stock_prices.set_index('date', inplace=True)

        self.news_data['date'] = pd.to_datetime(self.news_data['date'])
        self.news_data.set_index('date', inplace=True)

        self.social_media_data['date'] = pd.to_datetime(
            self.social_media_data['date'])
        self.social_media_data.set_index('date', inplace=True)

        # Merge collected data
        self.data = pd.concat(
            [self.stock_prices, self.news_data, self.social_media_data], axis=1)
        self.data.fillna(0, inplace=True)

    def sentiment_analysis(self):
        # Apply sentiment analysis to news articles and social media data
        sentiment_scores = self.data['news'].apply(
            lambda text: TextBlob(text).sentiment.polarity)
        self.data['sentiment'] = sentiment_scores

    def technical_analysis(self):
        # Calculate technical indicators
        self.data['moving_average'] = self.data['close_price'].rolling(
            window=20).mean()
        self.data['bollinger_upper'] = self.data['moving_average'] + \
            2 * self.data['close_price'].rolling(window=20).std()
        self.data['bollinger_lower'] = self.data['moving_average'] - \
            2 * self.data['close_price'].rolling(window=20).std()
        self.data['macd'] = self.data['close_price'].ewm(
            span=12).mean() - self.data['close_price'].ewm(span=26).mean()
        self.data['rsi'] = self.calculate_rsi()

    def calculate_rsi(self):
        delta = self.data['close_price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def prepare_data(self):
        self.preprocess_data()
        self.sentiment_analysis()
        self.technical_analysis()

    def train_model(self):
        # Split the data into train and test sets
        train_size = int(len(self.data) * 0.8)
        train_data = self.data.iloc[:train_size, :]
        test_data = self.data.iloc[train_size:, :]

        # Scale the data
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        # Prepare the train and test datasets
        X_train, y_train = self.create_sequences(train_scaled)
        X_test, y_test = self.create_sequences(test_scaled)

        # Build and train the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True,
                  input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=16)

        return model

    def create_sequences(self, data):
        X = []
        y = []
        for i in range(len(data) - 30):
            X.append(data[i:i + 30])
            y.append(data[i + 30])
        return np.array(X), np.array(y)

    def visualize_data(self):
        # Plot historical stock prices
        fig = go.Figure()
        fig.add_trace(go.Line(x=self.data.index,
                      y=self.data['close_price'], name="Close Price"))
        fig.update_layout(title_text='Historical Stock Prices',
                          xaxis_rangeslider_visible=True)
        fig.show()

        # Plot technical indicators
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(self.data.index, self.data['moving_average'])
        axes[0, 0].set_title('Moving Average')
        axes[0, 1].plot(
            self.data.index, self.data['bollinger_upper'], 'r--', label='Upper Band')
        axes[0, 1].plot(
            self.data.index, self.data['bollinger_lower'], 'r--', label='Lower Band')
        axes[0, 1].plot(self.data.index,
                        self.data['close_price'], label='Close Price')
        axes[0, 1].set_title('Bollinger Bands')
        axes[0, 1].legend()
        axes[1, 0].plot(self.data.index, self.data['macd'])
        axes[1, 0].set_title('MACD')
        axes[1, 1].plot(self.data.index, self.data['rsi'])
        axes[1, 1].set_title('RSI')
        plt.tight_layout()
        plt.show()


auto_trader = AutoTrader()
auto_trader.collect_stock_prices(
    symbol='AAPL', start_date='2021-01-01', end_date='2021-12-31')
auto_trader.collect_news_data(
    symbol='AAPL', start_date='2021-01-01', end_date='2021-12-31')
auto_trader.collect_social_media_data(
    symbol='AAPL', start_date='2021-01-01', end_date='2021-12-31')
auto_trader.prepare_data()
auto_trader.visualize_data()
model = auto_trader.train_model()
