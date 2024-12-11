import time
from typing import Any, Optional

import ccxt
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

logger.add(
    "trading_bot.log", rotation="1 MB", level="DEBUG", format="{time} {level} {message}"
)


class SmartTradingBot:
    """
    A professional-grade trading bot for automated market analysis and trading.

    Attributes:
        api_key (str): API key for the exchange.
        api_secret (str): Secret key for the exchange.
        exchange (ccxt.Exchange): Exchange instance for trading.
        symbol (str): Trading pair symbol, e.g., 'ETH/USDT'.
        timeframe (str): Timeframe for market data, e.g., '1m', '1h'.
        initial_balance (float): Starting balance for testing or real trading.
        balance (float): Current balance in base currency (e.g., USDT).
        trading_amount (float): Amount to trade in each iteration.
        mock_mode (bool): If True, runs in mock testing mode.
        trade_history (list): A list of executed trades with details.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str,
        timeframe: str,
        initial_balance: float,
        trading_amount: float,
        mock_mode: bool = True,
    ):
        """
        Initialize the trading bot.

        Args:
            api_key (str): API key for the exchange.
            api_secret (str): Secret key for the exchange.
            symbol (str): Trading pair symbol, e.g., 'ETH/USDT'.
            timeframe (str): Timeframe for market data, e.g., '1m', '1h'.
            initial_balance (float): Starting balance for testing or real trading.
            trading_amount (float): Amount to trade in each iteration.
            mock_mode (bool, optional): Runs in mock testing mode. Defaults to True.
        """
        self.exchange = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": api_secret,
            }
        )
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trading_amount = trading_amount
        self.mock_mode = mock_mode
        self.trade_history = []
        self.trained_model = None
        self.asset_balance = 0  # Track the balance of the traded asset (e.g., BTC)

    def fetch_data(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from the exchange.

        Args:
            limit (int): Number of candles to fetch. Defaults to 100.

        Returns:
            pd.DataFrame: Dataframe containing OHLCV data.
        """
        logger.debug(f"Fetching data for {self.symbol}...")
        bars = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
        df = pd.DataFrame(
            bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def train_model(self, data: pd.DataFrame) -> None:
        """
        Train a machine learning model using historical market data.

        Args:
            data (pd.DataFrame): Dataframe containing historical market data and indicators.
        """
        logger.info("Training decision model...")
        features = data[["RSI", "Bollinger_High", "Bollinger_Low", "MACD", "Signal"]]
        labels = (data["close"].pct_change().shift(-1) > 0).astype(
            int
        )  # 1 for buy, 0 for sell

        X_train, X_test, y_train, y_test = train_test_split(
            features.dropna(), labels.dropna(), test_size=0.2
        )

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        logger.info("Model training completed.")

        self.trained_model = model

    @staticmethod
    def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the market data.

        Args:
            data (pd.DataFrame): Dataframe containing OHLCV data.

        Returns:
            pd.DataFrame: Dataframe with additional indicators.
        """
        logger.debug("Calculating indicators...")
        data["RSI"] = RSIIndicator(data["close"], window=14).rsi()
        bollinger = BollingerBands(data["close"], window=20)
        data["Bollinger_High"] = bollinger.bollinger_hband()
        data["Bollinger_Low"] = bollinger.bollinger_lband()
        macd = MACD(data["close"])
        data["MACD"] = macd.macd()
        data["Signal"] = macd.macd_signal()
        return data

    def decide_action(
        self, data: pd.DataFrame, buy_rsi: float, sell_rsi: float
    ) -> Optional[str]:
        """
        Decide whether to buy, sell, or hold based on indicators.

        Args:
            data (pd.DataFrame): Dataframe containing market data and indicators.
            buy_rsi (float): RSI threshold for buying.
            sell_rsi (float): RSI threshold for selling.

        Returns:
            Optional[str]: 'buy', 'sell', or None for hold.
        """
        last_row = data.iloc[-1]
        logger.debug(
            f"RSI: {last_row['RSI']}, Close: {last_row['close']}, "
            f"Bollinger Low: {last_row['Bollinger_Low']}, Bollinger High: {last_row['Bollinger_High']}"
        )

        if last_row["RSI"] < buy_rsi and self.balance >= self.trading_amount:
            return "buy"
        elif last_row["RSI"] > sell_rsi and self.asset_balance > 0:
            return "sell"
        return None

    def execute_trade(self, action: str, price: float) -> None:
        """
        Execute a trade and update the mock or real balance.

        Args:
            action (str): 'buy' or 'sell'.
            price (float): Current market price.
        """
        logger.info(f"Executing {action.upper()} at price {price}...")
        if self.mock_mode:
            if action == "buy" and self.balance >= self.trading_amount:
                self.balance -= self.trading_amount
                self.asset_balance += self.trading_amount / price
                self.trade_history.append(
                    {"action": "buy", "price": price, "timestamp": pd.Timestamp.now()}
                )
            elif action == "sell" and self.asset_balance > 0:
                self.balance += self.asset_balance * price
                self.asset_balance = 0
                self.trade_history.append(
                    {"action": "sell", "price": price, "timestamp": pd.Timestamp.now()}
                )
            else:
                logger.warning("Trade not executed. Insufficient funds or assets.")

    def summarize_trades(self) -> dict[str, Any]:
        """
        Summarize trade history and calculate profit/loss.

        Returns:
            dict: Summary of profit/loss and trade statistics.
        """
        logger.info("Summarizing trades...")

        realized_profit = sum(
            (trade["price"] if trade["action"] == "sell" else -trade["price"])
            for trade in self.trade_history
        )

        summary = {
            "total_trades": len(self.trade_history),
            "realized_profit": realized_profit,
            "balance": self.balance,
        }

        logger.info(f"Trade Summary: {summary}")
        return summary

    def plot_trades(self, data: pd.DataFrame) -> None:
        """
        Plot trades along with market data and indicators.

        Args:
            data (pd.DataFrame): Dataframe containing market data.
        """
        # Add indicators if they are missing
        if "Bollinger_High" not in data or "Bollinger_Low" not in data:
            data = self.add_indicators(data)

        logger.info("Plotting trades...")
        plt.figure(figsize=(14, 8))

        # Plot the closing price and Bollinger Bands
        plt.plot(data["timestamp"], data["close"], label="Close Price", alpha=0.7)
        plt.plot(
            data["timestamp"],
            data["Bollinger_High"],
            label="Bollinger High",
            linestyle="--",
            alpha=0.5,
        )
        plt.plot(
            data["timestamp"],
            data["Bollinger_Low"],
            label="Bollinger Low",
            linestyle="--",
            alpha=0.5,
        )

        # Highlight buy and sell signals
        for trade in self.trade_history:
            if trade["action"] == "buy":
                plt.scatter(
                    trade["timestamp"],
                    trade["price"],
                    color="green",
                    label="Buy Signal",
                    alpha=0.8,
                    marker="^",
                )
            elif trade["action"] == "sell":
                plt.scatter(
                    trade["timestamp"],
                    trade["price"],
                    color="red",
                    label="Sell Signal",
                    alpha=0.8,
                    marker="v",
                )

        plt.title(f"Trading Strategy Visualization ({self.symbol})")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_balance_over_time(self) -> None:
        """
        Plot the balance over time based on the trade history.
        """
        logger.info("Plotting balance over time...")
        balance_over_time = [self.initial_balance]
        timestamps = [None]  # Placeholder for the initial balance entry

        # Calculate balance evolution
        for trade in self.trade_history:
            if trade["action"] == "buy":
                balance_over_time.append(balance_over_time[-1] - self.trading_amount)
            elif trade["action"] == "sell":
                balance_over_time.append(balance_over_time[-1] + self.trading_amount)
            timestamps.append(trade["timestamp"])

        plt.figure(figsize=(12, 6))
        plt.plot(
            timestamps[1:], balance_over_time[1:], label="Balance Over Time", marker="o"
        )
        plt.title("Mock Balance Evolution")
        plt.xlabel("Time")
        plt.ylabel("Balance (USDT)")
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def plot_performance(historical_data: pd.DataFrame) -> None:
        plt.figure(figsize=(12, 8))
        historical_data["Cumulative Return"] = (
            historical_data["close"].pct_change() + 1
        ).cumprod()
        plt.plot(
            historical_data["timestamp"],
            historical_data["Cumulative Return"],
            label="Cumulative Return",
        )
        plt.title("Performance Over Time")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid()
        plt.show()

    def optimize_strategy(self, historical_data: pd.DataFrame) -> None:
        logger.info("Optimizing strategy parameters...")
        best_params = {}
        best_performance = float("-inf")

        # Example grid search
        for rsi_threshold in range(20, 50, 5):
            for bb_window in range(15, 30, 5):
                self.params = {"RSI_threshold": rsi_threshold, "BB_window": bb_window}
                historical_data = self.add_indicators(historical_data)
                summary = self.backtest(historical_data)
                performance = summary["net_profit"]

                if performance > best_performance:
                    best_performance = performance
                    best_params = self.params

        logger.info(f"Best parameters found: {best_params}")
        self.params = best_params  # Update bot with optimal parameters

    def backtest(
        self, historical_data: pd.DataFrame, optimize: bool = False
    ) -> dict[str, Any]:
        logger.info("Starting backtest with historical data...")

        # Add indicators
        historical_data = self.add_indicators(historical_data)

        # Loop through data for backtesting
        for i in range(len(historical_data)):
            data = historical_data.iloc[: i + 1]
            action = self.decide_action(data)
            if action:
                price = data["close"].iloc[-1]
                self.execute_trade(action, price)
                self.trade_history[-1]["timestamp"] = data["timestamp"].iloc[
                    -1
                ]  # Add timestamp

        # Summarize results
        summary = self.summarize_trades()

        # Optimize strategy if requested
        if optimize:
            self.optimize_strategy(historical_data)

        # Plot results
        self.plot_trades(historical_data)
        self.plot_balance_over_time()

        # Plot cumulative performance
        self.plot_performance(historical_data)

        return summary

    def run(
        self, iterations: int, interval: int, buy_rsi: float, sell_rsi: float
    ) -> None:
        """
        Main execution loop for the trading bot.

        Args:
            iterations (int): Number of iterations to run.
            interval (int): Interval (seconds) between iterations.
            buy_rsi (float): RSI threshold for buying.
            sell_rsi (float): RSI threshold for selling.
        """
        logger.info(
            f"Starting bot for {iterations} iterations in {'mock' if self.mock_mode else 'live'} mode."
        )
        data = None  # Keep a reference to the latest data for plotting

        for i in range(iterations):
            try:
                # Fetch data
                data = self.fetch_data()
                if data is None or data.empty:
                    logger.warning(
                        f"No data fetched during iteration {i + 1}. Skipping."
                    )
                    continue

                # Add indicators
                data = self.add_indicators(data)

                # Decide action and execute trade if applicable
                action = self.decide_action(data, buy_rsi, sell_rsi)
                if action:
                    price = data["close"].iloc[-1]
                    self.execute_trade(action, price)
                    if self.trade_history:  # Ensure trade_history is not empty
                        self.trade_history[-1]["timestamp"] = data["timestamp"].iloc[-1]

                logger.debug(f"Iteration {i + 1} completed.")
            except Exception as e:
                logger.error(f"Error during iteration {i + 1}: {e}")

            # Wait before the next iteration
            logger.debug(f"Sleeping for {interval} seconds...")
            time.sleep(interval)

        # Summarize and plot performance
        if data is not None and not data.empty:
            summary = self.summarize_trades()
            logger.info("Summary of trades:")
            for key, value in summary.items():
                logger.info(f"{key}: {value}")

            self.plot_trades(data)
            self.plot_balance_over_time()
            self.plot_performance(data)
        else:
            logger.warning("No data available for plotting.")
