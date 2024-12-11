from typing import Optional

from pydantic import BaseModel, Field


class TradingBotConfig(BaseModel):
    api_key: Optional[str] = Field(None, description="API key for the exchange.")
    api_secret: Optional[str] = Field(None, description="Secret key for the exchange.")
    symbol: str = Field("BTC/USDT", description="Trading pair symbol.")
    timeframe: str = Field("1h", description="Timeframe for market data.")
    initial_balance: float = Field(1000.0, description="Starting balance.")
    trading_amount: float = Field(100.0, description="Amount to trade per iteration.")
    buy_rsi: float = Field(45.0, description="RSI threshold for buying.")
    sell_rsi: float = Field(55.0, description="RSI threshold for selling.")
    iterations: int = Field(100, description="Number of iterations to run.")
    interval: int = Field(60, description="Interval (seconds) between iterations.")
    mock_mode: bool = Field(True, description="Run in mock mode if True.")

    class Config:
        json_schema_extra = {
            "example": {
                "api_key": "your_api_key",
                "api_secret": "your_api_secret",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "initial_balance": 1000.0,
                "trading_amount": 100.0,
                "buy_rsi": 45.0,
                "sell_rsi": 55.0,
                "iterations": 100,
                "interval": 60,
                "mock_mode": True,
            }
        }
