# Smart Trading Bot

## Overview

Smart Trading Bot is an automated trading bot designed to perform technical analysis and execute trades based on customizable parameters. It supports mock trading to test strategies and real trading via exchange APIs.

## Features

- **Configurable Trading Parameters**: Set API keys, trading pairs, timeframes, RSI thresholds, and more.
- **Mock Mode**: Test strategies without real trading.
- **Interactive Menu**: Easy-to-use menu for configuration and execution.
- **Persisted Configuration**: Save and load settings for quick access.
- **Default Settings**: Run the bot with pre-configured default values.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/smart-trading-bot.git
   cd smart-trading-bot
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the bot as a CLI tool:
   ```bash
   pip install .
   ```

## Usage

Run the bot using the command:
```bash
smart-trading-bot
```

### Menu Options

1. **Run Script with Last Settings**: Execute the bot with the last saved configuration.
2. **Run Script with Defaults**: Execute the bot with default settings.
3. **Config Menu**: Modify bot configuration interactively.
4. **Exit**: Exit the bot.

### Configuration Menu

- Set API key (current: masked).
- Set API secret (current: masked).
- Set trading pair (e.g., BTC/USDT).
- Set timeframe (e.g., 1h).
- Set initial balance.

All configurations are persisted in a file for future use.

## Example Configuration

```json
{
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
  "mock_mode": true
}
```

## Development

To develop or extend the bot:

1. Install the project in editable mode:
   ```bash
   pip install -e .
   ```

2. Run the bot locally:
   ```bash
   python -m src.main
   ```

## Requirements

- Python 3.8+
- Libraries:
  - `pydantic`
  - `questionary`
  - `ccxt`
  - `rich`

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer

This bot is for educational purposes only. Use at your own risk. Ensure compliance with trading regulations in your region.
