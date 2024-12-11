import questionary

from config import load_config, save_config
from models import TradingBotConfig
from smart_trading_bot import SmartTradingBot


def interactive_menu():
    """Main interactive menu for the trading bot."""
    config = load_config() or TradingBotConfig()  # Load config or use default settings
    bot = SmartTradingBot(  # Create the bot instance once
        api_key=config.api_key,
        api_secret=config.api_secret,
        symbol=config.symbol,
        timeframe=config.timeframe,
        initial_balance=config.initial_balance,
        trading_amount=config.trading_amount,
        mock_mode=True,  # Default to mock mode
    )

    while True:
        print("\nSmart Bot Main")
        print("-------------------")
        choice = questionary.select(
            "Choose an option:",
            choices=[
                "Run in live trading mode (last settings)",
                "Run in mockup mode",
                "Show historical plots",
                "Config menu",
                "Exit",
            ],
        ).ask()

        if choice == "Run in live trading mode (last settings)":
            if not bot.exchange.apiKey or not bot.exchange.secret:
                print(
                    "[ERROR] API key and secret are required. Please configure the bot first."
                )
                continue

            bot.mock_mode = False  # Switch to live trading mode
            print(f"Running in live trading mode with settings: {config}")
            bot.run(
                iterations=config.iterations,
                interval=config.interval,
                buy_rsi=config.buy_rsi,
                sell_rsi=config.sell_rsi,
            )

        elif choice == "Run in mockup mode":
            bot.mock_mode = True  # Switch to mockup mode
            print(f"Running in mockup mode with settings: {config}")
            bot.run(
                iterations=config.iterations,
                interval=config.interval,
                buy_rsi=config.buy_rsi,
                sell_rsi=config.sell_rsi,
            )

        elif choice == "Show historical plots":
            print("Generating historical plots...")
            data = bot.fetch_data(limit=100)
            data = bot.add_indicators(data)  # Ensure indicators are added
            bot.plot_trades(data)

        elif choice == "Config menu":
            config_menu()
            # Reload updated configuration
            config = load_config() or TradingBotConfig()
            bot.api_key = config.api_key
            bot.api_secret = config.api_secret
            bot.symbol = config.symbol
            bot.timeframe = config.timeframe
            bot.initial_balance = config.initial_balance
            bot.trading_amount = config.trading_amount

        elif choice == "Exit":
            print("Exiting Smart Bot. Goodbye!")
            break


def config_menu():
    """Configuration menu for the trading bot."""
    config = load_config() or TradingBotConfig()  # Load config or use default settings

    while True:
        print("\nSmart Bot Config")
        print("-------------------")

        choice = questionary.select(
            "Choose a configuration option:",
            choices=[
                f"Set API key (current: {'*' * len(config.api_key) if config.api_key else 'Not Set'})",
                f"Set API secret (current: {'*' * len(config.api_secret) if config.api_secret else 'Not Set'})",
                f"Set trading pair (current: {config.symbol or 'Not Set'})",
                f"Set timeframe (current: {config.timeframe or 'Not Set'})",
                f"Set initial balance (current: {config.initial_balance})",
                f"Set trading amount (current: {config.trading_amount})",
                "Back",
                "Exit",
            ],
        ).ask()

        if choice.startswith("Set API key"):
            config.api_key = questionary.text(
                "Enter your API key:", default=config.api_key or ""
            ).ask()
        elif choice.startswith("Set API secret"):
            config.api_secret = questionary.text(
                "Enter your API secret:", default=config.api_secret or ""
            ).ask()
        elif choice.startswith("Set trading pair"):
            config.symbol = questionary.text(
                "Enter trading pair (e.g., BTC/USDT):", default=config.symbol or ""
            ).ask()
        elif choice.startswith("Set timeframe"):
            config.timeframe = questionary.text(
                "Enter timeframe (e.g., 1h, 1m):", default=config.timeframe or ""
            ).ask()
        elif choice.startswith("Set initial balance"):
            config.initial_balance = float(
                questionary.text(
                    "Enter initial balance:",
                    default=str(config.initial_balance or 1000.0),
                ).ask()
            )
        elif choice.startswith("Set trading amount"):
            config.trading_amount = float(
                questionary.text(
                    "Enter trading amount:", default=str(config.trading_amount or 100.0)
                ).ask()
            )
        elif choice == "Back":
            save_config(config)
            break
        elif choice == "Exit":
            save_config(config)
            print("Exiting Smart Bot. Goodbye!")
            exit(0)

    # Save any updates after exiting the menu loop
    save_config(config)


interactive_menu()
