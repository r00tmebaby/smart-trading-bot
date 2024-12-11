from setuptools import find_packages, setup

setup(
    name="smart-trading-bot",
    version="0.1.0",
    description="A smart trading bot for crypto and stock markets.",
    author="Your Name",
    author_email="r00tme@abv.bg",
    url="https://github.com/your-repo/smart-trading-bot",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "ccxt",
        "ta",
        "backtrader",
        "scikit-learn",
        "matplotlib",
        "loguru",
        "questionary",
        "pydantic>=2.0",
        "typer[all]",
    ],
    entry_points={
        "console_scripts": [
            "smart-trading-bot=main:interactive_menu",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,  # Ensures non-Python files are included
)
