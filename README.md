# Advanced Crypto Trading Bot with LSTM and Technical Indicators

This Python script is an advanced crypto trading bot that uses **LSTM (Long Short-Term Memory)** neural networks and **technical indicators** to predict price movements and generate buy/sell signals. It fetches historical data from Binance, computes advanced technical indicators, trains an LSTM model, and implements a scalping strategy with risk management.

## Features
- **Historical Data Fetching**: Fetches real-time historical data from Binance.
- **Technical Indicators**: Computes advanced indicators like RSI, MACD, Bollinger Bands, ATR, ADX, and Stochastic RSI.
- **LSTM Model**: Predicts future price movements using a deep learning model.
- **Scalping Strategy**: Implements a scalping strategy with buy/sell signals based on predicted price changes.
- **Risk Management**: Includes a 0.5% threshold for buy/sell signals to minimize risk.

## Prerequisites
Before using this script, ensure you have the following:
1. **Python 3.x** installed on your machine.
2. Required Python libraries installed (see `requirements.txt`).
3. A stable internet connection (for fetching data from Binance).

## Installation
1. Clone this repository (if applicable):
   ```bash
   git clone https://github.com/2Px1ONE/trading-bot.git
   cd crypto-trading-bot
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the script:
   ```bash
   python trading_bot.py
   ```
2. The script will:
   - Fetch historical data from Binance.
   - Compute technical indicators.
   - Train an LSTM model on the data.
   - Continuously monitor the market and generate buy/sell signals every 30 seconds.

## Technical Indicators Used
- **Moving Averages (MA5, MA10)**: Short-term and long-term trends.
- **RSI (Relative Strength Index)**: Momentum indicator.
- **Bollinger Bands**: Volatility indicator.
- **MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator.
- **Stochastic RSI**: Momentum oscillator.
- **ATR (Average True Range)**: Volatility measure.
- **ADX (Average Directional Index)**: Trend strength indicator.

## LSTM Model Architecture
The LSTM model consists of:
- **3 LSTM layers** with 50 units each.
- **Dropout layers** for regularization.
- **Batch Normalization** for faster convergence.
- **Dense layers** for output prediction.

## Scalping Strategy
The bot uses a scalping strategy with the following rules:
- **Buy Signal**: If the predicted price is **0.5% higher** than the current price.
- **Sell Signal**: If the predicted price is **0.5% lower** than the current price.
- **Wait**: If no significant price movement is predicted.

## Example Output
```
📡 Updating Data...
📈 Current Price: 1500.00 USDT | 🔮 Predicted Price: 1510.00 USDT
🟢 **Buy Signal!**
⌛️ Waiting 30 seconds before next update...

📡 Updating Data...
📈 Current Price: 1505.00 USDT | 🔮 Predicted Price: 1495.00 USDT
🔴 **Sell Signal!**
⌛️ Waiting 30 seconds before next update...
```

## Important Notes
- **Risk Disclaimer**: This bot is for educational purposes only. Use it at your own risk. The author is not responsible for any financial losses.
- **Infura Rate Limits**: Be aware of Binance's API rate limits. If you encounter issues, consider using a premium API key.
- **Model Training**: The LSTM model is trained on historical data. For better performance, retrain the model periodically with updated data.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Support
If you find this project helpful, consider giving it a ⭐ on GitHub. For questions or issues, please open an issue in the repository.


