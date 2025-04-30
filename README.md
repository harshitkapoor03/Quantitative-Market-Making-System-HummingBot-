# SmartTrendVolatilityMaker: An Adaptive Market-Making Strategy for Hummingbot

I've developed this custom market-making strategy from scratch to address what I saw as limitations in existing approaches. It's designed to dynamically adjust to market conditions while managing risk - something I found crucial after testing simpler strategies that would get caught in bad positions during volatile swings.

## Core Features

- **Intelligent Trend Detection**: Combines MACD, RSI, and Bollinger Bands in a weighted scoring system to determine market direction more reliably than single indicators
- **Volatility-Adaptive Order Sizing**: Uses NATR and Bollinger Band width to automatically widen/narrow spreads and adjust order sizes
- **Fractal-Based Retracement Detection**: Identifies key reversal points using Fibonacci levels only when confirmed by fractal patterns
- **Smart Inventory Balancing**: Adjusts trading aggressiveness based on current SOL/FDUSD holdings to prevent overexposure

## How It Works

The strategy continuously evaluates market conditions across multiple dimensions:

1. **Trend Analysis** 
   - MACD crossover signals
   - RSI position relative to overbought/oversold
   - Price position within Bollinger Bands

2. **Volatility Assessment** 
   - Normalized Average True Range (NATR)
   - Bollinger Band width expansion/contraction

3. **Retracement Confirmation** 
   - Requires recent fractal high/low (last 20 candles)
   - Must align with overall trend direction
   - Price must be near key Fib levels (0.382, 0.5, 0.618)

4. **Risk Management** 
   - Portfolio balance monitoring
   - Spread width requirements
   - Volatility dead-zone avoidance

## Technical Implementation

Built as a Hummingbot script with:
- Clean, modular Python architecture
- Efficient use of pandas_ta for technical indicators
- Configurable parameters for different trading pairs
- Comprehensive logging for performance analysis

## Getting Started

### Requirements
- Hummingbot installation
- Python 3.8+
- pandas_ta library

### Installation
1. Place files in your Hummingbot directory:
   ```bash
   cp updated_upload.py ~/hummingbot/scripts/
   cp conf_submissionupdated_1.yml ~/hummingbot/conf/
   ```

2. Install dependencies:
   ```bash
   pip install pandas_ta
   ```

3. Launch strategy:
   ```bash
   start --script updated_upload.py --conf conf_submissionupdated_1.yml
   ```

## Why This Approach Works

After backtesting and live paper trading, I found this multi-factor approach:
- Reduces false signals compared to single-indicator strategies
- Adapts better to changing market regimes
- Provides more consistent performance across different volatility environments
- Manages risk exposure more effectively

## Customization Options

The strategy can be tuned by adjusting:
- Indicator weightings in the scoring system
- Fib retracement sensitivity
- Inventory balance thresholds
- Minimum spread requirements

## Future Improvements

I'm currently working on:
- Adding machine learning for dynamic weight adjustment
- Incorporating order book depth analysis
- Developing a version for spot markets with real funds

