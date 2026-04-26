# PRD: Institutional Relative Strength (RS) Trading System

## 1. Product Overview
The **Institutional RS Trading System** is a high-velocity swing trading platform designed to identify and exploit momentum in market leaders. It utilizes a "Sprint" methodology (15-day cycles) combined with institutional-grade risk management and market regime filtering.

## 2. Core Strategy: "The Sprint"
- **Duration**: 15 Trading Days (3 Weeks).
- **Frequency**: New sprint begins every 15 days, or when the previous sprint completes.
- **Selection Universe**: Top 40 stocks by Relative Strength (RS) within leading sectors.
- **Filters**:
    - **Sector Alignment**: Must belong to a "Leading" sector ETF (XLK, XLE, etc.) according to RRG analysis.
    - **Institutional Flow**: Relative Volume (RVOL) > 1.2x.
    - **Volatility Control**: RMV (Relative Measured Volatility) < 40 (Optimal: 30).
    - **Earnings Shield**: No earnings announcements within the next 15 days.

## 3. Risk Management Engine
- **Equal Opportunity Risk**: 
    - Total Sprint Risk: 3.0% of Portfolio Capital.
    - Per Trade Risk: 0.6% of Portfolio Capital (Total Risk / 5).
- **Exit Logic**:
    - **Terminal Velocity (Target)**: 3.5R (Profit = 3.5x Risk).
    - **Hard Stop**: 21-Day Low or 15-Day Low (whichever is tighter, but not more than 8%).
- **Position Sizing**: `Shares = (Risk Budget) / (Entry Price - Stop Price)`.

## 4. Market Regime Shields (The "Gatekeeper")
Trades are only executed if the **Market Shield** is "FAVORABLE":
- **VIX Momentum**: VIX must be < 25 AND VIX must be < its 20-day SMA (Fear is declining).
- **Trend Alignment**: SPY 9-day EMA must be > 21-day SMA.

## 5. System Architecture
- **Engine (`sprint_engine.py`)**: The core logic that processes market data, applies filters, and generates trade recommendations.
- **Data Pipeline**: Uses `yfinance` for real-time and historical price data.
- **Frontend Dashboards**:
    - **Sprints (`sprints.html`)**: The primary execution cockpit with OTOCO order parameters.
    - **Sector Rotation (`rotation.html`)**: RRG trajectories for major ETFs.
    - **Industry Momentum (`industries.html`)**: Top-down heatmaps.
    - **Trade Journal**: Browser-persistent (localStorage) logging for active trades.

## 6. Key Performance Indicators (KPIs)
- **Win Rate**: Target > 50% (Current Recent: 70%).
- **Total Return**: Target > 20% Annualized alpha over SPY.
- **Sharpe Ratio**: Target > 1.5.

## 7. Operational Workflow
1. **Daily Quick Scan**: Automated via GitHub Actions to update RRG and RS scores.
2. **Sprint Generation**: On Day 1, the `sprint_engine.py` identifies 5 high-conviction targets.
3. **Execution**: Trader enters OTOCO (One-Cancels-the-Other) orders in Power E*TRADE using the provided parameters.
4. **Monitoring**: Track active trades in the dashboard journal; exit on Target or Stop.

---
*Last Updated: 2026-04-25*
