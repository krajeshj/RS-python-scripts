# Technical PRD: Institutional Relative Strength (RS) Trading System

## 1. Product Overview
The **Institutional RS Trading System** is a high-velocity swing trading engine designed to identify and exploit momentum in market leaders. It utilizes a "Sprint" methodology (15-day cycles) combined with deterministic risk management and market regime filtering.

---

## 2. Core Mathematical Formulas

### 2.1 Relative Strength (RS) Score
Calculated as the stock's percentage performance relative to the benchmark (SPY) over a 63-day (3-month) lookback.
- **Formula**: `RS = (Price_Today / Price_63d_Ago) / (SPY_Today / SPY_63d_Ago)`
- **Universe**: Top 40 tickers by RS within Leading Sectors are ranked for selection.

### 2.2 RRG Quadrant Analysis (Relative Rotation Graph)
Uses two metrics: **RS-Ratio** (Trend) and **RS-Momentum** (Velocity).
- **RS-Ratio**: 100 + ((SMA(14, RS_Line) - SMA(14, SMA(14, RS_Line))) / StdDev(14, RS_Line)) * 5
- **RS-Momentum**: 100 + ((ROC(10, RS-Ratio) - SMA(10, ROC(10, RS-Ratio))) / StdDev(10, ROC(10, RS-Ratio))) * 5
- **Quadrants**:
    - **Leading**: Ratio > 100, Momentum > 100
    - **Weakening**: Ratio > 100, Momentum < 100
    - **Lagging**: Ratio < 100, Momentum < 100
    - **Improving**: Ratio < 100, Momentum > 100

### 2.3 RMV (Relative Measured Volatility)
Used to identify VCP (Volatility Contraction Pattern) "coils."
- **Formula**: `RMV = (SMA(20, High - Low) / Price_Today) * 700`
- **Threshold**: RMV < 40 (Aggressive), RMV < 30 (Institutional Quality).

---

## 3. Data Schemas & State Management

### 3.1 Input: `price_history.json`
```json
{
  "TICKER": {
    "candles": [
      {"datetime": 1714000000, "open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0, "volume": 1000000}
    ]
  }
}
```

### 3.2 Output: `sprint_data.json`
Used by the dashboard to render cards.
```json
{
  "generated": "2026-04-25 18:36 UTC",
  "summary": {
    "portfolio_start": 17000.0,
    "portfolio_current": 20204.16,
    "win_rate": 68.6,
    "sharpe_ratio": 2.1
  },
  "current_sprint": {
    "start_date": "2026-04-24",
    "end_date": "2026-05-15",
    "market": "FAVORABLE",
    "orders": [
      {
        "ticker": "NVDA",
        "price": 850.0,
        "shares": 10,
        "buy_stop": 850.0,
        "buy_limit": 851.70,
        "target": 950.0,
        "stop": 820.0
      }
    ]
  }
}
```

### 3.3 Persistence: `localStorage`
Key: `sprint_{startDate}_{ticker}`
Value: `{"bought": true, "sold": false, "price": 850.25, "date": "2026-04-25"}`

---

## 4. Algorithmic Logic Flows

### 4.1 Market Regime Shield (The "Gatekeeper")
**Rule**: Execute `RUN_SPRINT` only if `IS_FAVORABLE == TRUE`.
1. Fetch latest 50 days of `SPY` and `^VIX`.
2. Calculate `SPY_9EMA` and `SPY_21SMA`.
3. Calculate `VIX_SMA20`.
4. `IS_FAVORABLE = (SPY_9EMA > SPY_21SMA) AND (VIX < 25) AND (VIX < VIX_SMA20)`.

### 4.2 Selection Algorithm
1. Filter all stocks in `web_data.json` for `Sector == Leading`.
2. Discard stocks with `Price < $15`.
3. Discard stocks with `Earnings_Date < Today + 15 days`.
4. Filter for `RVOL (Today Vol / 20d Avg Vol) > 1.2`.
5. Filter for `RMV < 30`.
6. Sort remaining by `RS_Score` descending.
7. Select Top 5 as `Current_Sprint_Orders`.

---

## 5. Risk Management & Execution Logic

### 5.1 OTOCO Order Parameters (Power E*TRADE)
- **Primary Order**: Buy Stop at `Price_Today`.
- **Buy Limit Offset**: `Price_Today * 1.002` (0.2% slippage buffer).
- **Secondary Order (Sell)**: OTO (One-Triggers-Other) containing:
    - **Limit (Target)**: `Price_Today + (Risk_Per_Share * 3.5)`.
    - **Stop (Loss)**: `Max(21d_Low, Price_Today * 0.92)`.
    - **Stop Limit Offset**: `Stop_Price * 0.998`.

### 5.2 Fractional Position Sizing
- `Sprint_Risk_Budget = Total_Capital * 0.03` (3% Total risk per sprint).
- `Risk_Per_Stock = Sprint_Risk_Budget / 5`.
- `Shares = Floor(Risk_Per_Stock / (Buy_Price - Stop_Price))`.

---

## 6. Data Availability Architecture

### 6.1 Priority Fetching (Resilience Pattern)
To ensure the dashboard is never stale, all client-side pages implement a dual-fetch fallback:
1.  **Primary**: `web_data.json` (Daily Full Scan - triggered at market close).
2.  **Fallback**: `quick_web_data.json` (Intraday/Quick Scan - triggered manually or on code push).

### 6.2 Automation Schedule
- **Full Scan**: Weekdays at 21:07 UTC (4:07 PM ET).
- **Quick Scan**: On-demand (User-initiated via Dashboard).

---

## 7. Versioning & Snapshots
- **Current Stable**: `v1.0-institutional-alpha`
- **Definition of Done**:
    - [x] Automated Daily Refresh via GitHub Actions.
    - [x] Sharpe Ratio > 2.0 displayed on Sprints.
    - [x] Persistent Active Trade Log via localStorage.
    - [x] Data-aware fallbacks (Daily > Quick) across all 7 dashboard pages.

---
*Last Updated: 2026-04-26 (v1.0 Final)*
