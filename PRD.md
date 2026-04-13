# Product Requirements Document (PRD) - Rajesh's Stock Screener

## 1. Vision & Core Objective
To provide a high-fidelity, institutional-grade workspace for tracking Relative Strength (RS) and Sector/Stock Rotation. The system must catch trend exhaustion ("Rotating Out") early using advanced weighted momentum and predictive visual cues.

---

## 2. Dashboard Ecosystem (Features)

### A. Sector Rotation Dashboard (`rotation.html`)
*   **Visual Map:** RRG-style (Relative Rotation Graph) using Native SVG. quadrants: Leading (Green), Weakening (Orange), Lagging (Red), Improving (Blue).
*   **Breadcrumb Trails:** Trajectories showing the path of the sector over the last 5-10 weeks with increasing opacity.
*   **Multi-Tab Analytics:**
    1.  **Rotation Map:** The primary trajectory graph.
    2.  **Rank Signals:** A matrix view showing Sector Name, Signal (Quadrant), Momentum Score, and Trend Status (Strength/Weakness).
    3.  **Fund Flows:** Visualization of institutional money flow in $M (inferred/imported).
*   **Data Source:** Hardcoded high-fidelity baseline for core sectors (XLK, XLE, etc.).

### B. Stock Rotation Dashboard (`stock_rotation.html`)
*   **Neural Momentum Radar:** Dynamic RRG tracking for individual stocks.
*   **Front-Weighted Lookback:** Momentum calculations carry a **60% weight on the most recent 3 days** of action. This is critical for catching trend-exits quickly.
*   **Predictive Alpha Curls:** Dashed projection lines using **Quadratic Acceleration** logic. If a stock’s velocity is decelerating, the projection "curls" towards the Lagging quadrant.
*   **Filtering Controls:**
    *   **RS Slider:** Real-time filtering by Relative Strength score (0-99).
    *   **VCP/Minervini:** Boolean filter for stocks meeting Stage 2 / Mark Minervini criteria.
    *   **Leaders Only:** Filter for tickers with positive trend alignment.
*   **Traversal Trails:** High-fidelity "circular crumbs" showing the historical path.

### C. Sector Stage Analysis (`stages.html`)
*   **Market Breath:** Top-level bar showing the % distribution of the total market across Stages 1, 2, 3, and 4.
*   **Sector-Specific health:** Health badges and stock distribution bars for every major sector.

### D. Landing Page Dashboard (`index.html`)
*   **Visual Ticker Banner:** Scrolled institutional status bar at top.
*   **Stock Cards:** Comprehensive metadata (RS, Sector, Industry, CANSLIM status, ATR, and RMV).

---

## 3. Technical Architecture (Zero-Regression Policy)

### **Frontend Development Rules:**
*   **Zero-Babel/Zero-React:** To prevent "Blank Page" regressions, all dashboards MUST use **Native Vanilla JavaScript**.
*   **No Heavy Libraries:** Charting must be done via **Native SVG** or Canvas to ensure instant load and cross-browser reliability.
*   **Persistence:** Use `?v=timestamp` on JSON fetches to bust cache.

### **Backend Data Pipeline:**
*   **relative-strength.py:** Core data fetcher.
*   **rs_ranking.py:** Logic engine for calculating RS percentiles and exporting `web_data.json`.
*   **seed_history.py:** Generates `stock_history.json` for path/trail rendering.

### **GitHub Actions (Scan Scopes):**
1.  **Quick Scan (Commit/Push Trigger):** Scans only **Nasdaq 100** tickers + Manual Tips. Designed for speed and instant dashboard feedback.
2.  **Full Scan (Scheduled Trigger):** Scans the full **5,000+ ticker** universe.

---

## 4. Key Algorithms & UI Logic

| Feature | Logic / Math |
| :--- | :--- |
| **RS Score** | Mansfield-style RS vs SPY, converted to a 0-99 percentile. |
| **Momentum** | `WeightedVelocity = (T0*0.6 + T-1*0.3 + T-2*0.1)`. |
| **Curl Logic** | `2nd Derivative (Acceleration)` determines the bend of the dashed projection. |
| **Design Style** | Deep Dark Mode (`#06060f`), Outfit/JetBrains fonts, Glassmorphism. |

---

## 5. Maintenance Checklist (Aids AI in Context)
*   [ ] Does the page use `<script type="text/babel">`? (If yes: **Fail**. Convert to Vanilla).
*   [ ] Does the commit scan the full 5000 universe? (If yes: **Fail**. Ensure Quick Scan is on push).
*   [ ] Are breadcrumbs visible on Stock Rotation?
*   [ ] Does the "Fund Flow" tab exist on Sector Rotation?
*   [ ] Is the header sticky on all pages?
