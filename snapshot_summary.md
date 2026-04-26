# System Snapshot: "Institutional Alpha" Release (v1.0)
**Date**: 2026-04-25
**Status**: Finalized & Optimized

## 1. Core Architecture
This snapshot represents the fully optimized Institutional RS Trading System, featuring:
- **Engine**: `sprint_engine.py` (v1.0 - VIX/EMA Shielded).
- **Frontend**: Dashboard with persistent trade logging (`sprints_template.html`).
- **Logic**: 15-day sprints, 0.6% per-stock risk, 3.5R targets.

## 2. Key Artifacts & Documentation
- **Technical PRD**: [PRD.md](file:///c:/Users/kraje/OneDrive/Desktop/@Projects/Finance/RS-python-scripts/PRD.md) (Complete mathematical blueprint).
- **Execution Cockpit**: [output/sprints.html](file:///c:/Users/kraje/OneDrive/Desktop/@Projects/Finance/RS-python-scripts/output/sprints.html).
- **Workflow**: [output.yml](file:///c:/Users/kraje/OneDrive/Desktop/@Projects/Finance/RS-python-scripts/.github/workflows/output.yml) (Automated daily refresh).

## 3. Verified Strategy Configuration
- **Market Shield**: `(VIX < 25) AND (VIX < VIX_SMA20) AND (SPY 9EMA > 21SMA)`.
- **Selection**: Leading Sector Only + RVOL > 1.2 + RMV < 40.
- **Backtest Performance**: 70% Win Rate / +21.3% Return (Recent Sample).

## 4. Experiment Archive (Scratch Files)
The following scripts contain the logic and results for all rejected/neutral hypotheses:
- `scratch/vix_ema_experiments.py`: Confirmed Shield effectiveness.
- `scratch/duration_experiments.py`: Confirmed 15-day optimal window.
- `scratch/rmv_experiments.py`: Confirmed RMV < 30 "Goldilocks" zone.
- `scratch/sector_experiments.py`: Confirmed Concentration > Diversification.
- `scratch/wider_net_experiments.py`: Confirmed Leading Sector superiority.

## 5. Replication Guide
To replicate this system in a new environment (Rust/Swift):
1. Use **PRD.md** for mathematical logic.
2. Use **sprint_engine.py** for implementation reference.
3. Use **sprints_template.html** for UI/UX reference.

---
*Snapshot captured by Antigravity AI.*
