import json
import pandas as pd
import numpy as np

with open('data/price_history.json', 'r') as f:
    all_data = json.load(f)

# The picks from the earlier screener run
long_term_picks = ['SNDK', 'WDC', 'STX', 'MRVL', 'MU']
swing_picks = ['CSX', 'TXN', 'AVGO', 'KLAC', 'MCHP']

PORTFOLIO = 17000.00
MAX_PORTFOLIO_LOSS_PCT = 0.03  # 3% max loss on total portfolio
MAX_PORTFOLIO_LOSS = PORTFOLIO * MAX_PORTFOLIO_LOSS_PCT  # $510

LOOKBACK_DAYS = 21

def get_prices(ticker, offset_from_end):
    """Get close price at offset trading days from end."""
    candles = all_data.get(ticker, {}).get('candles', [])
    if not candles or len(candles) < offset_from_end + 1:
        return None
    return candles[-(offset_from_end + 1)]['close']

def get_high_low_range(ticker, start_offset, end_offset=0):
    """Get high and low over a range of trading days."""
    candles = all_data.get(ticker, {}).get('candles', [])
    if not candles:
        return None, None
    subset = candles[-(start_offset + 1):len(candles) - end_offset if end_offset > 0 else len(candles)]
    highs = [c['high'] for c in subset]
    lows = [c['low'] for c in subset]
    return max(highs) if highs else None, min(lows) if lows else None

def calc_atr(ticker, at_offset=0):
    """Calculate 14-day ATR at a given offset from end."""
    candles = all_data.get(ticker, {}).get('candles', [])
    if not candles or len(candles) < at_offset + 15:
        return None
    end = len(candles) - at_offset
    start = end - 15
    subset = candles[start:end]
    trs = []
    for i in range(1, len(subset)):
        h = subset[i]['high']
        l = subset[i]['low']
        pc = subset[i-1]['close']
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return np.mean(trs) if trs else None

def backtest_bucket(name, tickers, is_swing):
    print(f"\n### {name}")
    print(f"Portfolio: ${PORTFOLIO:,.2f} | Max Loss: ${MAX_PORTFOLIO_LOSS:,.2f} (3%)")
    print()
    
    # Position sizing: divide max loss equally among picks
    num_picks = len(tickers)
    max_risk_per_pick = MAX_PORTFOLIO_LOSS / num_picks
    
    results = []
    total_invested = 0
    total_pnl = 0
    
    for ticker in tickers:
        entry_price = get_prices(ticker, LOOKBACK_DAYS)
        current_price = get_prices(ticker, 0)
        atr = calc_atr(ticker, LOOKBACK_DAYS)
        
        if entry_price is None or current_price is None or atr is None:
            print(f"  {ticker}: Insufficient data, skipping")
            continue
        
        # Calculate stop loss and target based on trade type
        if is_swing:
            # Swing: tight stop, 1 ATR or 6% max
            stop_dist = min(atr, entry_price * 0.06)
            target_dist = stop_dist * 3  # 3R
        else:
            # Long term: wider stop, 2 ATR
            stop_dist = 2 * atr
            target_dist = stop_dist * 3  # 3R
        
        stop_loss = round(entry_price - stop_dist, 2)
        target = round(entry_price + target_dist, 2)
        
        # Position sizing: how many shares can we buy with max_risk_per_pick?
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            continue
        shares = int(max_risk_per_pick / risk_per_share)
        if shares == 0:
            shares = 1
        
        position_cost = shares * entry_price
        
        # Check if position cost exceeds available capital
        if total_invested + position_cost > PORTFOLIO:
            shares = int((PORTFOLIO - total_invested) / entry_price)
            if shares <= 0:
                continue
            position_cost = shares * entry_price
        
        # Simulate OCO execution over the 21-day window
        candles = all_data[ticker]['candles']
        entry_idx = len(candles) - LOOKBACK_DAYS - 1
        
        exit_price = current_price
        exit_reason = "OPEN (held)"
        days_held = LOOKBACK_DAYS
        
        for d in range(1, LOOKBACK_DAYS + 1):
            idx = entry_idx + d
            if idx >= len(candles):
                break
            day_low = candles[idx]['low']
            day_high = candles[idx]['high']
            day_close = candles[idx]['close']
            
            # Check stop loss hit first (conservative: assume stop hit before target on gap days)
            if day_low <= stop_loss:
                exit_price = stop_loss
                exit_reason = "STOP HIT"
                days_held = d
                break
            # Check target hit
            if day_high >= target:
                exit_price = target
                exit_reason = "TARGET HIT"
                days_held = d
                break
        
        pnl = (exit_price - entry_price) * shares
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        
        total_invested += position_cost
        total_pnl += pnl
        
        results.append({
            'ticker': ticker,
            'shares': shares,
            'entry': entry_price,
            'stop': stop_loss,
            'target': target,
            'exit': exit_price,
            'reason': exit_reason,
            'days': days_held,
            'cost': position_cost,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        })
    
    # Print table
    print('| Ticker | Shares | Entry | Stop | Target | Exit | Result | Days | Cost | P&L | P&L % |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for r in results:
        color = '+' if r['pnl'] >= 0 else ''
        print(f"| **{r['ticker']}** | {r['shares']} | ${r['entry']:.2f} | ${r['stop']:.2f} | ${r['target']:.2f} | ${r['exit']:.2f} | {r['reason']} | {r['days']} | ${r['cost']:,.2f} | {color}${r['pnl']:,.2f} | {color}{r['pnl_pct']:.1f}% |")
    
    print(f"\n**Total Invested:** ${total_invested:,.2f}")
    print(f"**Total P&L:** ${total_pnl:,.2f} ({(total_pnl/PORTFOLIO)*100:.2f}% of portfolio)")
    print(f"**Final Value:** ${total_invested + total_pnl:,.2f}")
    
    return total_pnl

print("=" * 80)
print("BACKTEST: 21-Day Lookback from Today's Data")
print("E*TRADE OCO Order Simulation (One-Cancels-Other: Stop + Limit)")
print("=" * 80)

pnl_lt = backtest_bucket("LONG-TERM HOLD (2 ATR Stop, 3R Target)", long_term_picks, is_swing=False)
pnl_sw = backtest_bucket("SWING TRADE 1-2 Weeks (1 ATR/6% Stop, 3R Target)", swing_picks, is_swing=True)

print("\n" + "=" * 80)
print(f"COMBINED PORTFOLIO P&L: ${pnl_lt + pnl_sw:,.2f} ({((pnl_lt + pnl_sw)/PORTFOLIO)*100:.2f}%)")
print(f"Starting: ${PORTFOLIO:,.2f} → Ending: ${PORTFOLIO + pnl_lt + pnl_sw:,.2f}")
print("=" * 80)
