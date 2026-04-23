import pandas as pd
import json

df = pd.read_csv('output/rs_stocks.csv')

with open('output/web_data.json', 'r') as f:
    web_data = json.load(f)

# Combine stocks, all_stocks, and pulse (for ETFs)
stocks = web_data.get('stocks', []) + web_data.get('all_stocks', []) + web_data.get('pulse', [])

seen = set()
unique_stocks = []
for s in stocks:
    if s['ticker'] not in seen:
        seen.add(s['ticker'])
        unique_stocks.append(s)

df_csv = df.set_index('Ticker')

long_term = []
swing_trade = []

for s in unique_stocks:
    ticker = s['ticker']
    price = s.get('price', 0)
    if price == 0: continue
    
    c = s.get('canslim', {})
    score = sum([c.get('c',False), c.get('a',False), c.get('n',False), c.get('s',False), c.get('i',False), c.get('l',False), c.get('m',False)])
    rmv = s.get('rmv', 100)
    hl = s.get('highlights', '')
    rs = s.get('rs', 0)
    rs_1w = s.get('rs_1w_pct', 0)
    rs_1m = s.get('rs_1m_pct', 0)
    sector = s.get('sector', '')
    is_etf = 'ETF' in s.get('industry', '') or sector == 'Market Pulse'
    
    # -----------------------------
    # LONG TERM CRITERIA
    # -----------------------------
    # High fundamental score or ETF, stable volatility, leading RS
    is_lt_candidate = False
    if is_etf and rs > 60:
        is_lt_candidate = True
    elif score >= 4 and rmv <= 50 and rs >= 70:
        is_lt_candidate = True
        
    if is_lt_candidate:
        long_term.append({
            's': s,
            'score': score,
            'rs': rs,
            'rmv': rmv
        })
        
    # -----------------------------
    # SWING TRADE CRITERIA
    # -----------------------------
    # Imminent setup (VCP/Flip), positive momentum, very low volatility (RMV)
    if rmv <= 35 and rs >= 60 and rs_1w >= rs_1m:
        if 'VCP Contraction' in hl or 'Flip Setup' in hl or 'Not Extended' in hl:
            swing_trade.append({
                's': s,
                'momentum': rs_1w - rs_1m,
                'rmv': rmv,
                'setup': 1 if 'VCP Contraction' in hl or 'Flip Setup' in hl else 0
            })

# Sort Long Term by RS (desc), CANSLIM (desc)
long_term = sorted(long_term, key=lambda x: (-x['rs'], -x['score']))
top_lt = [x['s'] for x in long_term[:5]]

# Sort Swing by Setup (desc), RMV (asc), Momentum (desc)
swing_trade = sorted(swing_trade, key=lambda x: (-x['setup'], x['rmv'], -x['momentum']))
top_swing = [x['s'] for x in swing_trade[:5]]

def print_table(title, stock_list, is_swing):
    print(f"### {title}")
    print('| Ticker | Name | Type/Sector | Price | Stop Loss | Target (R=3) | Risk | Setup/Highlights |')
    print('|---|---|---|---|---|---|---|---|')
    for s in stock_list:
        ticker = s['ticker']
        price = s['price']
        
        try:
            atr_pct = df_csv.loc[ticker, 'ATR% (14)']
            atr_val = price * atr_pct
        except:
            atr_val = price * 0.03 # fallback 3%
            
        if is_swing:
            # Swing: Tighter stop. Max 6%. Or 1 ATR.
            stop_dist = min(1.0 * atr_val, price * 0.06)
            stop_loss = round(price - stop_dist, 2)
            target = round(price + (stop_dist * 3), 2) # 3R target
            risk_str = f"Max 6% or 1ATR"
        else:
            # Long term: Wider stop. 2 ATR.
            stop_dist = 2.0 * atr_val
            stop_loss = round(price - stop_dist, 2)
            target = round(price + (stop_dist * 3), 2) # 3R target
            risk_str = f"2 ATR"
            
        sl_pct = ((price - stop_loss) / price) * 100
        
        name = s.get('name','')[:15].replace('|', '')
        sector = s.get('sector','')[:12].replace('|', '')
        if s.get('industry') == 'Index/ETF': sector = 'ETF'
        
        hl = s.get('highlights','')
        if not hl and 'trend' in s: hl = s.get('trend')
        
        print(f"| **{ticker}** | {name} | {sector} | ${price} | ${stop_loss} (-{sl_pct:.1f}%) | ${target} | {risk_str} | {hl} |")
    print()

print_table("Long-Term Hold Candidates (CANSLIM + Leading RS + ETFs)", top_lt, is_swing=False)
print_table("Short-Term Swing Trades (1-2 Weeks, Tight SL, High Momentum)", top_swing, is_swing=True)
