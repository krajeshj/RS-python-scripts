import json
import traceback

filepath = r'c:\Users\kraje\OneDrive\Desktop\@Projects\Finance\RS-python-scripts\output\web_data.json'

try:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    stocks = data.get("stocks", [])
    
    found = []
    for s in stocks:
        if s.get("ticker") in ["META", "APP"]:
            found.append({
                "ticker": s.get("ticker"),
                "canslim": s.get("canslim"),
                "speculative": s.get("speculative")
            })
            
    print(json.dumps(found, indent=2))
except Exception as e:
    print("Error:", e)
    traceback.print_exc()
