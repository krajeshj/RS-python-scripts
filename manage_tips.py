import yaml
import os
import argparse

DIR = os.path.dirname(os.path.realpath(__file__))
TIPS_FILE = os.path.join(DIR, "tips.yaml")

def load_tips():
    if not os.path.exists(TIPS_FILE):
        return []
    with open(TIPS_FILE, 'r') as f:
        tips = yaml.safe_load(f)
    return tips if tips else []

def save_tips(tips):
    with open(TIPS_FILE, 'w') as f:
        yaml.dump(tips, f, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser(description="Manage manual stock tips in tips.yaml")
    parser.add_argument("--add", help="Comma-separated list of tickers to add")
    parser.add_argument("--delete", help="Comma-separated list of tickers to delete")
    parser.add_argument("--source", default="Manual Entry", help="Source name for added tips (default: Manual Entry)")
    
    args = parser.parse_args()
    
    tips = load_tips()
    existing_tickers = {t['ticker'] for t in tips}
    
    changed = False
    
    if args.add:
        added_count = 0
        new_tickers = [t.strip().upper() for t in args.add.split(',')]
        for ticker in new_tickers:
            if ticker and ticker not in existing_tickers:
                tips.append({"ticker": ticker, "source": args.source})
                existing_tickers.add(ticker)
                added_count += 1
                changed = True
        print(f"Added {added_count} new tips.")

    if args.delete:
        deleted_count = 0
        to_delete = {t.strip().upper() for t in args.delete.split(',')}
        new_tips = [t for t in tips if t['ticker'].upper() not in to_delete]
        deleted_count = len(tips) - len(new_tips)
        if deleted_count > 0:
            tips = new_tips
            changed = True
        print(f"Deleted {deleted_count} tips.")

    if changed:
        save_tips(tips)
        print("tips.yaml updated successfully.")
    else:
        print("No changes were made to tips.yaml.")

if __name__ == "__main__":
    main()
