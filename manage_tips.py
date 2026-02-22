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
    parser.add_argument("--label", default="Watching", help="Label for the tip group")
    parser.add_argument("--date", default=None, help="Date for the tip group (YYYY-MM-DD)")
    parser.add_argument("--source", default="Manual Entry", help="Source for the tip group")
    parser.add_argument("--delete", help="Label of the group to delete (exact match)")
    
    args = parser.parse_args()
    
    tips = load_tips()
    
    changed = False
    
    if args.add:
        tickers = [t.strip().upper() for t in args.add.split(',')]
        date = args.date if args.date else datetime.now().strftime("%Y-%m-%d")
        
        new_group = {
            "label": args.label,
            "date": date,
            "tickers": ", ".join(tickers),
            "source": args.source
        }
        tips.append(new_group)
        print(f"Added group '{args.label}' with {len(tickers)} tickers.")
        changed = True

    if args.delete:
        old_len = len(tips)
        tips = [t for t in tips if t.get('label') != args.delete]
        if len(tips) < old_len:
            changed = True
            print(f"Deleted group '{args.delete}'.")
        else:
            print(f"Group '{args.delete}' not found.")

    if changed:
        save_tips(tips)
        print("tips.yaml updated successfully.")
    else:
        print("No changes made.")

if __name__ == "__main__":
    from datetime import datetime
    main()
