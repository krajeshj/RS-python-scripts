@echo off
echo *** Starting Intraday Quick Scan (NQ100 + Market Pulse) ***
python rs_data.py --test
python rs_ranking.py --test --quick
echo *** Quick Scan Complete. Open quick_scan.html to view results. ***
pause
