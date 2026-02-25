# RS-python-scripts

This is a script that fetches data from Yahoo Finance, filters stocks, and ranks them based on Relative Strength (IBD style). Both Industries and Stocks are ranked and the output is a CSV file in the output directory.

## Features

- Fetches stock data from Yahoo Finance
- Calculates Relative Strength rankings
- Applies Minervini criteria filtering
- Generates industry and stock rankings
- Outputs CSV files for analysis

## Requirements

- Python 3.9+ (recommended) or Python 3.8+ (with some compatibility issues)
- Required packages listed in `requirements.txt`

## Installation

### Windows Laptop
1. **Install Python**: Download and install Python 3.11+ from [python.org](https://www.python.org/downloads/windows/). Ensure "Add Python to PATH" is checked during installation.
2. **Clone & Setup**: Open PowerShell or Command Prompt:
   ```powershell
   git clone https://github.com/krajeshj/RS-python-scripts.git
   cd RS-python-scripts
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Verify**: Run `python check_python_version.py`

### Mac (Intel or Apple Silicon)
1. **Install Python**: Use Homebrew (`brew install python@3.11`) or download from [python.org](https://www.python.org/downloads/macos/).
2. **Clone & Setup**: Open Terminal:
   ```bash
   git clone https://github.com/krajeshj/RS-python-scripts.git
   cd RS-python-scripts
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Verify**: Run `python3 check_python_version.py`

## Configuration

Edit `config.yaml` to customize:

- `DATA_SOURCE`: Choose between YAHOO or TD_AMERITRADE
- `REFERENCE_TICKER`: Reference ticker for performance comparison (default: SPY)
- `MIN_PERCENTILE`: Minimum percentile threshold (default: 85)
- `USE_ALL_LISTED_STOCKS`: Whether to use all stocks or specific indices
- Index selection: NQ100, SP500, SP400, SP600

## Usage

### Basic Usage
```bash
python relative-strength.py
```

### With Parameters
```bash
python relative-strength.py [skipEnter] [forceTDA] [api_key]
```

- `skipEnter`: Set to "true" to skip waiting for Enter key
- `forceTDA`: Set to "true" to force TD Ameritrade API
- `api_key`: TD Ameritrade API key (if using TD Ameritrade)

## Output Files

The program generates several CSV files in the `output/` directory:

- `rs_stocks.csv`: All stocks with relative strength rankings
- `rs_stocks_minervini.csv`: Stocks meeting Minervini criteria
- `Minervini_list.csv`: Simple list of top Minervini stocks
- `rs_industries.csv`: Industry rankings

## Troubleshooting

### Python Version Issues
If you encounter `TypeError: 'type' object is not subscriptable`, you're likely using Python 3.8. Upgrade to Python 3.9+ or install compatible package versions.

### API Rate Limiting
The program includes built-in rate limiting to avoid Yahoo Finance API limits. If you encounter rate limiting errors:

1. Wait a few minutes before retrying
2. Reduce the number of stocks being processed
3. Use a smaller test dataset first

### Missing Dependencies
If you get import errors, ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Recent Fixes

- Fixed yfinance API compatibility issues with multi-level columns
- Added error handling for empty DataFrames and missing columns
- Implemented rate limiting to avoid API failures
- Fixed industry analysis for single-stock industries
- Removed Git LFS dependency to avoid quota issues
- Updated Python version requirements for better compatibility

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Please check the license file for details.