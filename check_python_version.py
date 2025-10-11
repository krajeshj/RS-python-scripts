#!/usr/bin/env python3
"""
Python version compatibility checker for RS-python-scripts
"""
import sys

def check_python_version():
    """Check if Python version is compatible with required packages"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3:
        print("❌ Python 3 is required")
        return False
    elif version.major == 3 and version.minor < 8:
        print("❌ Python 3.8+ is required")
        return False
    elif version.major == 3 and version.minor == 8:
        print("⚠️  Python 3.8 detected - some packages may have compatibility issues")
        print("   Consider upgrading to Python 3.9+ for better compatibility")
        return True
    else:
        print("✅ Python version is compatible")
        return True

def check_required_packages():
    """Check if required packages can be imported"""
    packages = [
        'requests',
        'numpy', 
        'pandas',
        'scipy',
        'beautifulsoup4',
        'yaml',
        'yfinance',
        'yahoo_fin'
    ]
    
    failed_imports = []
    
    for package in packages:
        try:
            if package == 'beautifulsoup4':
                import bs4
            elif package == 'yaml':
                import yaml
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0, failed_imports

if __name__ == "__main__":
    print("=== Python Version Check ===")
    version_ok = check_python_version()
    
    print("\n=== Package Import Check ===")
    packages_ok, failed = check_required_packages()
    
    print("\n=== Summary ===")
    if version_ok and packages_ok:
        print("✅ All checks passed! The program should work correctly.")
    else:
        print("❌ Some issues found:")
        if not version_ok:
            print("   - Python version incompatible")
        if not packages_ok:
            print(f"   - Failed to import: {', '.join(failed)}")
            print("   - Try running: pip install -r requirements.txt")
