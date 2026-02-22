# originally from skyte's github

import rs_data
import rs_ranking
import sys


def main():
   skipEnter = None if len(sys.argv) <= 1 else sys.argv[1]
   forceTDA = None if len(sys.argv) <= 2 else sys.argv[2]
   fullScan = None if len(sys.argv) <= 4 else sys.argv[4] # Shifted due to potential api_key at 3
   api_key = None if len(sys.argv) <= 3 else sys.argv[3]
   
   # Correctly handle 3-4 arguments where 3 might be fullScan or api_key
   # For simplicity in CI: python relative-strength.py skipEnter forceTDA api_key fullScan
   
   if api_key and (api_key.lower() == "true" or api_key.lower() == "false"):
      # Legacy mismatch: if 3rd arg is true/false, it's likely fullScan and api_key is missing
      fullScan = api_key
      api_key = None

   if api_key:
      rs_data.main(forceTDA=="true", api_key, fullScan=="true")
   else:
      rs_data.main(forceTDA=="true", None, fullScan=="true")
   rs_ranking.main(skipEnter=="true")

if __name__ == "__main__":
   main()