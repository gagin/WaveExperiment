# rsi-downloader.py
"""
Test script for fetching data using the fetch_data utility.
"""
import datetime
# Import the function from our new library file
from data_utils import fetch_data

# --- Example Usage ---
if __name__ == "__main__":
    # Example Ticker and Dates (adjust as needed)
    TICKER = "SPY"
    DATA_START_DATE = "2023-01-01"
    # Use a slightly different end date to potentially trigger re-download/cache update
    # DATA_END_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    DATA_END_DATE = "2024-03-25" # Fixed end date for consistent testing

    try:
        # --- First call ---
        # Delete cache before first call if you want to force download:
        # from pathlib import Path
        # cache_file = Path("./data_cache") / f"{TICKER}_{DATA_START_DATE}_{DATA_END_DATE}.csv"
        # if cache_file.exists():
        #     print(f"Deleting existing cache: {cache_file}")
        #     cache_file.unlink()

        print("\n*** First call to fetch_data ***")
        spy_data_1 = fetch_data(TICKER, DATA_START_DATE, DATA_END_DATE)
        if spy_data_1 is not None:
            print("First call successful. DataFrame info:")
            spy_data_1.info() # Show info about the dataframe
            print("\nHead:\n", spy_data_1.head())
        else:
             print("First call failed to return data.")


        print("\n---------------------------------\n")

        # --- Second call (should ideally load from cache) ---
        print("*** Second call to fetch_data ***")
        spy_data_2 = fetch_data(TICKER, DATA_START_DATE, DATA_END_DATE)
        if spy_data_2 is not None:
            print("Second call successful.")
            # Optional: Check if dataframes are identical
            if spy_data_1 is not None and spy_data_1.equals(spy_data_2):
                print("Data from first call and second call (cache) are identical.")
            elif spy_data_1 is not None:
                 print("Warning: Data from first call and second call differ!")
        else:
            print("Second call failed to return data.")


    except Exception as e: # Catch any exception raised from fetch_data
        print(f"\n--- SCRIPT TERMINATED DUE TO ERROR ---")
        print(f"An error occurred during data fetching: {e}")
        # Optionally print traceback for more detail
        # import traceback
        # traceback.print_exc()
        print(f"------------------------------------")