# data_utils.py
"""
Utility functions for fetching and preparing financial data, including caching.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
from pathlib import Path
import datetime

# --- Configuration ---
CACHE_DIR = Path("./data_cache")

# --- Data Fetching Function ---

def fetch_data(ticker, start_date, end_date):
    """
    Fetches historical stock data using yfinance, implementing file-based caching.

    Args:
        ticker (str): The stock ticker symbol (e.g., "SPY").
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame: DataFrame with historical data, indexed by Date.
                          Returns None or raises error on failure.

    Raises:
        ValueError: If data download fails, results in an empty DataFrame,
                    or if essential columns are missing after processing.
        TypeError: If downloaded data isn't a DataFrame or cache is corrupt.
        FileNotFoundError: If cache file exists but cannot be read (permissions etc.)
        KeyError: If 'Close' column is missing unexpectedly.
        Exception: Catches other potential errors during fetch/load.
    """
    print(f"--- fetch_data for {ticker} [{start_date} to {end_date}] ---")

    # Ensure cache directory exists
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Warning: Could not create cache directory {CACHE_DIR}. Caching disabled. Error: {e}")

    # --- 1. Generate Cache Filename ---
    start_str = str(start_date)
    end_str = str(end_date)
    cache_filename = f"{ticker}_{start_str}_{end_str}.csv"
    cache_filepath = CACHE_DIR / cache_filename

    df = None # Initialize df to None

    # --- 2. Try Loading from Cache ---
    if cache_filepath.is_file():
        print(f"Cache hit: Found cache file: {cache_filepath}")
        try:
            df = pd.read_csv(cache_filepath, index_col='Date', parse_dates=True)
            print(f"Successfully loaded {len(df)} rows from cache.")
            if not isinstance(df, pd.DataFrame): raise TypeError("Cached file did not load as a DataFrame.")
            if df.empty: print("Warning: Cached DataFrame is empty."); df = None
            elif 'Close' not in df.columns: print("Warning: 'Close' missing in cache."); df = None
        except FileNotFoundError:
             print(f"Warning: Cache file reported existing but couldn't be found. Path: {cache_filepath}")
             df = None
        except pd.errors.EmptyDataError:
            print(f"Warning: Cache file {cache_filepath} is empty. Deleting and re-downloading.")
            try: cache_filepath.unlink()
            except OSError as e: print(f"Warning: Could not delete empty cache file {cache_filepath}. Error: {e}")
            df = None
        except Exception as e:
            print(f"Error loading data from cache file {cache_filepath}: {e}. Will re-download.")
            df = None

    # --- 3. Download if Cache Miss or Load Failure ---
    if df is None:
        print(f"Cache miss or invalid cache. Downloading from yfinance...")
        try:
            # Use yfinance to download data
            df = yf.download(ticker, start=start_date, end=end_date, progress=False) # progress=False

            # --- Validation of Downloaded Data ---
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"yf.download did not return a pandas DataFrame. Got: {type(df)}")

            if df.empty:
                raise ValueError(f"yf.download returned an empty DataFrame for {ticker} [{start_date} to {end_date}]. Check ticker and dates.")

            print(f"Downloaded {len(df)} rows successfully.")

            # --- Save to Cache (if download was successful) ---
            if CACHE_DIR.exists():
                try:
                    df.to_csv(cache_filepath)
                    print(f"Saved data to cache: {cache_filepath}")
                except Exception as e:
                    print(f"Warning: Failed to save data to cache file {cache_filepath}. Error: {e}")
            else:
                 print("Warning: Cache directory does not exist. Skipping save.")

        except Exception as e:
            # Catch errors during download or initial validation
            print(f"Error during yfinance download or initial processing: {e}")
            raise ValueError(f"Failed to obtain valid data for {ticker}") from e

    # --- End of Section 3 ---

    # Make sure df is a valid DataFrame at this point before proceeding
    if not isinstance(df, pd.DataFrame) or df.empty:
         # Should have been caught by earlier checks, but double-check
         raise ValueError("DataFrame is invalid or empty before NaN check.")

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ NEW: Intermediate Debugging: Inspect DataFrame +++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print("\nDEBUG: Inspecting DataFrame structure before NaN check...")
    print(f"DEBUG: DataFrame type: {type(df)}")
    print(f"DEBUG: DataFrame shape: {df.shape}")
    # Print the exact column names available
    print(f"DEBUG: DataFrame columns: {df.columns.tolist()}")
    print("DEBUG: DataFrame head:\n", df.head())
    print("------------------------------------------------------\n")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++


        # --- 4. Perform NaN Check on 'Close' Column (using MultiIndex) ---
    print("Checking for NaNs in 'Close' column...")
    try:
        # --- Define the correct column identifier ---
        # yfinance might return uppercase ticker in MultiIndex
        target_column = ('Close', ticker.upper())

        # Check if the EXACT target tuple is present in columns
        if target_column not in df.columns:
            raise KeyError(f"Target column {target_column} not found! Available columns are: {df.columns.tolist()}")

        # --- Step-by-step breakdown ---
        print(f"DEBUG: Accessing df[{target_column}] for NaN check...")
        # Access using the correct tuple identifier
        close_col_for_nan = df[target_column]
        print(f"DEBUG: Type of close_col_for_nan: {type(close_col_for_nan)}")

        # Now we expect close_col_for_nan to be a Series
        if not isinstance(close_col_for_nan, pd.Series):
             raise TypeError(f"FATAL: Accessed df[{target_column}] but got type {type(close_col_for_nan)}, expected Series.")

        print("DEBUG: Calling .isnull() on close_col_for_nan...")
        is_null_series_for_nan = close_col_for_nan.isnull()
        print(f"DEBUG: Type of is_null_series_for_nan: {type(is_null_series_for_nan)}")

        if not isinstance(is_null_series_for_nan, pd.Series):
            raise TypeError(f"FATAL: Expected pandas Series from .isnull(), got {type(is_null_series_for_nan)}")

        print("DEBUG: Calling .any() on is_null_series_for_nan...")
        has_any_nulls_result = is_null_series_for_nan.any()
        print(f"DEBUG: Result of .any(): {has_any_nulls_result}")
        print(f"DEBUG: Type of the result from .any(): {type(has_any_nulls_result)}")

        is_boolean_result = isinstance(has_any_nulls_result, (bool, np.bool_))
        print(f"DEBUG: Is the result a standard boolean? {is_boolean_result}")

        if not is_boolean_result:
             raise TypeError(f"FATAL: Expected boolean from .any(), got {type(has_any_nulls_result)}. Value: {has_any_nulls_result}")

        print("DEBUG: Now attempting the 'if' statement using the verified boolean result...")
        # Use the pre-calculated boolean result directly in the if statement
        if has_any_nulls_result:
            print("DEBUG: 'if' condition evaluated to True (NaNs found).")
            print("Warning: NaNs found in column {target_column}. Dropping rows with NaN in this column.")
            initial_rows = len(df)
            # --- IMPORTANT: Update dropna subset ---
            df.dropna(subset=[target_column], inplace=True)
            rows_dropped = initial_rows - len(df)
            print(f"Dropped {rows_dropped} rows due to NaN in {target_column}.")
            if df.empty:
                raise ValueError(f"DataFrame empty after dropping NaNs from {target_column}. Original: {initial_rows} rows.")
        else:
            print("DEBUG: 'if' condition evaluated to False (No NaNs found).")
            print(f"No NaNs found in column {target_column}.")

    except KeyError as e:
         print(f"ERROR during NaN check (KeyError): {e}")
         raise
    except TypeError as e:
        print(f"ERROR during NaN check (TypeError): {e}") # Catch the explicit TypeErrors raised above
        raise
    except Exception as e:
        print(f"ERROR during NaN check (Other Exception: {type(e).__name__}): {e}")
        raise

    # --- End of Section 4 ---


    print(f"--- fetch_data completed. Returning DataFrame with {len(df)} rows. ---")
    return df

# ... (Ensure the rest of data_utils.py, if any, is outside this function) ...

# Note: No if __name__ == "__main__": block here, this is a library file.