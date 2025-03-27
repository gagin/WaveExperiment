import torch
import torch.nn as nn
import pandas_ta as ta
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import yfinance as yf
import pandas as pd
import pandas_ta as ta # For RSI calculation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import numpy as np
import math
import os
import time
import datetime
import json          # <-- Import json
import argparse      # <-- Import argparse
from pathlib import Path # For cleaner path handling
from data_utils import fetch_data
from models import GenericWaveNet, GenericMLP, LstmPredictor, GruPredictor, WaveLayer

# --- Global Settings ---
TICKER = "SPY"
DATA_START_DATE = "2000-01-01"
DATA_END_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15 # Test split will be 1 - TRAIN_SPLIT - VAL_SPLIT
RSI_PERIOD = 14
BASE_SAVE_DIR = Path("./rsi_models") # Directory to save models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper Functions ---

def calculate_rsi(df, ticker="SPY", period=RSI_PERIOD): # Add ticker argument
    """Calculates RSI using pandas_ta, handling potential MultiIndex."""
    print(f"Calculating RSI with period {period} for ticker {ticker}...")

    # --- Define the target column using the ticker ---
    target_column = ('Close', ticker.upper()) # Use the same tuple format

    # Ensure target column exists
    if target_column not in df.columns:
        raise KeyError(f"DataFrame must contain column {target_column} for RSI calculation. Available: {df.columns.tolist()}")

    # --- Calculate RSI on the correct Series ---
    # pandas_ta can usually handle Series directly
    print(f"DEBUG: Calculating RSI on column: {target_column}")
    df[('RSI', ticker.upper())] = ta.rsi(df[target_column], length=period)

    # Check how many NaNs were produced by RSI
    rsi_column = ('RSI', ticker.upper())
    rsi_nan_count = df[rsi_column].isnull().sum()
    print(f"RSI calculation produced {rsi_nan_count} NaN values initially.")

    # Drop rows specifically with NaN RSI
    initial_len = len(df)
    df.dropna(subset=[rsi_column], inplace=True) # Drop based on the RSI tuple column
    final_len = len(df)

    if final_len == 0:
         raise ValueError(f"DataFrame became empty after dropping NaN RSI values. Initial length: {initial_len}. Ensure sufficient data exists before {DATA_START_DATE} + {period} days.")
    print(f"DataFrame length after dropping NaN RSI: {final_len}")

    return df

def create_sequences(data, seq_length):
    """Creates sequences and corresponding targets."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys).reshape(-1, 1) # Ensure y is (n_samples, 1)

def generate_model_filename(base_dir, run_id, model_type, hidden_size, activation_str, lr, epochs, seq_length):
    """Generates a descriptive filename within a base directory."""
    lr_str = str(lr).replace('.', 'p')
    filename = f"{model_type}-h{hidden_size}-act_{activation_str}-lr{lr_str}-seq{seq_length}-e{epochs}.pth"
    # Use run_id for subfolder organization
    save_path = Path(base_dir) / run_id / filename
    return save_path

# --- Training & Evaluation Functions ---

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs, model_path):
    print(f"Starting training for {epochs} epochs...")
    model.train()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 5 # Number of epochs to wait for improvement before early stopping
    total_train_time = 0

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train() # Set model to training mode
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item() * data.size(0)
        val_loss /= len(val_loader.dataset)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_train_time += epoch_duration

        print(f'Epoch {epoch+1}/{epochs} finished. Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Time: {epoch_duration:.2f}s')

        # Early stopping and saving best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, model_path) # Save the best model based on validation loss
            print(f"Validation loss improved. Saved model to {model_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training finished. Total training time: {total_train_time:.2f}s")
    # Reload best model weights
    print("Reloading best model weights based on validation loss...")
    load_model(model, model_path, device)
    return total_train_time


def evaluate_model(model, test_loader, criterion, scaler, device):
    print("Evaluating model on test set...")
    model.eval()
    predictions_scaled = []
    actuals_scaled = []
    eval_start_time = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            predictions_scaled.extend(outputs.cpu().numpy())
            actuals_scaled.extend(target.cpu().numpy())

    eval_end_time = time.time()
    eval_duration = eval_end_time - eval_start_time
    print(f"Evaluation finished. Time: {eval_duration:.2f}s")

    predictions_scaled = np.array(predictions_scaled)
    actuals_scaled = np.array(actuals_scaled)

    # Inverse transform to get original RSI scale
    # Scaler expects shape (n_samples, n_features), our data is (n_samples, 1) which is fine
    predictions_unscaled = scaler.inverse_transform(predictions_scaled)
    actuals_unscaled = scaler.inverse_transform(actuals_scaled)

    # --- Calculate Persistence Baseline ---
    # Predict RSI(t) = RSI(t-1). We need the unscaled test set's previous values.
    # The `actuals_unscaled` holds the true values y_test[t].
    # We need y_test[t-1].
    persistence_preds_unscaled = actuals_unscaled[:-1] # y_test[0] to y_test[end-1]
    persistence_actuals_unscaled = actuals_unscaled[1:] # y_test[1] to y_test[end]

    baseline_mse = mean_squared_error(persistence_actuals_unscaled, persistence_preds_unscaled)
    baseline_mae = mean_absolute_error(persistence_actuals_unscaled, persistence_preds_unscaled)
    baseline_rmse = np.sqrt(baseline_mse)
    print(f"\nPersistence Baseline (Predict RSI(t-1)):")
    print(f"  Baseline MSE:  {baseline_mse:.6f}")
    print(f"  Baseline MAE:  {baseline_mae:.6f}")
    print(f"  Baseline RMSE: {baseline_rmse:.6f}")

    # --- Calculate Model Metrics ---
    model_mse = mean_squared_error(actuals_unscaled, predictions_unscaled)
    model_mae = mean_absolute_error(actuals_unscaled, predictions_unscaled)
    model_rmse = np.sqrt(model_mse)
    print(f"\nModel Performance (Unscaled RSI):")
    print(f"  Model MSE:  {model_mse:.6f}")
    print(f"  Model MAE:  {model_mae:.6f}")
    print(f"  Model RMSE: {model_rmse:.6f}")

    metrics = {
        "test_mse": model_mse,
        "test_mae": model_mae,
        "test_rmse": model_rmse,
        "baseline_mse": baseline_mse,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "eval_time_s": eval_duration,
    }
    return metrics

# --- Generic Save/Load Functions (with fix) ---
def save_model(model, path):
    print(f"Attempting to save model to {path}...")
    try:
        model_dir = Path(path).parent # Use pathlib for parent directory
        model_dir.mkdir(parents=True, exist_ok=True) # Create directory tree
        torch.save(model.state_dict(), path)
        # print(f"Model successfully saved to {path}") # Less verbose during training loop
    except Exception as e:
        print(f"Error saving model to {path}: {e}")

def load_model(model, path, device):
    path = Path(path) # Ensure path is a Path object
    if path.is_file():
        print(f"Loading model state dictionary from {path}...")
        try:
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print("Model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            return False
    else:
        return False

# --- Main Execution Loop ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run WaveNet vs MLP experiments on RSI prediction.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments.json", # Default config file name
        help="Path to the JSON file containing experiment configurations."
    )
    args = parser.parse_args()

    # --- Load Experiments from JSON ---
    try:
        config_path = Path(args.config)
        with open(config_path, 'r') as f:
            experiments = json.load(f)
        print(f"Loaded {len(experiments)} experiment configurations from: {config_path}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse JSON configuration file {args.config}: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred loading the configuration: {e}")
        exit(1)

    overall_start_time = time.time()
    # --- 1. Data Fetching and RSI Calculation (Done ONCE) ---
    try:
        df_spy = fetch_data(TICKER, DATA_START_DATE, DATA_END_DATE)
        if df_spy is None:
            raise ValueError("fetch_data failed.")

        # Pass the ticker to calculate_rsi
        df_spy_rsi = calculate_rsi(df_spy, ticker=TICKER)

        # --- Access RSI data using the tuple ---
        rsi_column_name = ('RSI', TICKER.upper())
        if rsi_column_name not in df_spy_rsi.columns:
            raise KeyError(f"RSI column {rsi_column_name} not found after calculation!")

        rsi_data = df_spy_rsi[rsi_column_name].values # Get NumPy array from the correct column

        if len(rsi_data) == 0:
            raise ValueError("rsi_data array is empty after processing.")
        print(f"Prepared RSI data with {len(rsi_data)} samples.")

    except (ValueError, KeyError) as e:
        print(f"Error during data preparation: {e}")
        exit() # Stop execution if data prep fails

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # <<< INSERT DEBUG PRINTS RIGHT HERE >>>
    print("\n--- Debug Info Before Experiment Loop ---")
    print(f"Type of rsi_data: {type(rsi_data)}")
    if isinstance(rsi_data, np.ndarray):
        print(f"Shape of rsi_data: {rsi_data.shape}")
        # print(f"First 5 elements of rsi_data: {rsi_data[:5]}") # Optional: view data
    else:
        print("Warning: rsi_data is NOT a NumPy array!")
    print(f"Total samples available in rsi_data for splitting: {len(rsi_data)}")
    print("---------------------------------------\n")
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


    results_summary = [] # Store results from all runs

# --- 2. Iterate Through Experiments ---
    for i, config in enumerate(experiments):
        run_start_time = time.time()
        print(f"\n===== Starting Run {i+1}/{len(experiments)}: {config['run_id']} =====")
        print(f"Config: {config}")

# --- 3. Data Splitting and Scaling ---
        seq_length = config['sequence_length']
        total_samples = len(rsi_data)
        train_end_idx = int(total_samples * TRAIN_SPLIT)
        val_end_idx = train_end_idx + int(total_samples * VAL_SPLIT)

        # --- Check split indices --- # <--- THIS IS WHERE YOU CHECK THEM
        if train_end_idx == 0 or val_end_idx <= train_end_idx or val_end_idx >= total_samples:
            print(f"Error: Invalid data split indices. total_samples={total_samples}, train_end={train_end_idx}, val_end={val_end_idx}. Skipping run.")
            continue
        # --- End check ---

        train_data_raw = rsi_data[:train_end_idx]
        val_data_raw = rsi_data[train_end_idx:val_end_idx]
        test_data_raw = rsi_data[val_end_idx:]

        print(f"Data split sizes: Train={len(train_data_raw)}, Validation={len(val_data_raw)}, Test={len(test_data_raw)}")

        # Fit scaler ONLY on training data
        scaler = MinMaxScaler(feature_range=(0, 1)) # Scale to 0-1
        try:
            # Scaler expects (n_samples, n_features), so reshape
            scaler.fit(train_data_raw.reshape(-1, 1))
        except ValueError as e:
             print(f"Error fitting scaler: {e}. train_data_raw length might be 0. Skipping run.")
             continue

        # Scale the entire dataset (or splits individually)
        # Scaling splits individually after fitting on train is safer
        train_data_scaled = scaler.transform(train_data_raw.reshape(-1, 1)).flatten()
        val_data_scaled = scaler.transform(val_data_raw.reshape(-1, 1)).flatten()
        test_data_scaled = scaler.transform(test_data_raw.reshape(-1, 1)).flatten()

# --- 4. Create Sequences (Using Scaled Data) ---
        X_train_scaled, y_train_scaled = create_sequences(train_data_scaled, seq_length)
        X_val_scaled, y_val_scaled = create_sequences(val_data_scaled, seq_length)
        X_test_scaled, y_test_scaled = create_sequences(test_data_scaled, seq_length)

        # Check if sequence creation resulted in empty arrays (if splits were too small for seq_length)
        if len(X_train_scaled) == 0 or len(X_val_scaled) == 0 or len(X_test_scaled) == 0:
             print(f"Error: Sequence creation resulted in empty arrays. Check split sizes and sequence length. Skipping run.")
             continue
        print(f"Sequence shapes: X_train={X_train_scaled.shape}, X_val={X_val_scaled.shape}, X_test={X_test_scaled.shape}")

# --- 5. Create DataLoaders ---
        batch_size = config['batch_size']
        train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled))
        val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val_scaled))
        test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test_scaled))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- 6. Instantiate Model (UPDATED LOGIC) ---
        model = None
        model_type = config['model_type'].lower() # Use lower case for safety
        hidden_size = config['hidden_size']
        input_size_for_mlp_wave = config['sequence_length'] # MLP/WaveNet take flattened sequence
        input_features_for_rnn = 1 # LSTM/GRU take features per time step (univariate)
        output_size = 1 # Regression task

        print(f"Instantiating model type: {model_type}") # Debug print

        try:
            if model_type == 'wave':
                activation = config.get('activation_type', 'sin') # Default to sin if missing
                model = GenericWaveNet(input_size=input_size_for_mlp_wave,
                                       hidden_size=hidden_size,
                                       output_size=output_size,
                                       activation_type=activation).to(DEVICE)
            elif model_type == 'mlp':
                activation = config.get('activation_type', 'relu') # Default to relu if missing
                model = GenericMLP(input_size=input_size_for_mlp_wave,
                                   hidden_size=hidden_size,
                                   output_size=output_size,
                                   activation_type=activation).to(DEVICE)
            elif model_type == 'lstm':
                num_layers = config.get('num_layers', 1) # Allow specifying layers
                model = LstmPredictor(input_features=input_features_for_rnn,
                                      hidden_size=hidden_size,
                                      output_size=output_size,
                                      num_layers=num_layers).to(DEVICE)
            elif model_type == 'gru':
                num_layers = config.get('num_layers', 1)
                model = GruPredictor(input_features=input_features_for_rnn,
                                     hidden_size=hidden_size,
                                     output_size=output_size,
                                     num_layers=num_layers).to(DEVICE)
            else:
                print(f"ERROR: Invalid model_type '{model_type}' in config {config['run_id']}. Skipping run.")
                continue

            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Initialized {model_type.upper()} Model. Total parameters: {total_params:,}")

        except Exception as e:
             print(f"ERROR: Failed to instantiate model {model_type} for run {config['run_id']}: {e}")
             # import traceback # Optional for detailed error
             # traceback.print_exc()
             continue # Skip to next experiment if model creation fails


# --- 7. Generate Filename (keep as before, ensure activation_str uses config.get) ---
        activation_str = config.get('activation_type', 'default_act') # Handle missing activation type
        lr = config['learning_rate']
        epochs = config['epochs']
        model_path = generate_model_filename(
            base_dir=BASE_SAVE_DIR, run_id=config['run_id'], model_type=model_type,
            hidden_size=hidden_size, activation_str=activation_str, lr=lr, epochs=epochs, seq_length=seq_length
        )
        print(f"Model path: {model_path}")


# --- 8. Setup Optimizer and Criterion (keep as before) ---
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

# --- 9. Train or Load ---
        training_time = 0
        # If model file doesn't exist based on final epoch count, initiate training
        if not load_model(model, model_path, DEVICE):
            print(f"Model file not found or failed to load. Starting training...")
            # Pass validation loader to training function for early stopping
            training_time = train_model(model, train_loader, val_loader, optimizer, criterion, DEVICE, config['epochs'], model_path)
            # Note: train_model now saves the *best* model based on validation loss during training.
            # If training finished normally (not early stopped), we might still want to save final state? Optional.
            # save_model(model, model_path) # Re-saving final state (overwrites best) - decide if needed
        else:
            print("Loaded pre-trained model.")


# --- 10. Evaluate ---
        metrics = evaluate_model(model, test_loader, criterion, scaler, DEVICE)
        metrics["training_time_s"] = training_time
        metrics["total_params"] = total_params

# --- 11. Store Results ---
        run_result = {**config, **metrics} # Combine config and metrics
        results_summary.append(run_result)

        run_end_time = time.time()
        print(f"===== Finished Run {i+1}/{len(experiments)}: {config['run_id']} in {run_end_time - run_start_time:.2f}s =====")


# --- 12. Final Summary Table ---
    print("\n\n===== Overall Experiment Summary =====")
    # Create a pandas DataFrame for easy viewing
    results_df = pd.DataFrame(results_summary)
    # Select and reorder columns for clarity
    display_cols = [
        'run_id', 'model_type', 'hidden_size', 'activation_type', 'sequence_length',
        'learning_rate', 'epochs', 'total_params',
        'test_rmse', 'baseline_rmse', # Show key metric vs baseline
        'test_mae', 'baseline_mae',
        'test_mse', 'baseline_mse',
        'training_time_s', 'eval_time_s'
    ]
    # Ensure all expected columns exist, adding missing ones with NaN if necessary
    for col in display_cols:
        if col not in results_df.columns:
            results_df[col] = pd.NA
            
    results_df = results_df[display_cols]
    # Format floating point numbers for readability
    pd.set_option('display.float_format', '{:.6f}'.format)
    print(results_df.to_string()) # Print full dataframe without truncation

    overall_end_time = time.time()
    print(f"\nTotal script execution time: {overall_end_time - overall_start_time:.2f}s")
    print("=======================================")