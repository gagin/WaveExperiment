# models.py
"""
Neural network model definitions for WaveLayer experiments,
including generic MLP/WaveNet and specific time-series models (LSTM, GRU).
"""

import torch
import torch.nn as nn
import math

# --- Core Building Block ---

class WaveLayer(nn.Module):
    """
    A custom layer where connections are parametric wave functions:
    output = sum(A * sin(omega * x + phi)) + B
    """
    def __init__(self, n_in, n_out):
        super().__init__()
        if n_in <= 0 or n_out <= 0:
             raise ValueError("WaveLayer input and output sizes must be positive.")
        self.n_in = n_in
        self.n_out = n_out
        # Use Parameter for trainable tensors
        self.A = nn.Parameter(torch.Tensor(n_in, n_out))     # Amplitude
        self.omega = nn.Parameter(torch.Tensor(n_in, n_out)) # Frequency
        self.phi = nn.Parameter(torch.Tensor(n_in, n_out))   # Phase
        self.B = nn.Parameter(torch.Tensor(n_out))           # Bias
        self.reset_parameters()

    def reset_parameters(self):
        # Sensible initializations
        # Kaiming-like init for amplitude based on fan-in
        stdv_a = math.sqrt(2.0 / self.n_in)
        nn.init.uniform_(self.A, -stdv_a, stdv_a)

        # Initialize frequencies around 1 (can be tuned)
        nn.init.uniform_(self.omega, 0.5, 1.5)

        # Initialize phases uniformly within a full cycle
        nn.init.uniform_(self.phi, 0, 2 * math.pi)

        # Standard bias initialization based on fan-in
        fan_in = self.n_in
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.B, -bound, bound)

    def forward(self, x):
        # x shape: (batch_size, n_in)
        # Expand x for broadcasting: (batch_size, n_in, 1)
        x_expanded = x.unsqueeze(2)

        # Parameters broadcast implicitly from (n_in, n_out) during calculation
        # against x_expanded. Or expand explicitly for clarity:
        # omega_expanded = self.omega.unsqueeze(0) # (1, n_in, n_out)
        # phi_expanded = self.phi.unsqueeze(0)     # (1, n_in, n_out)
        # A_expanded = self.A.unsqueeze(0)       # (1, n_in, n_out)

        # Calculation: A * sin(ω * x + φ)
        # Broadcasting happens here: (1,n_in,n_out)*(batch,n_in,1) + (1,n_in,n_out) -> (batch,n_in,n_out)
        wave_arg = self.omega * x_expanded + self.phi
        wave_term = self.A * torch.sin(wave_arg)

        # Sum over input dimension: (batch, n_in, n_out) -> (batch, n_out)
        summed_output = torch.sum(wave_term, dim=1)

        # Add bias (broadcast B): (batch, n_out) + (n_out,) -> (batch, n_out)
        output = summed_output + self.B
        return output

    def extra_repr(self):
        return f'n_in={self.n_in}, n_out={self.n_out}'

# --- Generic Models (Applicable to MNIST or modified for RSI) ---

class GenericWaveNet(nn.Module):
    """
    A generic two-layer network using WaveLayers.
    Input -> WaveLayer -> Activation -> WaveLayer -> Output
    """
    def __init__(self, input_size, hidden_size, output_size, activation_type='sin'):
        super().__init__()
        self.layer1 = WaveLayer(input_size, hidden_size)
        self.activation_type = activation_type
        if activation_type == 'relu':
            self.activation_func = nn.ReLU()
        elif activation_type == 'sin':
            self.activation_func = torch.sin
        else:
             raise ValueError(f"Unsupported activation type '{activation_type}' for GenericWaveNet")
        self.layer2 = WaveLayer(hidden_size, output_size)

    def forward(self, x):
        # Assume x might need flattening (e.g., for MNIST images)
        # For sequences (batch, seq_len), ensure input_size matches seq_len
        x = x.view(x.size(0), -1) # Flatten input if needed
        x = self.layer1(x)
        x = self.activation_func(x)
        x = self.layer2(x)
        # No final activation (like LogSoftmax) - should be handled by loss function
        return x

class GenericMLP(nn.Module):
    """
    A generic two-layer standard MLP.
    Input -> Linear -> Activation -> Linear -> Output
    """
    def __init__(self, input_size, hidden_size, output_size, activation_type='relu'):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation_type = activation_type
        if activation_type == 'relu':
            self.activation_func = nn.ReLU()
        elif activation_type == 'sin':
            self.activation_func = torch.sin
        else:
             raise ValueError(f"Unsupported activation type '{activation_type}' for GenericMLP")
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Assume x might need flattening (e.g., for MNIST images)
        # For sequences (batch, seq_len), ensure input_size matches seq_len
        x = x.view(x.size(0), -1) # Flatten input if needed
        x = self.layer1(x)
        x = self.activation_func(x)
        x = self.layer2(x)
        # No final activation (like LogSoftmax) - should be handled by loss function
        return x

# --- Time-Series Specific Models ---

class LstmPredictor(nn.Module):
    """
    An LSTM-based predictor for sequence-to-one regression tasks.
    Input (batch, seq_len) -> LSTM -> Linear -> Output (batch, 1)
    """
    def __init__(self, input_features, hidden_size, output_size=1, num_layers=1):
        """
        Args:
            input_features (int): Number of features per time step (e.g., 1 for univariate RSI).
            hidden_size (int): Number of features in the hidden state h.
            output_size (int): Number of output values (usually 1 for prediction).
            num_layers (int): Number of recurrent layers.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # batch_first=True means input/output tensors are (batch, seq_len, features)
        self.lstm = nn.LSTM(input_size=input_features,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True)
        # Linear layer to map the hidden state of the last time step to the output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Input x shape: (batch, seq_len) -> Reshape to (batch, seq_len, input_features)
        # print(f"LSTM Input shape (before unsqueeze): {x.shape}") # Debug
        x = x.unsqueeze(-1) # Add feature dimension
        # print(f"LSTM Input shape (after unsqueeze): {x.shape}") # Debug

        # Initialize hidden and cell states for LSTM
        # Shapes: (num_layers, batch_size, hidden_size)
        device = x.device # Ensure states are on the same device as input
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        # out shape: (batch, seq_len, hidden_size)
        # hn/cn shape: (num_layers, batch_size, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        # Select output from the last sequence element: out[:, -1, :] -> (batch, hidden_size)
        last_time_step_out = out[:, -1, :]

        # Pass through the final linear layer: (batch, hidden_size) -> (batch, output_size)
        final_out = self.fc(last_time_step_out)
        # print(f"LSTM Output shape: {final_out.shape}") # Debug
        return final_out


class GruPredictor(nn.Module):
    """
    A GRU-based predictor for sequence-to-one regression tasks.
    Input (batch, seq_len) -> GRU -> Linear -> Output (batch, 1)
    """
    def __init__(self, input_features, hidden_size, output_size=1, num_layers=1):
        """
        Args:
            input_features (int): Number of features per time step (e.g., 1 for univariate RSI).
            hidden_size (int): Number of features in the hidden state h.
            output_size (int): Number of output values (usually 1 for prediction).
            num_layers (int): Number of recurrent layers.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # batch_first=True means input/output tensors are (batch, seq_len, features)
        self.gru = nn.GRU(input_size=input_features,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        # Linear layer to map the hidden state of the last time step to the output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Input x shape: (batch, seq_len) -> Reshape to (batch, seq_len, input_features)
        x = x.unsqueeze(-1)

        # Initialize hidden state for GRU
        # Shape: (num_layers, batch_size, hidden_size)
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate GRU
        # out shape: (batch, seq_len, hidden_size)
        # hn shape: (num_layers, batch_size, hidden_size)
        out, _ = self.gru(x, h0)

        # Decode the hidden state of the last time step
        # Select output from the last sequence element: out[:, -1, :] -> (batch, hidden_size)
        last_time_step_out = out[:, -1, :]

        # Pass through the final linear layer: (batch, hidden_size) -> (batch, output_size)
        final_out = self.fc(last_time_step_out)
        return final_out