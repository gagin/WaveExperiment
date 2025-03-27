import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import os
from PIL import Image
import numpy as np
import time # Added to time execution

# --- Configuration ---
MODEL_TYPE = 'mlp' # <--- 'wave' or 'mlp' - SELECT MODEL TYPE HERE
HIDDEN_SIZE = 72    # <--- Number of hidden units for the chosen model
                    #      NOTE: For MLP, H=72 gives ~57k params (like WaveNet H=24)
                    #            For MLP, H=60 gives ~48k params (like WaveNet H=20)
ACTIVATION_TYPE = 'relu' # <--- 'relu' or 'sin'
LEARNING_RATE = 0.005 # <--- Adjust LR (e.g., 0.001 for MLP, 0.005/0.001 for WaveNet)
EPOCHS = 12         # <--- Adjust total epochs
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BASE_MODEL_NAME will be set based on MODEL_TYPE

# --- 1. Filename Generation Function ---
def generate_model_filename(base_name, hidden_size, activation_str, lr, epochs):
    """Generates a descriptive filename for the model state dictionary."""
    lr_str = str(lr).replace('.', 'p')
    filename = f"{base_name}-h{hidden_size}-{activation_str}-lr{lr_str}-e{epochs}.pth"
    return filename

# --- 2. Model Definitions ---

# --- WaveLayer (Only needed if MODEL_TYPE is 'wave') ---
class WaveLayer(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        # Use Parameter for trainable tensors
        self.A = nn.Parameter(torch.Tensor(n_in, n_out))
        self.omega = nn.Parameter(torch.Tensor(n_in, n_out))
        self.phi = nn.Parameter(torch.Tensor(n_in, n_out))
        self.B = nn.Parameter(torch.Tensor(n_out))
        self.reset_parameters() # Use a method for initialization

    def reset_parameters(self):
        # Sensible initializations
        stdv_a = math.sqrt(2.0 / self.n_in) # Kaiming-like for amplitude
        self.A.data.uniform_(-stdv_a, stdv_a)
        self.omega.data.uniform_(0.5, 1.5) # Frequencies around 1
        self.phi.data.uniform_(0, 2 * math.pi) # Phases
        # Standard bias initialization
        fan_in = self.n_in
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        self.B.data.uniform_(-bound, bound)

    def forward(self, x):
        # x shape: (batch_size, n_in)
        x_expanded = x.unsqueeze(2) # (batch_size, n_in, 1)
        # Parameters broadcast: (1, n_in, n_out)
        omega_expanded = self.omega.unsqueeze(0)
        phi_expanded = self.phi.unsqueeze(0)
        A_expanded = self.A.unsqueeze(0)
        # Calculation: A * sin(ω * x + φ)
        wave_arg = omega_expanded * x_expanded + phi_expanded
        wave_term = A_expanded * torch.sin(wave_arg)
        # Sum over input dimension
        summed_output = torch.sum(wave_term, dim=1) # (batch_size, n_out)
        # Add bias (broadcast B)
        output = summed_output + self.B.unsqueeze(0) # (batch_size, n_out)
        return output

# --- WaveNet ---
class WaveNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=HIDDEN_SIZE, num_classes=10, activation_type='relu'):
        super().__init__()
        self.wave1 = WaveLayer(input_size, hidden_size)
        self.activation_type = activation_type
        if self.activation_type == 'relu':
            self.activation_func = nn.ReLU()
        elif self.activation_type == 'sin':
            self.activation_func = torch.sin # Use function directly
        else:
             raise ValueError(f"Unsupported activation type '{activation_type}' for WaveNet")
        self.wave2 = WaveLayer(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.wave1(x)
        x = self.activation_func(x)
        x = self.wave2(x)
        return x

# --- StandardMLP ---
class StandardMLP(nn.Module):
    def __init__(self, input_size=784, hidden_size=HIDDEN_SIZE, num_classes=10, activation_type='relu'):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation_type = activation_type
        if self.activation_type == 'relu':
            self.activation_func = nn.ReLU()
        elif self.activation_type == 'sin':
            self.activation_func = torch.sin
        else:
             raise ValueError(f"Unsupported activation type '{activation_type}' for StandardMLP")
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation_func(x)
        x = self.fc2(x)
        return x

# --- 3. Load Data (No changes needed) ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST specific normalization
])
# Ensure data is downloaded (download=True)
train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True) # Larger batch for eval


# --- 4. Training Function (Add timing per epoch) ---
def train_model(model, train_loader, optimizer, criterion, device, epochs):
    print(f"Starting training on {device} for {epochs} epochs...")
    model.train()
    total_train_time = 0
    for epoch in range(epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True) # Use non_blocking with pin_memory

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += target.size(0)
            correct_predictions += (predicted == target).sum().item()

            # Optional: Print less frequently if needed
            # if batch_idx % 200 == 199:
            #      print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_train_time += epoch_duration
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        print(f'Epoch {epoch+1}/{epochs} finished. Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Time: {epoch_duration:.2f}s')
    print(f"Training finished. Total training time: {total_train_time:.2f}s")
    return total_train_time

# --- 5. Evaluation Function (Add timing) ---
def evaluate_model(model, test_loader, criterion, device):
    print("Evaluating model...")
    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    eval_start_time = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += target.size(0)
            correct_predictions += (predicted == target).sum().item()
    eval_end_time = time.time()
    eval_duration = eval_end_time - eval_start_time
    avg_loss = test_loss / total_samples
    accuracy = correct_predictions / total_samples
    print(f'Evaluation finished. Time: {eval_duration:.2f}s')
    print(f'Test Set: Average loss: {avg_loss:.4f}, Accuracy: {correct_predictions}/{total_samples} ({accuracy*100:.2f}%)')
    return accuracy, avg_loss, eval_duration

# --- 6. Generic Save/Load Functions ---
# --- 6. Generic Save/Load Functions ---
def save_model(model, path):
    print(f"Saving model state dictionary to {path}...")
    try:
        # --- FIX ---
        # Get the directory part of the path
        model_dir = os.path.dirname(path)
        # If model_dir is not empty (i.e., path includes directories), create it
        if model_dir:
             os.makedirs(model_dir, exist_ok=True)
        # --- END FIX ---

        torch.save(model.state_dict(), path)
        print("Model saved.")
    except Exception as e:
        print(f"Error saving model to {path}: {e}")

# load_model remains the same
def load_model(model, path, device):
    if os.path.exists(path):
        print(f"Loading model state dictionary from {path}...")
        try:
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print("Model loaded.")
            return True
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            return False
    else:
        return False
    
# --- 7. Prediction Utility (Identical to before) ---
def predict_digit(image_path, model, device):
    try:
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_prob, predicted_class = torch.max(probabilities, 1)
    print(f"Predicted Digit: {predicted_class.item()} (Confidence: {predicted_prob.item():.4f})")
    return predicted_class.item()


# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()

    print(f"--- Configuration ---")
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Hidden Size: {HIDDEN_SIZE}")
    print(f"Activation: {ACTIVATION_TYPE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Device: {DEVICE}")
    print(f"--------------------")

    # --- Instantiate Model based on MODEL_TYPE ---
    model = None
    base_model_name_for_file = ""
    if MODEL_TYPE == 'wave':
        model = WaveNet(hidden_size=HIDDEN_SIZE, activation_type=ACTIVATION_TYPE).to(DEVICE)
        base_model_name_for_file = "wave_mnist_model"
    elif MODEL_TYPE == 'mlp':
        model = StandardMLP(hidden_size=HIDDEN_SIZE, activation_type=ACTIVATION_TYPE).to(DEVICE)
        base_model_name_for_file = "mlp_mnist_model"
    else:
        raise ValueError(f"Invalid MODEL_TYPE '{MODEL_TYPE}'. Choose 'wave' or 'mlp'.")

    # Calculate and print parameter count AFTER instantiation
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized {MODEL_TYPE.upper()} Model. Total parameters: {total_params:,}")

    # --- Generate Filename ---
    model_path = generate_model_filename(
        base_name=base_model_name_for_file,
        hidden_size=HIDDEN_SIZE,
        activation_str=ACTIVATION_TYPE,
        lr=LEARNING_RATE,
        epochs=EPOCHS
    )
    print(f"Target model path: {model_path}")

    # --- Setup Optimizer and Criterion ---
    # Consider allowing different optimizers/params based on model type if needed
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # --- Train or Load ---
    training_time = 0
    if not load_model(model, model_path, DEVICE): # Try loading first
        print(f"Model file not found or failed to load. Starting training...")
        training_time = train_model(model, train_loader, optimizer, criterion, DEVICE, EPOCHS)
        save_model(model, model_path) # Save after training

    # --- Evaluate the final model ---
    test_accuracy, test_loss, eval_time = evaluate_model(model, test_loader, criterion, DEVICE)

    # --- Example Prediction ---
    # (Identical prediction code - consider putting in a function if reused more)
    print("\n--- Prediction Example ---")
    try:
        # Reuse data loader iterators if possible, or create new one
        data_iter = iter(test_loader)
        sample_data, sample_target = next(data_iter)
        actual_label = sample_target[0].item()
        temp_image_path = "temp_digit.png"

        unnormalize = transforms.Normalize((-0.1307/0.3081,), (1.0/0.3081,))
        image_to_save = unnormalize(sample_data[0])
        img_pil = transforms.ToPILImage()(image_to_save.squeeze(0))
        img_pil.save(temp_image_path)
        # print(f"Saved a test image to {temp_image_path}. Actual label: {actual_label}") # Less verbose

        predicted = predict_digit(temp_image_path, model, DEVICE)
        if predicted is not None:
           print(f"-> Prediction matches actual: {predicted == actual_label}")
        os.remove(temp_image_path)

    except Exception as e:
        print(f"Error during prediction example: {e}")

    # --- Final Summary ---
    end_time = time.time()
    total_duration = end_time - start_time
    print("\n--- Run Summary ---")
    print(f"Config: {MODEL_TYPE.upper()}, H={HIDDEN_SIZE}, Act={ACTIVATION_TYPE}, LR={LEARNING_RATE}, Epochs={EPOCHS}")
    print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Total Parameters: {total_params:,}")
    print(f"Total Training Time: {training_time:.2f}s")
    print(f"Evaluation Time: {eval_time:.2f}s")
    print(f"Total Script Time: {total_duration:.2f}s")
    print(f"Model saved to/loaded from: {model_path}")
    print("--------------------")