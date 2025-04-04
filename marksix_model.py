import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class MarkSixLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2):
        super(MarkSixLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Adjust input size to handle sequence data
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 6)  # Output 6 numbers
        
    def forward(self, x):
        # Ensure input is 3D: [batch_size, sequence_length, features]
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # Replace invalid values in the input
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Replace invalid values from LSTM
        lstm_out = torch.nan_to_num(lstm_out, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.fc1(lstm_out)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Final clamp to avoid propagation of NaNs
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Ensure output is in valid range (1-49)
        x = torch.sigmoid(x) * 48 + 1
        return x

class MarkSixMLP(nn.Module):
    def __init__(self, input_size: int):
        super(MarkSixMLP, self).__init__()
        
        # Calculate total input size (sequence_length * features)
        self.total_input_size = input_size * 5  # 5 is sequence length
        
        self.network = nn.Sequential(
            nn.Linear(self.total_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # Output 6 numbers
        )
    
    def forward(self, x):
        # Ensure input is 3D for sequence data
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        # Replace invalid values in the input
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Reshape the input: [batch_size, sequence_length, features] -> [batch_size, sequence_length * features]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        x = self.network(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Ensure output is in valid range (1-49)
        x = torch.sigmoid(x) * 48 + 1
        return x

class ModelTrainer:
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              epochs: int = 100,
              early_stopping_patience: int = 10):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                
                # Check for NaN values in outputs
                if torch.isnan(outputs).any():
                    print(f"Warning: NaN values detected in training outputs at epoch {epoch + 1}")
                    continue
                
                loss = self.criterion(outputs, batch_targets)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs = self.model(batch_features)
                    
                    # Check for NaN values in validation
                    if torch.isnan(outputs).any():
                        print(f"Warning: NaN values detected in validation outputs at epoch {epoch + 1}")
                        continue
                        
                    loss = self.criterion(outputs, batch_targets)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            self.history['val_loss'].append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}")
    
    def plot_training_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    def predict(self, features: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            features = features.to(self.device)
            predictions = self.model(features)
            
            # Check for NaN values in predictions
            if torch.isnan(predictions).any():
                print("Warning: NaN values detected in predictions, using fallback values")
                # Return fallback predictions
                return np.array([sorted(np.random.choice(range(1, 50), 6, replace=False))])
            
            predictions = predictions.cpu().numpy()
            
            # Round and ensure values are between 1 and 49
            predictions = np.clip(np.round(predictions), 1, 49)
            
            # Ensure unique numbers
            for i in range(len(predictions)):
                numbers = predictions[i]
                unique_numbers = []
                
                # First, add all valid numbers
                for num in numbers:
                    try:
                        num = int(num)
                        if not np.isnan(num) and num not in unique_numbers and 1 <= num <= 49:
                            unique_numbers.append(num)
                    except (ValueError, TypeError):
                        continue  # Skip invalid numbers
                
                # If we don't have enough numbers, add more
                while len(unique_numbers) < 6:
                    num = np.random.randint(1, 50)
                    if num not in unique_numbers:
                        unique_numbers.append(num)
                
                # Sort the numbers
                predictions[i] = sorted(unique_numbers[:6])
            
            return predictions.astype(int) 