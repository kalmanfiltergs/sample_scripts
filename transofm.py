# Set model to evaluation mode
model.eval()

# Create DataLoader for inference data
inference_dataloader = DataLoader(
    dataset=ParquetDataset(file_list, columns=['col1', 'col2', 'col3', 'returns']),  # Add 'returns' column
    batch_size=128,  # Adjust based on memory
    shuffle=False,   # No need to shuffle for inference
    num_workers=4
)

# Store predictions and actual returns for comparison
all_predictions = []
all_returns = []

with torch.no_grad():  # Disable gradient calculation for inference
    for batch in inference_dataloader:
        inputs = batch[:, :-1].to(device)  # All columns except the last one (returns)
        returns = batch[:, -1].to(device)  # The 'returns' column (last column)
        
        # Get model predictions
        predictions = model(inputs)
        
        # Store predictions and returns for comparison
        all_predictions.append(predictions.cpu())  # Move to CPU for comparison
        all_returns.append(returns.cpu())          # Move to CPU for comparison

# Concatenate all results from batches
all_predictions = torch.cat(all_predictions, dim=0)
all_returns = torch.cat(all_returns, dim=0)

# Example: Compare predictions with returns
comparison = torch.stack((all_predictions, all_returns), dim=1)  # Shape: [num_samples, 2]
print(comparison[:10])  # Print the first 10 comparisons


import os
import pyarrow.parquet as pq  # For reading Parquet files
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import bisect
from collections import OrderedDict  # For implementing LRU cache

class ParquetDataset(Dataset):
    def __init__(self, file_paths, columns=None, cache_size=10):
        """
        Args:
            file_paths (list): List of paths to Parquet files.
            columns (list, optional): List of column names to load. Loads all if None.
            cache_size (int): Maximum number of files to cache in memory.
        """
        self.file_paths = file_paths
        self.columns = columns
        self.cache_size = cache_size
        self.cache = OrderedDict()  # Initialize LRU cache
        
        # Calculate number of rows per file
        self.rows_per_file = []
        for file in self.file_paths:
            try:
                parquet_file = pq.ParquetFile(file)
                self.rows_per_file.append(parquet_file.metadata.num_rows)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                self.rows_per_file.append(0)
        
        # Create cumulative row indices for global indexing
        self.cumulative_rows = pd.Series(self.rows_per_file).cumsum().tolist()
        self.total_rows = self.cumulative_rows[-1] if self.cumulative_rows else 0

    def __len__(self):
        return self.total_rows

    def _load_file(self, file_path):
        """
        Loads a Parquet file into a pandas DataFrame.
        """
        try:
            table = pq.read_table(file_path, columns=self.columns)
            df = table.to_pandas()
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

    def __getitem__(self, idx):
        # Determine which file the index belongs to
        file_idx = bisect.bisect_right(self.cumulative_rows, idx)
        row_idx = idx - self.cumulative_rows[file_idx - 1] if file_idx > 0 else idx
        file_path = self.file_paths[file_idx]
        
        # Check if the file is in cache
        if file_path in self.cache:
            df = self.cache.pop(file_path)  # Remove to update order
            self.cache[file_path] = df      # Re-insert as most recently used
        else:
            df = self._load_file(file_path)  # Load from disk
            self.cache[file_path] = df       # Add to cache
            if len(self.cache) > self.cache_size:
                evicted_file, _ = self.cache.popitem(last=False)  # Evict LRU file
                print(f"Evicted {evicted_file} from cache")
        
        # Retrieve the specific row
        try:
            data = df.iloc[row_idx].to_dict()
            # Ensure all values are numeric; adjust if necessary
            numeric_data = {k: v for k, v in data.items() if isinstance(v, (int, float))}
            return torch.tensor(list(numeric_data.values()), dtype=torch.float)
        except Exception as e:
            print(f"Error accessing row {row_idx} from {file_path}: {e}")
            return torch.zeros(1)  # Default tensor on failure

def create_dataloader(file_paths, columns=None, batch_size=64, shuffle=True, num_workers=4, cache_size=10):
    """
    Creates a DataLoader for the given Parquet files with caching.
    
    Args:
        file_paths (list): List of Parquet file paths.
        columns (list, optional): Specific columns to load. Loads all if None.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of parallel data loading workers.
        cache_size (int): Number of files to cache in memory.
    
    Returns:
        DataLoader: Configured DataLoader object.
    """
    dataset = ParquetDataset(file_paths, columns=columns, cache_size=cache_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False  # Set to True if using GPU
    )
    return dataloader

# Example Usage
if __name__ == "__main__":
    # Directory containing 100 Parquet files
    directory = '/path/to/parquet/files'  # Replace with your directory path
    file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]
    
    # Specify the columns you want to load
    selected_columns = ['col1', 'col2', 'col3', 'target']  # Replace with your column names
    
    # Create DataLoader with caching
    batch_size = 128
    dataloader = create_dataloader(
        file_paths=file_list,
        columns=selected_columns,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,      # Adjust based on your CPU cores
        cache_size=20        # Number of files to cache; adjust as needed
    )
    
    # Define a simple PyTorch model
    class SimpleModel(torch.nn.Module):
        def __init__(self, input_size, num_classes):
            super(SimpleModel, self).__init__()
            self.fc = torch.nn.Linear(input_size, num_classes)
    
        def forward(self, x):
            return self.fc(x)
    
    # Initialize model, loss, and optimizer
    input_size = len(selected_columns) - 1  # Assuming last column is the target
    num_classes = 10  # Adjust based on your task
    model = SimpleModel(input_size + 1, num_classes)  # +1 if you add new features
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training Loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            # Assume last column is the target
            inputs = batch[:, :-1]
            targets = batch[:, -1].long()
    
            # Example: Generate a new feature (col1 + col2)
            new_feature = inputs[:, 0] + inputs[:, 1]
            new_feature = new_feature.unsqueeze(1)  # Make it a column
            inputs = torch.cat((inputs, new_feature), dim=1)  # Append new feature
    
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
    
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
    
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")




import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq
import torch.nn as nn
import torch.optim as optim

# Define the MLP Model
class SignalTransformer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SignalTransformer, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()  # Ensures positive scaling factor
        
    def forward(self, s, features):
        h1 = self.activation(self.hidden_layer(features))  # [batch_size, hidden_size]
        scaling_factors = self.output_activation(self.output_layer(h1))  # [batch_size, 1]
        h = s * scaling_factors  # [batch_size, 1]
        return h

# Custom Dataset to handle loading and sampling from parquet files
class ParquetDataset(Dataset):
    def __init__(self, data_dir, symbol_date_pairs, num_files_per_batch, samples_per_file):
        """
        Initializes the dataset by storing the necessary parameters.

        Parameters:
        - data_dir: Root directory where parquet files are stored.
        - symbol_date_pairs: List of (symbol, date) tuples indicating file locations.
        - num_files_per_batch: Number of (symbol, date) files to load per batch.
        - samples_per_file: Number of rows to sample from each file.
        """
        self.data_dir = data_dir
        self.symbol_date_pairs = symbol_date_pairs
        self.num_files_per_batch = num_files_per_batch
        self.samples_per_file = samples_per_file

    def __len__(self):
        # Define an arbitrary large number since we're sampling with replacement
        return 1000000  # Adjust based on your training needs

    def __getitem__(self, idx):
        # Randomly select a subset of files for this batch
        sampled_files = random.sample(self.symbol_date_pairs, self.num_files_per_batch)
        
        all_data = []
        for symbol, date in sampled_files:
            file_path = os.path.join(self.data_dir, symbol, date)
            try:
                df = pq.read_table(file_path).to_pandas()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue  # Skip files that cannot be read
            
            # Sample rows from the dataframe
            if len(df) < self.samples_per_file:
                df_sampled = df.sample(n=len(df), replace=True)
            else:
                df_sampled = df.sample(n=self.samples_per_file)
            
            all_data.append(df_sampled)
        
        if not all_data:
            # In case no data was loaded, return empty tensors
            return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
        
        # Concatenate all sampled data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Extract and convert to tensors
        s_batch = torch.tensor(combined_data['s'].values, dtype=torch.float32).unsqueeze(1)  # [batch_size, 1]
        features_batch = torch.tensor(combined_data[['volatility', 'spread', 'price']].values, dtype=torch.float32)  # [batch_size, 3]
        r_future_batch = torch.tensor(combined_data['r_future'].values, dtype=torch.float32).unsqueeze(1)  # [batch_size, 1]
        w_batch = torch.tensor(combined_data['w'].values, dtype=torch.float32).unsqueeze(1)  # [batch_size, 1]
        
        return s_batch, features_batch, r_future_batch, w_batch

# Function to build the dataset and dataloader
def build_dataloader(data_dir, num_files_per_batch=10, samples_per_file=1000, num_workers=4):
    # Build list of (symbol, date) pairs
    symbols = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    symbol_date_pairs = []
    for symbol in symbols:
        date_files = [f for f in os.listdir(os.path.join(data_dir, symbol)) if f.endswith('.parquet')]
        symbol_date_pairs.extend([(symbol, date_file) for date_file in date_files])
    
    # Create the dataset
    dataset = ParquetDataset(data_dir, symbol_date_pairs, num_files_per_batch, samples_per_file)
    
    # Create the dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=1,  # Dataset returns a full batch
        shuffle=False,
        num_workers=num_workers
    )
    
    return dataloader

# Training Function
def train_model(model, dataloader, num_epochs=1000, learning_rate=0.001, device='cpu'):
    """
    Trains the SignalTransformer model.

    Parameters:
    - model: Instance of SignalTransformer.
    - dataloader: DataLoader instance providing training data.
    - num_epochs: Number of training epochs.
    - learning_rate: Learning rate for the optimizer.
    - device: 'cpu' or 'cuda' for GPU acceleration.
    """
    model.to(device)
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (s_batch, features_batch, r_future_batch, w_batch) in enumerate(dataloader):
            # Move data to device
            s_batch = s_batch.to(device)
            features_batch = features_batch.to(device)
            r_future_batch = r_future_batch.to(device)
            w_batch = w_batch.to(device)
            
            # Forward pass: compute h(s, features)
            outputs = model(s_batch, features_batch)  # [batch_size, 1]
            
            # Compute weighted MSE loss
            loss = torch.sum(w_batch * (outputs - r_future_batch) ** 2) / torch.sum(w_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Compute average loss for the epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        
        # (Optional) Print progress every 100 epochs
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.6f}')

# Example Usage
if __name__ == "__main__":
    data_dir = 'path_to_your_data_directory'  # Replace with your data directory path
    
    # Build the dataloader
    dataloader = build_dataloader(
        data_dir=data_dir, 
        num_files_per_batch=10, 
        samples_per_file=1000, 
        num_workers=4  # Adjust based on your CPU cores
    )
    
    # Instantiate the model
    input_size = 3  # Volatility, Spread, Price
    hidden_size = 10  # Adjust as needed
    model = SignalTransformer(input_size, hidden_size)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train the model
    train_model(
        model=model, 
        dataloader=dataloader, 
        num_epochs=1000, 
        learning_rate=0.001, 
        device=device
    )
