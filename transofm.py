import pyarrow.parquet as pq  # For reading Parquet files
import torch
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool, cpu_count
import pandas as pd
import bisect
import random  # For sampling
def get_num_rows(file_path):
    """
    Returns the number of rows in a Parquet file.
    """
    try:
        parquet_file = pq.ParquetFile(file_path)
        return parquet_file.metadata.num_rows
    except:
        return 0  # Return 0 if there's an error

class ParquetDataset(Dataset):
    def __init__(self, file_paths, cache_size=1000):
        """
        Initializes the dataset with a subset of Parquet files.
        
        Args:
            file_paths (list): List of Parquet file paths for the current subset.
            cache_size (int): Maximum number of files to cache in memory.
        """
        self.file_paths = file_paths
        self.cache_size = cache_size
        self.cache = {}         # Cached DataFrames
        self.cache_order = []   # Track cache order for LRU
        
        # Get row counts for all files in the subset
        with Pool(min(100, cpu_count())) as pool:
            self.rows_per_file = pool.map(get_num_rows, self.file_paths)
        
        # Create cumulative row indices for quick access
        self.cumulative_rows = pd.Series(self.rows_per_file).cumsum().tolist()
        self.total_rows = self.cumulative_rows[-1] if self.cumulative_rows else 0

    def __len__(self):
        return self.total_rows

    def _load_file(self, file_path):
        """
        Loads a Parquet file into a pandas DataFrame.
        """
        try:
            table = pq.read_table(file_path)
            return table.to_pandas()
        except:
            return pd.DataFrame()  # Return empty DataFrame on failure

    def __getitem__(self, idx):
        """
        Retrieves a single data sample by global index.
        """
        # Determine which file the index belongs to
        file_idx = bisect.bisect_right(self.cumulative_rows, idx)
        row_idx = idx - self.cumulative_rows[file_idx - 1] if file_idx > 0 else idx
        file_path = self.file_paths[file_idx]

        # Check if the file is cached
        if file_path in self.cache:
            df = self.cache[file_path]
            self.cache_order.remove(file_path)  # Update cache order
        else:
            df = self._load_file(file_path)      # Load file
            self.cache[file_path] = df           # Add to cache
            if len(self.cache_order) >= self.cache_size:
                oldest = self.cache_order.pop(0)  # Evict oldest file
                del self.cache[oldest]
        
        self.cache_order.append(file_path)  # Mark as recently used

        # Retrieve the specific row and convert to tensor
        try:
            data = df.iloc[row_idx].to_dict()
            return torch.tensor(list(data.values()), dtype=torch.float)
        except:
            return torch.zeros(1)  # Return default tensor on failure

def create_dataloader(file_paths, batch_size=64, shuffle=True, num_workers=100):
    """
    Creates a PyTorch DataLoader for a subset of Parquet files.
    
    Args:
        file_paths (list): List of Parquet file paths for the current subset.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of parallel data loading workers.
    
    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    dataset = ParquetDataset(file_paths)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,            # Not needed for CPU-only
        persistent_workers=True      # Keeps workers alive for efficiency
    )


def stratified_sampling(file_list, n_samples=1000, n_repeats=10):
    """
    Generates stratified random subsets of files.
    
    Args:
        file_list (list): Complete list of Parquet file paths.
        n_samples (int): Number of files per subset.
        n_repeats (int): Number of subsets to generate.
    
    Yields:
        list: A subset of file paths.
    """
    for _ in range(n_repeats):
        subset = random.sample(file_list, n_samples)
        yield subset

if __name__ == "__main__":
    # Example list of all 100,000 Parquet files
    file_list = ['/path/to/data/file1.parquet', '/path/to/data/file2.parquet', ...]  # Replace with actual paths

    # Parameters
    batch_size = 128
    num_epochs = 10
    learning_rate = 1e-3
    n_files = 1000    # Number of files per subset
    n_repeats = 10    # Number of subsets to iterate through

    # Define your PyTorch model
    class YourPyTorchModel(torch.nn.Module):
        def __init__(self, input_size, num_classes):
            super(YourPyTorchModel, self).__init__()
            self.fc = torch.nn.Linear(input_size, num_classes)
        
        def forward(self, x):
            return self.fc(x)

    # Initialize model, optimizer, and loss function
    input_size = 100  # Replace with actual input size
    num_classes = 10  # Replace with actual number of classes
    model = YourPyTorchModel(input_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Move model to CPU
    device = torch.device('cpu')
    model.to(device)

    # Training Loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for subset_idx, subset_files in enumerate(stratified_sampling(file_list, n_files, n_repeats)):
            dataloader = create_dataloader(
                file_paths=subset_files,
                batch_size=batch_size,
                shuffle=True,
                num_workers=100
            )
            print(f"  Training on subset {subset_idx+1}/{n_repeats}")
            steps = 0
            for batch in dataloader:
                inputs = batch[:, :-1].to(device)  # Adjust based on your data
                targets = batch[:, -1].long().to(device)  # Adjust based on your data
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                steps += 1
                if steps >= M:  # Define M (number of steps per subset)
                    break
        print(f"Epoch {epoch+1} completed.")





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
