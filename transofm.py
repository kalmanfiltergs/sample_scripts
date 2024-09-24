import os
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from multiprocessing import Pool, cpu_count
import pandas as pd
import bisect

def get_num_rows(file_path):
    try:
        parquet_file = pq.ParquetFile(file_path)
        return parquet_file.metadata.num_rows
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

class ParquetDataset(Dataset):
    def __init__(self, file_paths):
        """
        Args:
            file_paths (list): List of paths to Parquet files.
        """
        self.file_paths = file_paths
        self.num_workers = min(100, cpu_count())  # Utilize up to 100 CPUs

        # Parallelize the row counting
        with Pool(self.num_workers) as pool:
            self.rows_per_file = pool.map(get_num_rows, self.file_paths)

        # Create a cumulative sum for quick index mapping
        self.cumulative_rows = pd.Series(self.rows_per_file).cumsum().tolist()
        self.total_rows = self.cumulative_rows[-1] if self.cumulative_rows else 0

    def __len__(self):
        return self.total_rows

    def __getitem__(self, idx):
        # Binary search to find the file that contains the idx
        file_idx = bisect.bisect_right(self.cumulative_rows, idx)
        if file_idx == 0:
            row_idx = idx
        else:
            row_idx = idx - self.cumulative_rows[file_idx - 1]

        file_path = self.file_paths[file_idx]
        
        # Read only the required row
        try:
            table = pq.read_table(
                file_path,
                columns=None,  # Specify columns if needed
                use_threads=True,
                skip_rows=row_idx,
                num_rows=1
            )
            df = table.to_pandas()
            data = df.iloc[0].to_dict()
            
            # Convert to tensors or desired format
            # Example: Assuming all values are numerical and needed as float tensors
            tensor = torch.tensor(list(data.values()), dtype=torch.float)
            return tensor
        except Exception as e:
            print(f"Error reading row {row_idx} from {file_path}: {e}")
            # Return a default tensor or handle the error as needed
            return torch.zeros(1)  # Example default


def create_dataloader(file_paths, batch_size=64, shuffle=True, num_workers=100):
    dataset = ParquetDataset(file_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,  # Disabled since no GPU is used
        prefetch_factor=2,  # Adjust based on memory and performance
        persistent_workers=True  # Keep workers alive between epochs for efficiency
    )
    return dataloader


if __name__ == "__main__":
    # Example list of Parquet files
    file_list = [
        '/path/to/data/file1.parquet',
        '/path/to/data/file2.parquet',
        # Add all your Parquet file paths here
    ]

    # Parameters
    batch_size = 128
    num_epochs = 10
    learning_rate = 1e-3

    # Create DataLoader
    dataloader = create_dataloader(
        file_paths=file_list,
        batch_size=batch_size,
        shuffle=True,
        num_workers=100  # Utilize all available CPUs
    )

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
        model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)

            # Assume the last column is the target; adjust as needed
            inputs = batch[:, :-1]
            targets = batch[:, -1].long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}")




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
