import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle

torch.manual_seed(23)
epochs = 159 #159
column = "Ind_wind"

# Normalize using Torch
class Normalizer:
    def __init__(self):
        self.mean_X = None
        self.std_X = None
        self.mean_Y = None
        self.std_Y = None

    def fit(self, X, Y):
        self.mean_X = torch.mean(X, dim=0)
        self.std_X = torch.std(X, dim=0)
        self.mean_Y = torch.mean(Y, dim=0)
        self.std_Y = torch.std(Y, dim=0)

    def transform(self, X, Y):
        X_normalized = (X - self.mean_X) / self.std_X
        Y_normalized = (Y - self.mean_Y) / self.std_Y
        return X_normalized, Y_normalized

    def inverse_transform_Y(self, Y_normalized):
        return Y_normalized * self.std_Y + self.mean_Y
    

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.Y = torch.tensor(Y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
# Define a neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        return self.model(x)

# Sample DataFrames (Replace these with your actual data)
# X = pd.DataFrame(...)  # Input features
# Y = pd.DataFrame(...)  # Target values (with columns 'Ind_temp' and 'Ind_wind')

# Example Data (replace with your actual data)
if __name__ == "__main__":
    

    path = "./clean_norm_data/concat_clean_data_simulate_middle_day_test/"
    X = pd.read_csv(path + "X_train_middle.csv", header=None)
    Y = pd.read_csv(path + "Y_train_last.csv", index_col=0)
    X_test = pd.read_csv(path + "X_test_middle.csv", header=None)
    Y_test = pd.read_csv(path + "Y_test_last.csv", index_col=0)
    Y_test = Y_test[[column]]
    Y = Y[[column]]


    # Convert data to torch tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    Y_tensor = torch.tensor(Y.values, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32)

    # Normalize data
    normalizer = Normalizer()
    normalizer.fit(X_tensor, Y_tensor)
    X_normalized, Y_normalized = normalizer.transform(X_tensor, Y_tensor)
    X_test_normalized, Y_test_normalized = normalizer.transform(X_test_tensor, Y_test_tensor)


    # Prepare dataset with normalized data
    dataset = CustomDataset(
        pd.DataFrame(X_normalized.numpy()), 
        pd.DataFrame(Y_normalized.numpy())
        )
    test_dataset = CustomDataset(
        pd.DataFrame(X_test_normalized.numpy()), 
        pd.DataFrame(Y_test_normalized.numpy())
        )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)


    # Initialize the model, loss function, and optimizer
    input_size = X.shape[1]  # Number of features in X
    output_size = Y.shape[1]  # Number of targets in Y
    model = NeuralNet(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_Y in dataloader:
            # Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # model.eval()  # Set the model to evaluation mode
        # test_loss = 0
        # with torch.no_grad():
        #     for batch_X, batch_Y in test_dataloader:
        #         predictions = model(batch_X)
        #         batch_loss = criterion(predictions, batch_Y)
        #         test_loss += batch_loss.item()

        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {test_loss:4f}")
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

    # Save the trained model
    torch.save(model.state_dict(), "model.pth")
    with open("normalizer.pkl", "wb") as f:
        pickle.dump(normalizer, f)
    print("Model training complete and saved as model.pth")
