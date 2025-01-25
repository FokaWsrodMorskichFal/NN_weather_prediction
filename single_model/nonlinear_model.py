'''
This file creates a model to predict weather in the given city.
First run files "data_proc.py" with selected cities in line 29.
Then run file "window_slasher.py". This should create data required for training in the
path "./clean_norm_data/concat_clean_data_simulate_middle_day_test/".
Then run this file to train the model, which will be saved in the ./models/ directory,
together with the normalizer object. Remember, to change
number of cities to the number of cities you had selected in "data_proc.py".
I the file ./models/cities.json we store the names of the cities,
that predictions rely on.
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import numpy as np

seed = 12
torch.manual_seed(seed)
np.random.seed(seed)

column_to_predict = "max_wind_day_5"
columns_to_use = [
    "temperature",
    "pressure",
    ]

hours = 4

time_points = 72 // hours
epochs = 23
time = False
save_predictions = False
input_size = len(columns_to_use) * time_points
if time:
    input_size += 2
    columns_to_use.append("time_encoding")
net_architecture = [input_size, 32, 1]



# Normalize using Torch
class Normalizer:
    def __init__(self, net_architecture):
        self.mean_X = None
        self.std_X = None
        self.mean_Y = None
        self.std_Y = None
        self.net_architecture = net_architecture

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
    def __init__(self, structure):
        super(NeuralNet, self).__init__()
        layers = []
        for i in range(len(structure) - 1):
            layers.append(nn.Linear(structure[i], structure[i + 1]))
            if i < len(structure) - 2:  # Add ReLU only between layers, not after the output
                layers.append(nn.ReLU())
                #layers.append(nn.Dropout(p=0.1, inplace = False))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Sample DataFrames (Replace these with your actual data)
# X = pd.DataFrame(...)  # Input features
# Y = pd.DataFrame(...)  # Target values (with columns 'Ind_temp' and 'Ind_wind')

# Example Data (replace with your actual data)
if __name__ == "__main__":
    

    train = pd.read_csv("./big_data/new_train.csv")
    test = pd.read_csv("./big_data/new_test.csv")

    #old_predictions = pd.read_csv("./big_data/predictions.csv", sep = ";", index_col = 0)
    #old_predictions.rename(columns = {"avg_temp_day_4": "avg_temperature_mid_day"}, inplace = True)
    #train = pd.concat([train, old_predictions], axis = 1)

    print(train.columns)

    X = train[[column for column in train.columns if any(s in column for s in columns_to_use)]]
    X_test = test[[column for column in test.columns if any(s in column for s in columns_to_use)]]

    shuffled_indices = np.random.permutation(X.index)
    X = X.loc[shuffled_indices]


    Y = train[[column_to_predict]]
    Y = Y.loc[shuffled_indices]
    Y_test = test[[column_to_predict]]


    # Convert data to torch tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    Y_tensor = torch.tensor(Y.values, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32)

    # Normalize data
    normalizer = Normalizer(
        net_architecture=net_architecture
    )
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
    model = NeuralNet(normalizer.net_architecture)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    max_accuracy = 0
    max_accuracy_epoch = 0
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
        
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            predictions = model(X_test_normalized) #predictions normalized
            predictions_denormalized = normalizer.inverse_transform_Y(predictions)  # Denormalize
            Y_pred = pd.DataFrame(predictions_denormalized.numpy(), columns=[column_to_predict])
            Y_pred.index = Y_test.index
            if "temp" in column_to_predict:
                Y_pred["is good?"] = (Y_pred - Y_test).abs() < 2
                accuracy = Y_pred["is good?"].mean()
                print(f"Epoch: {epoch + 1}, accuracy: {accuracy}")
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                max_accuracy_epoch = epoch + 1


    # Save the trained model
    if save_predictions:
        Y_train_pred = normalizer.inverse_transform_Y(model(X_normalized))
        Y_test_pred = normalizer.inverse_transform_Y(model(X_test_normalized))

        Y_train_pred = pd.DataFrame(Y_train_pred.detach().numpy(), columns=[column_to_predict])
        Y_test_pred = pd.DataFrame(Y_test_pred.detach().numpy(), columns=[column_to_predict])


        Y_train_pred.to_csv(f"./big_data/predictions_train.csv", sep = ";")
        Y_test_pred.to_csv(f"./big_data/predictions_test.csv", sep = ";")
        torch.save(model.state_dict(), f"./models/mini_models/model.pth")
    with open(f"./models/mini_models/normalizer.pkl", "wb") as f:
        pickle.dump(normalizer, f)
    print(f"Max accuracy: {max_accuracy} at epoch {max_accuracy_epoch}")
    print(f"seed: {seed}, architecture: {net_architecture}")  
