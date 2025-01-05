from torch_train import NeuralNet, Normalizer, cities_number, input_size
import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt

column = "Ind_temp"
output_size = 1
model = NeuralNet(input_size, output_size)

model_path = f"./models/model_{column}.pth"  # Path to the saved model
model.load_state_dict(torch.load(model_path))

with open(f"./models/normalizer_{column}.pkl", "rb") as f:
    normalizer = pickle.load(f)

path = "./clean_norm_data/concat_clean_data_simulate_middle_day_test/"
Y_test = pd.read_csv(path + "Y_test_last.csv", index_col=0)
X_test = pd.read_csv(path + "X_test_middle.csv", header=None)

Y_test = Y_test[[column]]
 
sample_X = torch.tensor(X_test.values, dtype=torch.float32)

# Example of using the trained model for predictions
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    sample_X_normalized = (sample_X - normalizer.mean_X) / normalizer.std_X  # Normalize sample_X
    predictions = model(sample_X_normalized)  # Predictions in normalized space
    predictions_denormalized = normalizer.inverse_transform_Y(predictions)  # Denormalize
    Y_pred = df = pd.DataFrame(predictions_denormalized.numpy(), columns=[column])
    Y_pred.index = Y_test.index
    Y = Y_test.merge(Y_pred, left_index=True, right_index=True, suffixes=('_real', '_pred'))
    if "Ind_temp" == column:
        Y["is good?"] = (Y["Ind_temp_real"] - Y["Ind_temp_pred"]).abs() < 2
        print(f"Accuracy: {Y["is good?"].mean()}")

    if "Ind_wind" == column:
        Y["is good?"] = (Y["Ind_wind_pred"] > 6) == (Y["Ind_wind_real"] > 6)
        print(f"Klasa mniejszosciowa/wiekszosciowa: {(Y["Ind_wind_real"] > 6).mean()}")
        print(f"Accuracy: {Y["is good?"].mean()}")

    plt.scatter(Y.index, Y[column + "_real"], c = "blue", label = "real")
    plt.scatter(Y.index, Y[column + "_pred"], c = "red", label = "pred")
    plt.show()
