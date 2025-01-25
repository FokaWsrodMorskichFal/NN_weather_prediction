import torch
import pickle
from nonlinear_model import NeuralNet, Normalizer
import pandas as pd


def main():
    # temperture
    normalizer = pickle.load(open(f"./big_data/models/normalizer_avg_temp_day_5.pkl", "rb"))
    model = NeuralNet(normalizer.net_architecture)
    model.load_state_dict(torch.load(f"./big_data/models/model_avg_temp_day_5.pth"))

    test = pd.read_csv("./big_data/new_test.csv")
    columns_to_use = [
    "temperature",
    "pressure"
    ]
    X_test = test[[column for column in test.columns if any(s in column for s in columns_to_use)]]
    sample_X = torch.tensor(X_test.values, dtype=torch.float32)
    normalized_X = (sample_X - normalizer.mean_X) / normalizer.std_X
    model.eval()
    with torch.no_grad():
        predictions = model(normalized_X)
        predictions_denormalized = normalizer.inverse_transform_Y(predictions)
        Y_pred = pd.DataFrame(predictions_denormalized.numpy(), columns=["avg_temp_day_5"])
        Y_pred.index = test.index
        Y_pred["city"] = test["city"]
        Y_pred.rename(columns = {"avg_temp_day_5": "predicted temp"}, inplace = True)
        Y_pred["real temp"] = test["avg_temp_day_5"]
        Y_pred.to_csv("./results/all_temp_predictions.csv", sep = ";")

    normalizer = pickle.load(open(f"./big_data/models/normalizer_max_wind_day_5.pkl", "rb"))
    model = NeuralNet(normalizer.net_architecture)
    model.load_state_dict(torch.load(f"./big_data/models/model_max_wind_day_5.pth"))

    test = pd.read_csv("./big_data/new_test.csv")
    columns_to_use = [
    "temperature",
    "pressure"
    ]
    X_test = test[[column for column in test.columns if any(s in column for s in columns_to_use)]]
    sample_X = torch.tensor(X_test.values, dtype=torch.float32)
    normalized_X = (sample_X - normalizer.mean_X) / normalizer.std_X
    model.eval()
    with torch.no_grad():
        predictions = model(normalized_X)
        predictions_denormalized = normalizer.inverse_transform_Y(predictions)
        Y_pred = pd.DataFrame(predictions_denormalized.numpy(), columns=["max_wind_day_5"])
        Y_pred.index = test.index
        Y_pred["city"] = test["city"]
        Y_pred.rename(columns = {"max_wind_day_5": "predicted wind"}, inplace = True)
        Y_pred["real wind"] = test["max_wind_day_5"]
        Y_pred.to_csv("./results/all_wind_predictions.csv", sep = ";")
    



if __name__ == "__main__":
    main()