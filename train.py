import pandas as pd
from ai2 import NeuralNetwork
import matplotlib.pyplot as plt


if __name__ == "__main__":
    path = "./clean_norm_data/concat_clean_data_simulate_middle_day_test/"

    X_train = pd.read_csv(path + "X_train_middle.csv", header=None)
    Y_train = pd.read_csv(path + "Y_train_last.csv", index_col=0)
    AI = NeuralNetwork([866, 100, 100, 2])
    AI.train(X_train, Y_train, -1, ["Ind_temp", "Ind_wind"])
    AI.save("model.pkl")

    