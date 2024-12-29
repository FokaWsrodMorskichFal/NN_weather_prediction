import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ai2 import NeuralNetwork


if __name__ == "__main__":
    path = "./clean_norm_data/concat_clean_data_simulate_middle_day_test/"
    Y_test = pd.read_csv(path + "Y_test_last.csv", index_col=0)
    X_test = pd.read_csv(path + "X_test_middle.csv", header=None)
    AI = NeuralNetwork([866, 100, 10, 1])
    AI.load("model.pkl")
    Y_pred = AI(X_test, -1, ["Ind_temp"])
    Y = Y_test.merge(Y_pred, left_index=True, right_index=True, suffixes=('_real', '_pred'))
    Y["is good?"] = (Y["Ind_temp_real"] - Y["Ind_temp_pred"]).abs() < 2
    print(Y["is good?"].mean())
    plt.scatter(Y.index, Y["Ind_temp_real"], c = "blue", label = "real")
    plt.scatter(Y.index, Y["Ind_temp_pred"], c = "red", label = "pred")

    plt.show()

