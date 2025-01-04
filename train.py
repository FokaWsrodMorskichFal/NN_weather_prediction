import pandas as pd
from ai2 import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    np.random.seed(23)
    path = "./clean_norm_data/concat_clean_data_simulate_middle_day_test/"
    X_train = pd.read_csv(path + "X_train_middle.csv", header=None)
    Y_train = pd.read_csv(path + "Y_train_middle.csv", index_col=0)
    X_test = pd.read_csv(path + "X_test_middle.csv", header=None)
    Y_test = pd.read_csv(path + "Y_test_middle.csv", index_col=0)
    AI = NeuralNetwork([1298, 32, 23, 1])
    AI.train(X_train, Y_train, -1, ["Ind_temp"], learning_rate=0.01, epochs = 200, X_test=X_test, Y_test=Y_test)
    AI.save("model.pkl")

    