import pandas as pd
from ai2 import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    np.random.seed(13)
    path = "./clean_norm_data/concat_clean_data_simulate_middle_day_test/"
    X_train = pd.read_csv(path + "X_train_middle.csv", header=None)
    Y_train = pd.read_csv(path + "Y_train_last.csv", index_col=0)
    AI = NeuralNetwork([1298, 64, 64, 1])
    if True:
        AI.load("model.pkl")
    AI.train(X_train, Y_train, -1, ["Ind_temp"], learning_rate=0.1, epochs = 1000)
    AI.save("model.pkl")

    