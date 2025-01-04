import pandas as pd


if __name__ == "__main__":
    path = "./clean_norm_data/concat_clean_data_simulate_middle_day_test/"
    Y_test_last = pd.read_csv(path + "Y_train_last.csv", index_col=0)
    Y_test_middle = pd.read_csv(path + "Y_train_middle.csv", index_col=0)

    """
    Dlaczego tu jest tak wielka rozbieznosc? Zarowno dla test jak i dla train?
    """
    print(Y_test_last["Ind_wind"].mean())  
    print(Y_test_middle["Ind_wind"].mean())  
