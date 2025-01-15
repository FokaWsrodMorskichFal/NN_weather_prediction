import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

column_to_predict = "avg_temp_day_5"
columns_to_use = [
    "temperature",
    "wind_speed",
    
    
    ]

if __name__ == "__main__":
    train_data = pd.read_csv('./big_data/new_train.csv', sep = ',')
    test_data = pd.read_csv('./big_data/new_test.csv', sep = ',')

    train_Y = train_data[column_to_predict]
    test_Y = test_data[column_to_predict]

    date = pd.to_datetime(test_data["datetime"])

    X_columns = [col for col in train_data.columns if any(s in col for s in columns_to_use)]
    train_X = train_data[X_columns]
    test_X = test_data[X_columns]

    beta = np.linalg.inv(train_X.values.T @ train_X.values) @ train_X.values.T @ train_Y.values

    predictions = test_X.values @ beta

    plt.plot(date, test_Y, label="True")
    plt.plot(date, predictions, label="Predictions")

    plt.gcf().autofmt_xdate()

    plt.legend()
    #plt.show()

    is_correct = np.abs(predictions - test_Y) < 2
    print(f"Accuracy: {np.mean(is_correct)}")


    

