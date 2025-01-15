import pandas as pd

if __name__ == "__main__":
    train_data = pd.read_csv('./big_data/train.csv', sep = ',')
    new_train_frame = pd.DataFrame()
    new_train_frame["avg_temp_day_4"] = train_data[[f"temperature_{i}" for i in range(72, 96)]].mean(axis=1)

    # Obliczanie średniej temperatury dla dnia piątego
    new_train_frame["avg_temp_day_5"] = train_data[[f"temperature_{i}" for i in range(96, 120)]].mean(axis=1)

    # Obliczanie maksymalnej siły wiatru dla dnia czwartego
    new_train_frame["max_wind_day_4"] = train_data[[f"wind_speed_{i}" for i in range(72, 96)]].max(axis=1)

    # Obliczanie maksymalnej siły wiatru dla dnia piątego
    new_train_frame["max_wind_day_5"] = train_data[[f"wind_speed_{i}" for i in range(96, 120)]].max(axis=1)

    test_data = pd.read_csv('./big_data/test.csv', sep = ',')
    new_test_frame = pd.DataFrame()
    new_test_frame["avg_temp_day_4"] = test_data[[f"temperature_{i}" for i in range(72, 96)]].mean(axis=1)

    # Obliczanie średniej temperatury dla dnia piątego
    new_test_frame["avg_temp_day_5"] = test_data[[f"temperature_{i}" for i in range(96, 120)]].mean(axis=1)

    # Obliczanie maksymalnej siły wiatru dla dnia czwartego
    new_test_frame["max_wind_day_4"] = test_data[[f"wind_speed_{i}" for i in range(72, 96)]].max(axis=1)

    # Obliczanie maksymalnej siły wiatru dla dnia piątego
    new_test_frame["max_wind_day_5"] = test_data[[f"wind_speed_{i}" for i in range(96, 120)]].max(axis=1)


    # obliczanie średnich wartości dla wszystkich atrybutów
    hours = 1

    for atribute in ['humidity', 'temperature', 'wind_speed', 'pressure']:
        for i in range(72 // hours):
            new_train_frame[f"avg_{atribute}_{i}"] = train_data[[f"{atribute}_{j}" for j in range(i*hours, (i+1)*hours)]].mean(axis=1)
            new_test_frame[f"avg_{atribute}_{i}"] = test_data[[f"{atribute}_{j}" for j in range(i*hours, (i+1)*hours)]].mean(axis=1)

    new_train_frame[[
        'city', 'datetime', "time_encoding_sin", "time_encoding_cos"
        ]] = train_data[[
            'city', 'datetime', "time_encoding_sin", "time_encoding_cos"
            ]]
    new_test_frame[[
        'city', 'datetime', "time_encoding_sin", "time_encoding_cos"
        ]] = test_data[[
            'city', 'datetime', "time_encoding_sin", "time_encoding_cos"
            ]]
    new_train_frame.to_csv('./big_data/new_train.csv', index=False)
    new_test_frame.to_csv('./big_data/new_test.csv', index=False)

    