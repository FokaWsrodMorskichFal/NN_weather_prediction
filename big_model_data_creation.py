import pandas as pd
import numpy as np

def slash(dataframe):

    # Określ długość okna (5 dni = 120 godzin) i przesunięcie (1 dzień = 24 godziny)
    window_size = 120
    step_size = 24

    # Lista przechowująca dane dla nowej ramki danych
    new_data = []

    # Iteracja przez okna czasowe
    for start in range(0, len(dataframe) - window_size + 1, step_size):
        # Pobierz okno czasowe
        window = dataframe.iloc[start:start + window_size]
        
        # Początek okna czasowego
        start_time = window["datetime"].iloc[0]
        
        # Miasto (zakładamy jedno miasto na okno)
        city = window["city"].iloc[0]

        time_encoding_sin = window["time_encoding_sin"].iloc[0]
        time_encoding_cos = window["time_encoding_cos"].iloc[0]
        
        # Dane zmienne w czasie
        flattened_data = {
            f"{col}_{i}": window[col].iloc[i]
            for col in ["temperature", "humidity", "pressure", "wind_speed" ]
            for i in range(window_size)
        }
        
        # Dodanie kolumn stałych
        flattened_data["datetime"] = start_time
        flattened_data["city"] = city
        flattened_data["time_encoding_sin"] = time_encoding_sin
        flattened_data["time_encoding_cos"] = time_encoding_cos
        
        # Dodanie wiersza do nowej ramki danych
        new_data.append(flattened_data)

    # Tworzenie nowej ramki danych
    return pd.DataFrame(new_data)



if __name__ == '__main__':
    # Load the data
    atributes = ['humidity', 'temperature', 'wind_direction', 'wind_speed', 'pressure']
    
    dataframes = {
        atribute: pd.read_csv(f'data/{atribute}.csv').iloc[12:-1].ffill().bfill() for atribute in atributes
    }

    dataframes = {
        atribute: pd.melt(
            dataframes[atribute],
            id_vars=["datetime"],
            var_name="city",
            value_name = atribute
        ) for atribute in atributes
    }


    df = dataframes['humidity']
    for atribute in atributes[1:]:
        df = pd.merge(df, dataframes[atribute], on=['datetime', 'city'])
    
    
    df["day"] = pd.to_datetime(df["datetime"]).dt.day_of_year


    df["time_encoding_sin"] = df["day"].apply(lambda x: np.sin(2*np.pi*x/365))
    df["time_encoding_cos"] = df["day"].apply(lambda x: np.cos(2*np.pi*x/365))


    df["hour"] = pd.to_datetime(df["datetime"]).dt.hour

    df.drop(columns=["day"], inplace=True)

    df["wind_direction_sin"] = df["wind_direction"].apply(lambda x: np.sin(2*np.pi*x/360))
    df["wind_direction_cos"] = df["wind_direction"].apply(lambda x: np.cos(2*np.pi*x/360))


    df.drop(columns=["wind_direction"], inplace=True)
        
    

    df = slash(df)
    
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(by=["datetime"])

    split_ratio = 0.8
    split_index = int(split_ratio * len(df))

    df_train = df.iloc[:split_index]
    df_test = df.iloc[split_index:]
    
    df_train.to_csv('./big_data/train.csv', index=False)
    df_test.to_csv('./big_data/test.csv', index=False)
    
