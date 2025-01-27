import pandas as pd
import json

def main():
    cities = json.load(open("./models/cities.json"))
    result = []
    for city, neighbours in cities.items():
        df = pd.read_csv(f"./results/{city}_predictions.csv", index_col=0)
        temp_accuracy = ((df[f"{city}_temp_real"] - df[f"{city}_temp_pred"]).abs() < 2).mean()
        wind_accuracy = ((df[f"{city}_wind_real"] < 6) == (df[f"{city}_wind_pred"] < 6)).mean()
        result.append([neighbours[0], temp_accuracy, wind_accuracy])
    df = pd.DataFrame(result, columns=['City', 'temperature accuracy small model', 'wind accuracy small model'])
        
    big_temp = pd.read_csv("./results/all_temp_predictions.csv", index_col=0, sep=";")
    big_wind = pd.read_csv("./results/all_wind_predictions.csv", index_col=0, sep=";")

    big_temp["is good?"] = (big_temp["real temp"] - big_temp["predicted temp"]).abs() < 2
    big_wind["is good?"] = (big_wind["real wind"] < 6) == (big_wind["predicted wind"] < 6)
    big_temp = big_temp.groupby("city")["is good?"].mean()
    big_wind = big_wind.groupby("city")["is good?"].mean()
    
    result = pd.merge(df, big_temp, left_on="City", right_index=True, right_on="city", how="outer")
    result.rename(columns={"is good?": "temperature accuracy big model"}, inplace=True)
    result = pd.merge(result, big_wind, left_on="City", right_index=True, right_on="city", how="outer")
    result.rename(columns={"is good?": "wind accuracy big model"}, inplace=True)
    result.reset_index(drop=True, inplace=True)
    result.to_csv("./results/final_results.csv", index=False)

    




if __name__ == "__main__":
    main()