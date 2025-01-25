import json
import window_slasher
import torch_test
import pandas as pd
import data_proc
from torch_train import NeuralNet, Normalizer

def main():
    starting_index = 16
    i = 0
    cities_dict = json.load(open("./models/cities.json", "r"))
    for key, cities in cities_dict.items():
        i += 1
        if i < starting_index:
            continue
        city = key
        data_proc.main(cities, len(city))
        window_slasher.main()
        Y_temp = torch_test.main(city, "temp")
        Y_wind = torch_test.main(city, "wind")
        Y = Y_temp.merge(Y_wind, left_index=True, right_index=True)
        Y.to_csv(f"./results/{city}_predictions.csv")
        print(f"Saved city {cities[0]}.")

if __name__ == "__main__":
    main()