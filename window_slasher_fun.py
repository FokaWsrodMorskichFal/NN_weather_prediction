import numpy as np
import pandas as pd
import csv

def save_array_of_strings_to_csv(array, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for item in array:
            writer.writerow([item])

def array_of_strings_display(arr, c, f):
    for i in range(c):
        for j in range(f):
            print(arr[i*f + j], end="\t")
        print()

def convert_date_time(df, day_ticks):
    dims = df.shape
    day_zero = 0

    for i in range(dims[0]):
        df['day'][i] = day_zero
        if (i % day_ticks == day_ticks-1):
            if (day_zero < 4):
                day_zero += 1
            else:
                day_zero = 0

    for i in range(dims[0]):
        df['time'][i] = i % day_ticks

    return df

def split_train_test_data(df, ratio, CONST_day_ticks):
    dim = df.shape
    threshold = int(ratio*dim[0])
    if threshold%CONST_day_ticks != 0:
        threshold = threshold - threshold%CONST_day_ticks
    return df[:][0:threshold], df[:][threshold:dim[0]]

def split_windows_X_Y(windows, m, day_ticks, window_tick_size):
    dims = windows.shape

    X_windows = np.array( [ windows[i][0:(m*day_ticks)] for i in range(dims[0]) ] )
    Y_windows = np.array( [ windows[i][(m*day_ticks):window_tick_size] for i in range(dims[0]) ] )

    return X_windows, Y_windows

def save_array_to_csv(array, filename):
    try:
        np.savetxt(filename, array, delimiter=',', fmt='%.6f')
        print(f"Array saved to {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")
