from pandas import read_csv
import numpy as np
import pandas as pd
import csv
import window_slasher_fun

CONST_SKIP_HOURS = 2
CONST_day_ticks = 24
#number of days based on which the forecast is made
CONST_m = 3
#number of days to forecast
CONST_n = 2

CONST_window_day_size = CONST_n+CONST_m
CONST_window_tick_size = CONST_window_day_size * CONST_day_ticks

# load data
df = pd.DataFrame(read_csv('./proc_data/concat_clean_data_simulate_middle_day.csv', sep="\t"))
# basic adjustment
df = df.reset_index(drop=True)
df = df.apply(pd.to_numeric, errors='coerce')

# Obtaining CONST_NUM_CITIES, CONST_NUM_FEATS, city_names and feature_names from the data
cols = df.columns
ind = cols[0].find('_')

city_names = [cols[i][0:ind] for i in range(len(cols)-2)]
city_names = list(dict.fromkeys(city_names))

feature_names = [cols[i][(ind + 1):len(cols[i])] for i in range(len(cols)-2)]
feature_names = list(dict.fromkeys(feature_names))

city_name = cols[0][0:ind]
num_feats_tmp = 0
for i in range(len(cols)):
    if cols[i][0:ind] == city_name:
        num_feats_tmp += 1
    else:
        break
CONST_NUM_FEATS = num_feats_tmp
CONST_NUM_CITIES = int((len(cols) - 2)/CONST_NUM_FEATS)

# seperate time from data
enc_time = df[['enc_time_1', 'enc_time_2']]
df = df.drop(["enc_time_1", "enc_time_2"], axis=1)
dim = df.shape

print("Checking if data consists of only full days...")
# one more check if data consists of only full days
if dim[0] % CONST_day_ticks != 0:
    print("Data does not consist of full days. Exiting...")
    exit()
else:
    print("##### Data consists of only full days. #####")
    print()

print("Splitting data into train and test set...")
# split data in ratio
X, Y = window_slasher_fun.split_train_test_data(df, 0.80, CONST_day_ticks)
X = X.reset_index(drop=True)
Y = Y.reset_index(drop=True)

# checks for X and Y if they consist of only full days
if (X.shape[0] % CONST_day_ticks != 0 and Y.shape[0] % CONST_day_ticks != 0):
    print("Data does not consist of full days. Exiting...")
    exit()
else:
    print("##### Data split to X and Y performed successfully. #####")
    print()

print("Normalizing data...")
# params to normalize data
max = np.max(X, 0)
min = np.min(X, 0)

# normalizing data to [lower, upper] interval
lower = 0
upper = 1
'''
for i in df.columns:
    X[i] = (X[i]-min[i])*(upper - lower)/(max[i] - min[i]) + lower
    Y[i] = (Y[i]-min[i])*(upper - lower)/(max[i] - min[i]) + lower
'''
print("##### Data normalized successfully. #####")
print()

print("Selecting features for X and Y...")
cols = df.columns
# determine which features are present in X
# since the data is already preprocessed in proc_data, usually all features are selected 
x_bools = [True for _ in range(X.shape[1])]
print("X:")
print("Features:")
window_slasher_fun.array_of_strings_display(df.columns[x_bools], CONST_NUM_CITIES, CONST_NUM_FEATS)


# determine which features are present in Y to predict
# middle day simulation
y_city_feats_middle = df.columns
y_city_feats_num_middle = [df.columns.get_loc(i) for i in y_city_feats_middle]
y_bools_middle = [False for _ in range(dim[1])]
for ind in y_city_feats_num_middle:
    y_bools_middle[ind] = True
print("Y middle day:")
print("Features:")
window_slasher_fun.array_of_strings_display(y_city_feats_middle, CONST_NUM_CITIES, CONST_NUM_FEATS)

# last day predictions
y_city_name_last = [city_names[0]]
y_feats_last = ["temp", "wind"]
y_city_feats_last = [y_city_name_last[0] + "_" + i for i in y_feats_last]
y_city_feats_num_last = [df.columns.get_loc(i) for i in y_city_feats_last]
y_bools_last = [False for _ in range(dim[1])]
for ind in y_city_feats_num_last:
    y_bools_last[ind] = True
print("Y last day:")
print("Features:")
window_slasher_fun.array_of_strings_display(y_city_feats_last, len(y_city_name_last), len(y_feats_last))

max = max[x_bools]
min = min[x_bools]

dim_train = X.shape
dim_test = Y.shape
print("##### Features for X and Y selected successfully. #####")
print()

print("Slashing data into windows...")
num_of_windows_train = int((dim_train[0] - CONST_window_tick_size)/CONST_day_ticks)
num_of_windows_test = int((dim_test[0] - CONST_window_tick_size)/CONST_day_ticks)

windows_train = np.array([np.array(X[:][i*CONST_day_ticks:(i*CONST_day_ticks + CONST_window_tick_size)]) 
                        for i in range(num_of_windows_train)])
windows_test = np.array([np.array(Y[:][i*CONST_day_ticks:(i*CONST_day_ticks + CONST_window_tick_size)]) 
                        for i in range(num_of_windows_test)])

#print(windows_train.shape)
# first model makes predictions for the middle day, so only for one day into the future
X_windows_train_middle, Y_windows_train_middle = window_slasher_fun.split_windows_X_Y(windows_train, 
                        CONST_m, CONST_day_ticks, CONST_window_tick_size)
_, Y_windows_train_last =  window_slasher_fun.split_windows_X_Y(windows_train, 
                        CONST_m, CONST_day_ticks, CONST_window_tick_size)
X_windows_test_middle, Y_windows_test_middle =  window_slasher_fun.split_windows_X_Y(windows_test, 
                        CONST_m, CONST_day_ticks, CONST_window_tick_size)
_, Y_windows_test_last =  window_slasher_fun.split_windows_X_Y(windows_test, 
                        CONST_m, CONST_day_ticks, CONST_window_tick_size)

#print(X_windows_train_middle.shape)
#print(Y_windows_train_middle.shape)
#print(X_windows_test_middle.shape)
#print(Y_windows_test_middle.shape)
#print("No X_train for second model, only Y_train.")
#print(Y_windows_train_last.shape)
#print(X_windows_test_last.shape)
#print(Y_windows_test_last.shape)

print("##### Data slashed into windows. #####")
print()

print("Formating data in X and Y windows...")
num_of_cities_feat = CONST_NUM_CITIES*CONST_NUM_FEATS
y_mask_middle = [False for _ in range(num_of_cities_feat)]
y_mask_last = [False for _ in range(num_of_cities_feat)]

for k in range(len(y_city_feats_num_middle)):
    y_mask_middle[y_city_feats_num_middle[k]] = True

for k in range(len(y_city_feats_num_last)):
    y_mask_last[y_city_feats_num_last[k]] = True

Y_windows_train_middle = Y_windows_train_middle.T[y_mask_middle].T
Y_windows_test_middle = Y_windows_test_middle.T[y_mask_middle].T
Y_windows_test_last = Y_windows_test_last.T[y_mask_last].T
Y_windows_train_last = Y_windows_train_last.T[y_mask_last].T

#print(Y_windows_train_middle.shape)
#print(Y_windows_test_middle.shape)
#print(Y_windows_train_last.shape)
#print(Y_windows_test_last.shape)

Y_windows_train_middle = Y_windows_train_middle[:,0:CONST_day_ticks]
Y_windows_test_middle = Y_windows_test_middle[:,0:CONST_day_ticks]
Y_windows_train_last = Y_windows_train_last[:, CONST_day_ticks:2*CONST_day_ticks]
Y_windows_test_last = Y_windows_test_last[:, CONST_day_ticks:2*CONST_day_ticks]

#print(Y_windows_train_middle.shape)
#print(Y_windows_test_middle.shape)
#print(Y_windows_train_last.shape)
#print(Y_windows_test_last.shape)

Y_train_middle = [None for _ in range(num_of_windows_train)]
Y_test_middle = [None for _ in range(num_of_windows_test)]
Y_train_last = [None for _ in range(num_of_windows_train)]
Y_test_last = [None for _ in range(num_of_windows_test)]

for i in range(num_of_windows_train):
    Y_train_middle[i] = np.mean(Y_windows_train_middle[i], axis = 0)
    Y_train_last[i] = [np.mean(Y_windows_train_last[i].T[0]), np.max(Y_windows_train_last[i].T[1])]
    
for i in range(num_of_windows_test):
    Y_test_middle[i] = np.mean(Y_windows_test_middle[i], axis = 0)
    Y_test_last[i] = [np.mean(Y_windows_test_last[i].T[0]), np.max(Y_windows_test_last[i].T[1])]

#print(np.array(Y_train_middle).shape)
#print(np.array(Y_test_middle).shape)
#print(np.array(Y_train_last).shape)
#print(np.array(Y_test_last).shape)

print("##### Data formating in X and Y windows finished. #####")
print()

print("Skipping hours based on CONST_SKIP_HOURS to reduce data volume...")
X_train_middle_w = X_windows_train_middle[:,::CONST_SKIP_HOURS]
X_test_middle_w = X_windows_test_middle[:,::CONST_SKIP_HOURS]
#X_test_last_w = X_windows_test_last[:,::CONST_SKIP_HOURS]
#print(X_train_middle_w.shape)
#print(X_test_middle_w.shape)
#print(X_test_last_w.shape)
print("##### Data volume reduced. #####")
print()

# flattening windows
X_train_middle = np.array([X_train_middle_w[i].reshape(-1) for i in range(num_of_windows_train)])
X_test_middle = np.array([X_test_middle_w[i].reshape(-1) for i in range(num_of_windows_test)])
#X_test_last = np.array([X_test_last_w[i].reshape(-1) for i in range(num_of_windows_test)])
#print(X_train_middle.shape)
#print(X_test_middle.shape)
#print(X_test_last.shape)

print("Appending encoded time to data...")
time_to_append_train = np.array([np.array(enc_time.iloc[[i*CONST_day_ticks]]).reshape(-1) 
                        for i in range(num_of_windows_train)]) 
time_to_append_test = np.array([np.array(enc_time.iloc[[i*CONST_day_ticks]]).reshape(-1) 
                        for i in range(num_of_windows_test)]) 
X_train_middle = np.append(X_train_middle, time_to_append_train, axis=1)
X_test_middle = np.append(X_test_middle, time_to_append_test, axis=1)
#print(X_train_middle.shape)
#print(X_test_middle.shape)
print("##### Encoded time appended to data. #####")
print()

print('Input size of the first model: ', X_train_middle.shape[1])

print("Saving data to files...")
path = './clean_norm_data/concat_clean_data_simulate_middle_day_test/'

window_slasher_fun.save_array_to_csv(X_train_middle, path + 'X_train_middle.csv')
window_slasher_fun.save_array_to_csv(Y_train_middle, path + 'Y_train_middle.csv')
print(X_train_middle.shape)
print(np.array(Y_train_middle).shape)
window_slasher_fun.save_array_to_csv(X_test_middle, path + 'X_test_middle.csv')
window_slasher_fun.save_array_to_csv(Y_test_middle, path + 'Y_test_middle.csv')
print(X_test_middle.shape)
print(np.array(Y_test_middle).shape)
window_slasher_fun.save_array_to_csv(Y_train_last, path + 'Y_train_last.csv')
window_slasher_fun.save_array_to_csv(Y_test_last, path + 'Y_test_last.csv')
print(np.array(Y_train_last).shape)
print(np.array(Y_test_last).shape)
window_slasher_fun.save_array_to_csv(min, path + 'min.csv')
window_slasher_fun.save_array_to_csv(max, path + 'max.csv')
window_slasher_fun.save_array_to_csv(np.array([lower, upper]), path + 'normalization_boundaries.csv')
print(min.shape)
print(max.shape)
window_slasher_fun.save_array_of_strings_to_csv(y_city_feats_middle, path + 'y_city_feats_middle.csv')
window_slasher_fun.save_array_of_strings_to_csv(y_city_feats_last, path + 'y_city_feats_last.csv')
print(np.array(y_city_feats_middle).shape)
print(np.array(y_city_feats_last).shape)
print("##### Data saved successfully. #####")