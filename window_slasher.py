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
df = pd.DataFrame(read_csv('./proc_data/concat_clean_data_temp_desc_wind_neighbors.csv', sep="\t"))
# basic adjustment
df = df.reset_index(drop=True)
df = df.apply(pd.to_numeric, errors='coerce')

# Obtaining CONST_NUM_CITIES and CONST_NUM_FEATS from the data
cols = df.columns
ind = cols[0].find('_')
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
for i in df.columns:
    X[i] = (X[i]-min[i])*(upper - lower)/(max[i] - min[i]) + lower
    Y[i] = (Y[i]-min[i])*(upper - lower)/(max[i] - min[i]) + lower
print("##### Data normalized successfully. #####")
print()

print("Selecting features for X and Y...")
# determine which features are present in X
x_bools = [True for _ in range(dim[1])]

# determine which features are present in Y to predict
y_feats = ["Ind_temp", "Ind_wind"]
y_feats_num = [df.columns.get_loc(i) for i in y_feats]
y_bools = [False for _ in range(dim[1])]
for ind in y_feats_num:
    for i in range(CONST_NUM_CITIES):
        y_bools[ind + i*CONST_NUM_FEATS] = True

#m = m[x_bools]
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

X_windows_train, Y_windows_train = window_slasher_fun.split_windows_X_Y(windows_train, 
                        x_bools, y_bools, CONST_m, CONST_n, CONST_day_ticks, CONST_window_tick_size)
X_windows_test, Y_windows_test =  window_slasher_fun.split_windows_X_Y(windows_test, 
                        x_bools, y_bools, CONST_m, CONST_n, CONST_day_ticks, CONST_window_tick_size)
print("##### Data slashed into windows. #####")
print()

print("Formating data in X and Y windows...")
num_of_feat_Y = len(y_feats)
temp_mask = [False for _ in range(CONST_day_ticks*CONST_NUM_CITIES*CONST_n*num_of_feat_Y)]
wind_mask = [False for _ in range(CONST_day_ticks*CONST_NUM_CITIES*CONST_n*num_of_feat_Y)]

for j in range(CONST_n):
    for i in range(CONST_day_ticks):
        temp_mask[j*CONST_day_ticks*num_of_feat_Y*CONST_NUM_CITIES 
                  + num_of_feat_Y*CONST_NUM_CITIES*i+y_feats_num[0]] = True

for j in range(CONST_n):
    for i in range(CONST_day_ticks):
        wind_mask[j*CONST_day_ticks*num_of_feat_Y*CONST_NUM_CITIES 
                  + num_of_feat_Y*CONST_NUM_CITIES*i + y_feats_num[1]] = True

Y_temp_train = Y_windows_train.T[temp_mask].T
Y_temp_test = Y_windows_test.T[temp_mask].T

Y_wind_train = Y_windows_train.T[wind_mask].T
Y_wind_test = Y_windows_test.T[wind_mask].T

Y_windows_train = [None for _ in range(Y_temp_train.shape[0])]
Y_windows_test = [None for _ in range(Y_temp_test.shape[0])]

# get mean temperature and max wind speed
# TO DO: generating bool variable which idicates if "strong wind" took place should also be possible
# to do so, one needs to define a threshold for wind speed and replace the np.max with a bool variable
# indicating if the threshold was exceeded
for i in range(Y_temp_train.shape[0]):
    Y_windows_train[i] = np.array([np.mean(Y_temp_train[i][0:CONST_day_ticks]), 
                                    np.max(Y_wind_train[i][0:CONST_day_ticks]), 
                                    np.mean(Y_temp_train[i][CONST_day_ticks:2*CONST_day_ticks]), 
                                    np.max(Y_temp_train[i][CONST_day_ticks:2*CONST_day_ticks])])
    
for i in range(Y_temp_test.shape[0]):
    Y_windows_test[i] = np.array([np.mean(Y_temp_test[i][0:CONST_day_ticks]), 
                                    np.max(Y_wind_test[i][0:CONST_day_ticks]), 
                                    np.mean(Y_temp_test[i][CONST_day_ticks:2*CONST_day_ticks]), 
                                    np.max(Y_temp_test[i][CONST_day_ticks:2*CONST_day_ticks])])

print("##### Data formating in X and Y windows finished. #####")
print()

X_windows_train = np.array(X_windows_train)
X_windows_test = np.array(X_windows_test)
Y_windows_train = pd.DataFrame(Y_windows_train)
Y_windows_test = pd.DataFrame(Y_windows_test)

print("Skipping hours based on CONST_SKIP_HOURS to reduce data volume...")
skip_hours = [False for _ in range(X_windows_train.shape[1])]
for i in range(int(CONST_m*CONST_day_ticks/CONST_SKIP_HOURS)):
    for j in range((CONST_NUM_FEATS*CONST_NUM_CITIES)):
        skip_hours[i*(CONST_NUM_CITIES*CONST_NUM_FEATS)*CONST_SKIP_HOURS + j] = True

X_windows_train = X_windows_train.T[skip_hours].T
X_windows_test = X_windows_test.T[skip_hours].T
print("##### Data volume reduced. #####")
print()

print("Appending encoded time to data...")
time_to_append_train = np.array([np.array(enc_time.iloc[[i*CONST_day_ticks]]).reshape(-1) 
                        for i in range(num_of_windows_train)]) 
time_to_append_test = np.array([np.array(enc_time.iloc[[i*CONST_day_ticks]]).reshape(-1) 
                        for i in range(num_of_windows_test)]) 
X_windows_train = np.append(X_windows_train, time_to_append_train, axis=1)
X_windows_test = np.append(X_windows_test, time_to_append_test, axis=1)
print("##### Encoded time appended to data. #####")
print()
print('Input size: ', X_windows_train.shape[1])
print("Saving data to files...")
path = './clean_norm_data/concat_clean_data_temp_desc_wind_neighbors/'
window_slasher_fun.save_array_to_csv(X_windows_train, path + 'X_train.csv')
window_slasher_fun.save_array_to_csv(Y_windows_train, path + 'Y_train.csv')
window_slasher_fun.save_array_to_csv(min, path + 'min.csv')
window_slasher_fun.save_array_to_csv(max, path + 'max.csv')
window_slasher_fun.save_array_to_csv(np.array([lower, upper]), path + 'normalization_boundaries.csv')
window_slasher_fun.save_array_to_csv(X_windows_test, path + 'X_test.csv')
window_slasher_fun.save_array_to_csv(Y_windows_test, path + 'Y_test.csv')
print("##### Data saved successfully. #####")