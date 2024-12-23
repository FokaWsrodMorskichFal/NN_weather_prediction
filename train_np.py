import read_data
import ai_np
import numpy as np
import itertools
from pandas import read_csv
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import figure
import matplotlib.colors as mcolors



if __name__ == "__main__":
    print("Loading data...")
    
    path = "./clean_norm_data/concat_clean_data_simulate_middle_day/"

    X_train = read_csv(path + 'X_train_middle.csv', sep=",", header=None, dtype=np.float32)
    Y_train = read_csv(path + 'Y_train_middle.csv', sep=",", header=None, dtype=np.float32)
    
    X_test = read_csv(path + 'X_test_middle.csv', sep=",", header=None, dtype=np.float32)
    Y_test = read_csv(path + 'Y_test_middle.csv', sep=",", header=None, dtype=np.float32)
    # read params to denormalize
    norm_boundaries = read_csv(path + 'normalization_boundaries.csv', header=None, dtype=np.float32)
    min = read_csv(path + 'min.csv', header=None, dtype=np.float32)
    max = read_csv(path + 'max.csv', header=None, dtype=np.float32)

    y_city_feats_middle = np.array(read_csv(path + 'y_city_feats_middle.csv', header=None)).reshape(-1)
    y_city_feats_last = np.array(read_csv(path + 'y_city_feats_last.csv', header=None)).reshape(-1)

    X_train = np.array(X_train.apply(pd.to_numeric, errors='coerce'))
    Y_train = np.array(Y_train.apply(pd.to_numeric, errors='coerce'))
    
    X_test = np.array(X_test.apply(pd.to_numeric, errors='coerce'))
    Y_test = np.array(Y_test.apply(pd.to_numeric, errors='coerce'))

    norm_boundaries = np.array(norm_boundaries.apply(pd.to_numeric, errors='coerce'))
    min = np.array(min.apply(pd.to_numeric, errors='coerce'))
    max = np.array(max.apply(pd.to_numeric, errors='coerce'))
    
    y_city_feats_middle = np.array(y_city_feats_middle)
    y_city_feats_last = np.array(y_city_feats_last)

    if False:
        # for tests, reduce the dataset to 'end' instances
        # test - true, proper run - false
        end = 104
        X_train = X_train[0:end]
        Y_train = Y_train[0:end]
        X_test = X_test[0:end]
        Y_test = Y_test[0:end]

    print("Data loaded.")
    print("Processing data...")

    num_train_images = X_train.shape[0]
    num_test_images = X_test.shape[0]

    X_train = X_train.T
    X_test = X_test.T
    Y_train = Y_train.T
    Y_test = Y_test.T

    print("Data processed.")

    print("         feat. num.")
    print('X_train: ' + str(X_train.shape))
    print('Y_train: ' + str(Y_train.shape))
    print('X_test:  ' + str(X_test.shape))
    print('Y_test:  ' + str(Y_test.shape))
    print("Shapes should always be 2D matrices. Even if it means that 1st dimension is of size 1.")

    print("Creating neural network...")

    input = X_train.shape[0]
    output = Y_train.shape[0]

    if True:
        nn = ai_np.NeuralNetwork(
            structure= [input, 124, 124, 124, 124, 124, 124, 124, output], 
            activation='sigmoid', 
            last_layer_activation='sigmoid',
            biases_present = True
        )
        print("Neural network created.")
    else:
        path_model = "./models/2nd_iter/"
        #path_model = "./models/"
        path_weights = path_model + 'weights-big2.npz'
        path_biases = path_model + 'biases-big2.npz'
        weights = np.load(path_weights)
        biases = np.load(path_biases)

        act = 'sigmoid'
        last_layer_act = 'tanh'
        num_hidden_layers = len(weights) - 1
        structure = []
        
        structure.append(X_test.shape[0])
        for array_name in weights.files:
            structure.append(len(weights[array_name]))  
        
        nn = ai_np.NeuralNetwork(
            structure=structure, 
            biases_present = True,
            activation = act, 
            last_layer_activation = last_layer_act
        )
        nn.weights = [weights[arr_name] for arr_name in weights.files]
        nn.biases = [biases[arr_name] for arr_name in biases.files]

        print("Neural network model successfully recreated.")

    print("Beginning NN training.")

    # planning the trainning for the NN
    #   each row represents one stage of the training
    #       numbers in each row correspond to: batch size, learning rate 
    #       and number of epochs for the particular stage 
    training_plan = [
        [32, 0.1, 20],
        [2, 0.1, 10],
        #[4, 0.1, 5]
    ]

    '''
    ,
        [16, 0.05, 0],
        [8, 0.005, 0]
    '''

    costs_list = []
    parameter_progress = []

    # number of randomly selected weights from each layer user wishes to monitor
    monitor_w = 1
    # selecting weights
    weights_to_monitor = np.array(
        [ 
            [
                i, np.random.randint(0, nn.structure[i+1]), np.random.randint(0, nn.structure[i])
            ] for i in range(nn.layers) for _ in range(monitor_w) 
        ]
    )

    for stage in range(len(training_plan)):
        print("Stage #", stage + 1, '/', len(training_plan))
        c, p = nn.perform_training(
            X_train, 
            Y_train, 
            X_test, 
            Y_test,
            batch_size=training_plan[stage][0], 
            learning_rate=training_plan[stage][1], 
            number_of_epochs=training_plan[stage][2],
            weights_to_monitor=weights_to_monitor,
            monitor_w=monitor_w
        )
        costs_list.append(c)
        parameter_progress.append(p)

    counter = 0
    for arr in costs_list:
        counter += len(arr)

    costs = np.array((counter))
    costs = np.array(list(itertools.chain(*costs_list)))
    
    #print(parameter_progress[0].shape)
    #print(parameter_progress[0])
    for i in range(1, len(parameter_progress)):
        #print(parameter_progress[i].shape)
        #print(parameter_progress[i])
        parameter_progress[0] = np.concatenate((np.array(parameter_progress[0]), np.array(parameter_progress[i])), axis=1)
    #print(parameter_progress[0])
    #print(parameter_progress[0].shape)

    #parameter_progress = np.array(list(itertools.chain(*parameter_progress)))

    print("NN training finished.")
    print("Validation...")

    Y_pred = nn.forward(X_test)
   
    # normalization params, normalizing to [-1, 1]
    lower = norm_boundaries[0]
    upper = norm_boundaries[1]

    # denormalize data
    for i in range(Y_test.shape[0]):
        Y_test[i] = (Y_test[i] - lower) * (max[i]-min[i]) / (upper - lower) + min[i]
        Y_pred[i] = (Y_pred[i] - lower) * (max[i]-min[i]) / (upper - lower) + min[i]
    
    print("Y_test shape: ", Y_test.shape)
    print("Y_pred shape: ", Y_pred.shape)
    
    # plotting results and saving them
    result_path = './middle_results/'
    for i in range(len(y_city_feats_middle)):
        fig, ax = plt.subplots()
        ax.scatter(range(len(Y_pred[i])), [tmp for tmp in Y_pred[i]], c='r', s=10, 
                    label=(y_city_feats_middle[i] + '_pred'))
        ax.scatter(range(len(Y_test[i])), [tmp for tmp in Y_test[i]], c='g', s=10, 
                    label=y_city_feats_middle[i] + '_test')    
        
        plt.title(y_city_feats_middle[i])
        plt.legend()
        ax.grid(True)

        plt.savefig(result_path + y_city_feats_middle[i] + '.png')
        matplotlib.pyplot.close()
        
    fig, ax = plt.subplots()
    ax.scatter(range(len(Y_pred[i])), [tmp1**2 + tmp2**2 for tmp1, tmp2 in zip(Y_pred[4], Y_pred[5])], c='r', s=10, 
                    label=(y_city_feats_middle[i] + '_pred'))  
        
    plt.title(y_city_feats_middle[i])
    plt.legend()
    ax.grid(True)

    plt.savefig(result_path + 'trig.png')
    matplotlib.pyplot.close()
    
    plot_path = './plots/'
    if True:
        fig, ax = plt.subplots()
        ax.scatter(range(len(costs)), [cost for cost in costs], c='b', s=10, label='Cost')
        
        plt.title("Cost over epochs")
        plt.legend()
        ax.grid(True)
        
        plt.savefig(plot_path + 'cost.png')
        #plt.show()
        
        for p in range(nn.layers*monitor_w):
            fig, ax = plt.subplots()
            ax.scatter(range(len(parameter_progress[0][p])), np.array(parameter_progress[0][p]), c='b', s=10, label='parameter')
            
            plt.title("Chosen parameters over epochs")
            plt.legend()
            ax.grid(True)
            
            plt.savefig(plot_path + 'w' + str(p) + '.png')
            #plt.show()

    np.savez(f'./models/weights-simple.npz', *nn.weights)
    np.savez(f'./models/biases-simple.npz', *nn.biases)
