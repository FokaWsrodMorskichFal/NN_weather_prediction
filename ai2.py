import ai_np
import numpy as np  
import copy
import pandas as pd
import pickle


class NeuralNetwork(ai_np.NeuralNetwork):
    def __init__(self, structure, biases_present=True, activation='sigmoid', last_layer_activation='sigmoid'):
        if last_layer_activation not in ['tanh', 'sigmoid', 'softmax']:
            raise ValueError('Invalid last_layer_activation')
        self.last_layer_activation_name = last_layer_activation
        self.activation_name = activation
        self.created_normalization_functions = False
        super().__init__(structure, biases_present, activation, last_layer_activation)


    def calculate_normalization_parameters(self, X, y):
        self.min_y = np.min(y, axis = 1)
        self.max_y = np.max(y, axis = 1)
        self.mean_x = np.mean(X, axis = 1)
        self.std_x = np.std(X, axis = 1)
        self.create_normalization_functions()

    def create_normalization_functions(self):
        def one_hot_encode(y):
            y_one_hot = np.zeros((y.size, y.max()+1))
            y_one_hot[np.arange(y.size), y] = 1
            return y_one_hot
        def denormalize_y_one_hot(y: np.ndarray):
            """
            Denormalize y to be between 0 and 1
            """
            return np.argmax(y, axis = 1)
        
        def denormalize_y_sigmoid(y: np.ndarray):
            """
            Denormalize y to be between 0 and 1
            """
            return (y.T * (self.max_y - self.min_y) + self.min_y).T
        def normalize_y_sigmoid(y: np.ndarray):
            """
            Normalize y to be between 0 and 1
            """
            return ((y.T - self.min_y) / (self.max_y - self.min_y)).T
        
        def normalize_y_tanh(y):
            """
            Normalize y to be between -1 and 1
            """
            return normalize_y_sigmoid(y) * 2 - 1
        
        def denormalize_y_tanh(y):
            """
            Denormalize y to be between -1 and 1
            """
            return denormalize_y_sigmoid((y + 1) / 2)
        
        if self.last_layer_activation_name == 'tanh':
            self.normalize_y = normalize_y_tanh
            self.denormalize_y = denormalize_y_tanh
        elif self.last_layer_activation_name == 'sigmoid':
            self.normalize_y = normalize_y_sigmoid
            self.denormalize_y = denormalize_y_sigmoid
        else:
            self.normalize_y = one_hot_encode
            self.denormalize_y = denormalize_y_one_hot

        def normalize_x(X: np.ndarray):
            return ((X.T - self.mean_x) / self.std_x).T
        
        def denormalize_x(X: np.ndarray):
            return (X * self.std_x + self.mean_x)
        
        self.normalize_x = normalize_x
        self.denormalize_x = denormalize_x
        self.created_normalization_functions = True

    def __call__(self, X_df, X_col_names, Y_col_names = None):
        if Y_col_names is None:
            Y_col_names = list("y" + str(i) for i in range(self.structure[-1]))
        if X_col_names == -1:
            X_col_names = list(X_df.columns)
        self.validate_dataframes(X_df, X_col_names)
        new_x = X_df[X_col_names].values.T
        y = self.new_forward(new_x)
        return pd.DataFrame(y.T, columns = Y_col_names)

    def new_forward(self, X):
        if not self.created_normalization_functions:
            raise ValueError('Normalization functions not created')
        self.validate(X)
        new_x = copy.deepcopy(X)
        new_x = self.normalize_x(new_x)
        return self.denormalize_y(super().__call__(new_x))
    
    def validate(self, X, Y = None):
        if type(X) is not np.ndarray:
            raise ValueError('Invalid X type, should be np.ndarray')
        if len(X.shape) != 2:
            raise ValueError(
                """
                Invalid X shape, shape should be (n, m), 
                where n is the number of features and m is the number of samples
                """
            )
        if X.shape[0] != self.structure[0]:
            raise ValueError('Invalid X shape, number of features should be equal to the number of input nodes')
        if Y is not None:
            if type(Y) is not np.ndarray:
                raise ValueError('Invalid Y type, should be np.ndarray')
            if len(Y.shape) != 2:
                raise ValueError(
                    """
                    Invalid Y shape, shape should be (n, m), 
                    where n is the number of features and m is the number of samples
                    """
                )
            if Y.shape[0] != self.structure[-1]:
                raise ValueError('Invalid Y shape, number of features should be equal to the number of output nodes')
            if X.shape[1] != Y.shape[1]:
                raise ValueError('Invalid X and Y shape, number of samples should be equal')


    def validate_dataframes(self, X_df, X_col_names, Y_df = None, Y_col_names = None):
        if type(X_df) is not pd.DataFrame:
            raise ValueError('Invalid X_df type, should be pd.DataFrame')

        
        if type(X_col_names) is not list:
            raise ValueError('Invalid X_col_names type, should be list')
        
        if Y_df is not None:
            if type(Y_df) is not pd.DataFrame:
                raise ValueError('Invalid Y_df type, should be pd.DataFrame')
            if type(Y_col_names) is not list:
                raise ValueError('Invalid Y_col_names type, should be list')

    def train(self, X_df, Y_df, X_col_names, Y_col_names, learning_rate=0.1, epochs=100, batch_size=1):
        if X_col_names == -1:
                X_col_names = list(X_df.columns)
        if Y_col_names == -1:
            Y_col_names = list(Y_df.columns)

        self.validate_dataframes(X_df, X_col_names, Y_df, Y_col_names)
        
        new_x = copy.deepcopy(X_df[X_col_names].values.T)
        new_y = copy.deepcopy(Y_df[Y_col_names].values.T)

        self.validate(new_x, new_y)

        self.calculate_normalization_parameters(new_x, new_y)
        new_x = self.normalize_x(new_x)
        new_y = self.normalize_y(new_y)

        self.perform_training(new_x, new_y, learning_rate = learning_rate, number_of_epochs = epochs, batch_size = batch_size)
        
    def save(self, path):
        if not self.created_normalization_functions:
            raise ValueError('Normalization functions not created')
        
        with open(path, 'wb') as f:
            pickle.dump(
            {"weights": self.weights,
            "biases": self.biases,
            "activation": self.activation_name,
            "last_layer_activation": self.last_layer_activation_name,
            "x_mean": self.mean_x,
            "x_std": self.std_x,
            "y_min": self.min_y,
            "y_max": self.max_y,
            "structure": self.structure }, f
        )
            
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.weights = data["weights"]
            self.biases = data["biases"]
            self.activation_name = data["activation"]
            self.last_layer_activation_name = data["last_layer_activation"]
            self.mean_x = data["x_mean"]
            self.std_x = data["x_std"]
            self.min_y = data["y_min"]
            self.max_y = data["y_max"]
            self.structure = data["structure"]
            self.created_normalization_functions = True
        self.create_normalization_functions()

    

    