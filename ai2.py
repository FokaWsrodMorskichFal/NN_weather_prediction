import ai_np
import numpy as np  
import copy


class NeuralNetwork(ai_np.NeuralNetwork):
    def __init__(self, structure, biases_present=True, activation='sigmoid', last_layer_activation='tanh'):
        if last_layer_activation not in ['tanh', 'sigmoid', 'softmax']:
            raise ValueError('Invalid last_layer_activation')
        self.last_layer_activation_name = last_layer_activation
        self.created_normalization_functions = False
        super().__init__(structure, biases_present, activation, last_layer_activation)


    def create_normalization_functions(self, X, Y):
        self.min_y = np.min(y, axis = 1)
        self.max_y = np.max(y, axis = 1)
        self.mean_x = np.mean(X, axis = 1)
        self.std_x = np.std(X, axis = 1)

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
            return y * (self.max_y - self.min_y) + self.min_y
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

    def __call__(self, X):
        if not self.created_normalization_functions:
            raise ValueError('Normalization functions not created')
        new_x = copy.deepcopy(X)
        new_x = self.normalize_x(new_x)
        new_x = new_x.reshape((new_x.shape[0], new_x.shape[1], 1))
        output = self.denormalize_y(super().forward(new_x))
        return output.reshape((output.shape[0], output.shape[1]))

    def train(self, X: np.ndarray, Y: np.ndarray, learning_rate=0.1, epochs=100):
        if X.shape[0] != self.structure[0]:
            raise ValueError('Invalid input shape')
        if Y.shape[0] != self.structure[-1]:
            raise ValueError('Invalid output shape')
        new_x: np.ndarray = copy.deepcopy(X)
        new_y: np.ndarray = copy.deepcopy(Y)
        self.create_normalization_functions(new_x, new_y)
        new_x = self.normalize_x(new_x)
        new_y = self.normalize_y(new_y)
        new_x = new_x.reshape((new_x.shape[1],new_x.shape[0], 1))
        new_y = new_y.reshape((new_y.shape[1],new_y.shape[0], 1))
        print("X")
        print(new_x)
        print("Y")
        print(new_y)
        self.perform_training(new_x, new_y, learning_rate = learning_rate, number_of_epochs = epochs)
        

        
        
        

    


if __name__ == "__main__":
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    y = np.array([[1, 0, 0, 1]])
    A = NeuralNetwork([2, 2, 1])
    A.train(X, y, learning_rate=1, epochs=100)
    

    