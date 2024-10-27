import numpy as np

class NeuralNetwork:
    #initialize a new neural network with random weights and bias
    #provide a learning rate such as 0.1, 0.01, or 0.001
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    #this function is used by the predict function in layer 2
    #this function is not meant to be called directly
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #this function is used by the compute gradients function
    #this function is not meant to be called directly
    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    #make a prediction using the input, weights, and bias
    #first layer uses dot product
    #second layer uses a sigmoid function
    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    #this function calculates how the weights and bias should be updated
    #using gradient descent to find direction and rate to update parameters
    #apply chain rule - backpropagation
    #this function is used in the train function
    #this function is not meant to be used directly
    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)
        derror_dbias = (derror_dprediction * dprediction_dlayer1 * dlayer1_dbias)
        derror_dweights = (derror_dprediction * dprediction_dlayer1 * dlayer1_dweights)
        return derror_dbias, derror_dweights

    #this function updates the weights and bias based on results of compute_gradients function
    #this function is used in the train function
    #this function is not meant to be used directly
    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (derror_dweights * self.learning_rate)

    #this function trains the neural network
    #user provides input, targets, and number of iterations
    #this function will output a graph of cumulative error every 100 iterations
    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))
            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]
            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(input_vector, target)
            self._update_parameters(derror_dbias, derror_dweights)
            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]
                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)
                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)
        return cumulative_errors
