import math
import random
E = math.e
random.seed(31415)


class Layer:
    def __init__(self, previous_height, height):
        self.biases = [(1*(random.random())*2-1) for n in range(height)]
        self.weights = [[(1*(random.random())*2-1) for n in range(previous_height)] for m in range(height)]
       
        self.delta_biases = [0] * height
        self.delta_weights = [[0] * previous_height for _ in range(height)]
    
    def forward(self, previous_layer_outputs):
        self.previous_layer_outputs = previous_layer_outputs
        self.outputs = [0]*len(self.biases)
        for j in range(len(self.biases)):
            for k in range(len(self.weights[0])):
                self.outputs[j] += ((previous_layer_outputs[k]) * (self.weights[j][k]))
            self.outputs[j] += self.biases[j]

    def activation_function(self, activation_function_type, is_last_layer):
        self.is_last_layer = is_last_layer
        self.activation_function_type = activation_function_type
        if self.activation_function_type == "ReLU":
            if self.is_last_layer == False:
                self.outputs = [max(0, n) for n in self.outputs]
        if self.activation_function_type == "Leaky_ReLU":
            if self.is_last_layer == False:
                self.outputs = [0.1*n if n<0 else n for n in self.outputs]
        if self.activation_function_type == "Softmax":
            maximum_output = max(self.outputs)
            self.outputs = [self.outputs[n]-maximum_output for n in range(len(self.outputs))]
            sum = 0
            for i in range(len(self.outputs)):
                sum += E**(self.outputs[i])
            for i in range(len(self.outputs)):
                self.outputs[i] = (E**(self.outputs[i]))/sum

    def loss(self, prediced_list, expected_list, type):
        self.loss_type = type
        if self.loss_type == "mse":
            self.mean_loss = 0
            for i in range(len(prediced_list)):
                self.mean_loss += 0.5*((prediced_list[i] - expected_list[i])**2)
            self.mean_loss /= len(prediced_list)
            self.d_loss = [(prediced_list[i] - expected_list[i]) for i in range(len(prediced_list))]
        elif self.loss_type == "log":
            epsilon = 1e-15  # Small constant to avoid taking the logarithm of zero
            loss = [math.log(max(prediced_list[i] - expected_list[i], epsilon)) for i in range(len(prediced_list))]
            self.mean_loss = 0
            for i in range(len(loss)):
                self.mean_loss += loss[i]
        
    def back_prop(self, inputted_loss_array):
        self.passed_on_loss_array = inputted_loss_array
        if self.activation_function_type == "ReLU":
            if self.is_last_layer == False:
                for i in range(len(self.passed_on_loss_array)):
                    if self.passed_on_loss_array[i]<1:
                        self.passed_on_loss_array[i] = 0
        if self.activation_function_type == "Leaky_ReLU":
            if self.is_last_layer == False:
                for i in range(len(self.passed_on_loss_array)):
                    if self.passed_on_loss_array[i] < 0:
                        self.passed_on_loss_array[i] *= 0.1
        if self.activation_function_type == "Softmax":
            for i in range(len(self.passed_on_loss_array)):
                self.passed_on_loss_array[i] *= (1 - self.outputs[i]) * self.outputs[i]

        for i in range(len(self.biases)):
            self.delta_biases[i] += self.passed_on_loss_array[i]

        for i in range(len(self.weights)):
            for j in range(len(self.weights[0])):
                self.delta_weights[i][j] += self.passed_on_loss_array[i]*self.previous_layer_outputs[j]

        loss_to_pass = [0] * len(self.weights[0])
        for i in range(len(loss_to_pass)):
            for j in range(len(self.weights)):
                loss_to_pass[i] += self.passed_on_loss_array[j] * self.weights[j][i]
        self.loss_to_pass = loss_to_pass

    def update_w_and_b(self, batch_size, learning_rate):
        for i in range(len(self.delta_weights)):
            for j in range(len(self.delta_weights[0])):
                self.delta_weights[i][j] /= batch_size
                self.delta_weights[i][j] *= learning_rate
                self.weights[i][j] -= self.delta_weights[i][j]
        for i in range(len(self.biases)):
            self.delta_biases[i] /= batch_size
            self.delta_biases[i] *= learning_rate
            self.biases[i] -= self.delta_biases[i]

        self.delta_biases = [0] * len(self.biases)
        self.delta_weights = [[0] * len(self.weights[0]) for _ in range(len(self.weights))]


#literally the sum
classification_data = [[2, 9, 1], [4, -7, 4], [4, 7, -6], [-1, -3, -4], [7, -6, 5], [-8, -9, -6], [-1, 10, 8], [9, -8, 4], [-3, 2, -6], [0, 2, 4], [-5, -8, -7], [-10, -3, -10], [2, 8, -7], [1, -5, -1], [2, 9, 4], [8, 9, -8], [8, -3, 8], [-7, 6, -9], [0, 7, 0], [7, -4, 5], [7, 6, 2], [6, -1, -8], [-5, -7, 7], [2, -8, -7], [-7, -3, 5], [-1, -3, -3], [-8, -3, -8], [-10, -7, -9], [1, -8, 4], [1, 9, 2], [-4, 4, 9], [-6, -9, -4], [1, 7, 6], [-4, 2, -4], [-9, -5, -5], [-10, -6, -7], [-6, -9, 9], [-4, -5, 10], [7, 5, -6], [0, -2, -6], [9, 1, -8], [4, -6, -5], [-7, 10, 9], [4, -6, -2], [1, 7, -4], [1, -3, 9], [6, 5, 8], [-8, 6, 9], [-2, -5, -7], [0, 7, -9], [-8, -9, 1], [-5, 4, -2], [0, 1, -2], [7, -2, 5], [-8, 3, 9], [4, 0, -6], [8, -6, -3], [5, -3, -3], [-7, 8, -4], [9, -7, 10], [5, -10, -8], [-8, -7, 0], [3, -7, -8], [4, -8, 4], [0, 4, 7], [-10, 8, 9], [-2, 9, -9], [0, 4, 2], [-3, -10, 2], [-1, 2, 0], [3, 0, 8], [8, -3, -9], [-6, -7, 7], [10, 2, -9], [8, 9, -2], [-2, -10, 3], [6, 8, 3], [9, 1, -9], [4, -6, 3], [1, -6, -2], [9, -3, 7], [-7, -3, 3], [2, 1, -6], [-3, -9, -10], [-5, 5, -2], [7, 6, 10], [-9, 4, -6], [-9, -8, 2], [7, 3, -5], [3, 6, -1], [-10, -7, 9], [7, -8, 2], [-9, 5, 2], [-5, 2, 3], [-9, 5, 2], [-6, 0, 4], [-9, -2, -7], [-2, -1, -8], [-6, 0, -9], [8, 4, -8], [-1, -9, -4], [-6, -6, 4], [-6, -6, 2], [-1, 7, 4], [8, 5, 3], [0, -4, 3], [7, -6, -7], [0, 4, 8], [5, -1, -6], [2, -7, -6], [3, 9, -7], [-5, 6, -3], [4, 6, 4], [-1, 1, 6], [-4, -9, -3], [-4, -3, 1], [3, 7, 8], [6, -6, 5], [-2, 3, -3], [1, 6, 8], [7, 7, 7], [8, 10, -4], [6, -3, 0], [7, 7, 2], [9, 7, 5], [1, -6, 1], [10, -6, -4], [-10, 9, 4], [-1, -2, 2], [9, 1, 1], [5, -1, -2], [-9, 3, -3], [-7, 2, 7], [2, -7, -1], [3, 7, -8], [-2, -9, 7], [4, -1, -2], [-1, -4, -7], [-2, -2, -10], [4, -7, 9], [9, 3, -6], [5, -8, -5], [-7, -8, 6], [-2, 3, 9], [4, -10, 0], [-9, -2, 3], [-3, 3, 2], [5, 0, 6], [-3, 4, 3], [-7, 8, -7], [9, -4, 10], [1, -3, -4], [-5, 3, -6], [-4, -5, 0], [7, -6, -8], [-9, 0, 1], [-2, 8, 7], [-5, -8, -7], [8, -2, -8], [8, -3, -1], [1, 1, -5], [6, 6, 3], [-1, -4, 4], [-1, 1, 1], [4, -7, -9], [3, 2, 5], [-9, 3, 4], [6, -4, 1], [7, 7, 7], [4, 6, 10], [1, -8, 4], [10, 9, -6], [7, -1, -10], [10, 10, 6], [3, 10, 7], [8, 7, 2], [-1, -9, -2], [3, 6, 10], [8, 6, -8], [-6, -6, -7], [-6, -9, -6], [2, -1, 5], [-6, -5, 3], [6, -3, -4], [7, -6, -7], [7, -10, 8], [-5, -1, 6], [3, 7, 1], [3, 0, 3], [8, 3, -7], [-3, 2, 2], [-2, 8, -9], [9, 3, 7], [-1, -7, 1], [2, -4, 1], [8, 6, 8], [9, 0, -4], [-6, -9, -2], [9, 4, 3], [-8, 2, -7], [7, 3, -5], [-5, 5, -7], [-9, 9, -9], [0, 8, -10], [-5, 10, 6], [-7, 1, -2], [-9, 5, -9], [-4, -2, -3], [-8, -1, -1], [1, 9, -5], [-2, 4, 6], [-10, 5, -5], [1, 1, -5], [6, 9, -2], [-3, -4, -7], [-9, -8, 0], [-2, -7, 0], [8, -5, 10], [10, -2, 6], [4, -3, 3], [2, 5, 9], [6, -3, -5], [1, -8, -5], [6, -7, 9], [-2, 3, -9], [-5, 0, 7], [1, 8, 2], [6, 8, 5], [-2, -6, 0], [0, 0, -2], [2, -4, 4], [-4, 7, -1], [-9, 5, -9], [4, 5, 5], [9, -9, 6], [-3, 5, 3], [9, -2, -1], [0, -7, 2], [3, -8, -3], [0, 0, 9], [7, -5, 10], [-9, -9, -7], [1, 5, -8], [-3, -7, 5], [1, -9, -5], [0, -5, 10], [5, 3, -3], [5, 9, 4], [2, -8, -2], [7, 0, 2], [-4, -6, 2], [5, 4, -1], [-4, -2, -2], [3, 10, 9], [1, -4, -10], [-3, 8, 7], [-6, -2, -5], [8, -5, 0], [-8, -2, -3], [8, 10, -6], [5, 6, -9], [-4, 4, 4], [8, -7, -9], [0, 9, -4], [7, 6, 6], [-2, -3, 1], [9, 8, 5], [-5, -2, 8], [-5, -9, -7], [2, 7, 9], [-8, 9, 5], [10, -4, 0], [9, 10, -6], [-4, 3, 4], [5, -4, 0], [-10, -1, -8], [0, 0, 7], [3, 1, 8], [7, 5, -6], [-8, -8, 8], [1, -7, 8], [-2, -8, -4], [-7, 8, -2], [0, 5, 8], [2, 2, 8], [-4, -6, -8], [-1, 8, -4], [-9, 5, 2], [-10, 0, 1], [1, 9, 9], [1, 1, 1], [-1, -5, -5], [9, 6, 10], [7, 5, 1], [5, -6, 8], [1, 9, 3], [1, 8, 1], [4, -1, -7], [-4, 6, 1], [2, 9, -4], [5, -8, -10], [8, -4, 7], [5, 10, 2], [-3, 0, 3], [7, 3, -9], [-1, -8, -5], [-9, -1, -7], [1, 2, 4], [-5, 5, 5], [-5, 10, 4], [10, -1, -1], [7, -3, 5], [0, 6, -3], [-2, -9, -2], [-6, 7, 2], [3, 9, 1], [3, -1, 4], [-4, -9, 5], [-4, 2, 6], [4, -8, 7], [-1, 4, -8], [-4, -3, 10], [-2, -3, -3], [-2, 9, 5], [2, -10, 6], [1, -3, 6], [-9, 7, -9], [6, -5, 2], [10, -8, 6], [6, 4, -3], [2, 2, -3], [5, -5, 7], [-5, -5, 9], [1, 9, 6], [-3, 4, -3], [-9, -7, -4], [-7, 2, -8], [-6, -6, 0], [-4, 1, -9], [-8, -4, 3], [7, -7, 4], [-2, -5, 5], [0, 6, 1], [4, 4, -4], [8, 7, 5], [-9, 0, -4], [-9, 6, 3], [-6, 4, 3], [6, -4, 8], [-5, 0, 6], [8, -7, -3], [2, 5, 4], [-3, -4, -6], [8, -5, -6], [5, -3, 1], [1, -9, -9], [7, 2, -6], [-10, 6, 7], [9, 5, 8], [9, 4, -9], [-6, 6, 7], [9, -7, -8], [-9, 4, 4], [2, 7, 8], [-1, 1, 4], [9, -4, -4], [6, 7, 0], [-1, -8, 9], [1, -1, 7], [6, 1, -3], [0, 4, 6], [2, -4, 4], [5, 2, 7], [8, 2, -5], [-10, 5, 9], [1, 1, 1], [3, 0, 1], [5, -6, 9], [-2, 5, 6], [4, -2, 6], [5, -2, 1], [-3, 3, -4], [7, 9, 1], [6, 2, 0], [-6, 2, 0], [9, 6, -6], [-1, -2, -2], [-7, 9, 9], [8, 4, 0], [0, -4, -6], [-6, -7, 8], [7, -10, 0], [4, -3, 4], [-2, 1, 9], [-3, -3, 3], [4, -5, -2], [-7, 1, 4], [9, -4, 4], [4, -7, -4], [-9, 0, 2], [6, 4, 8], [-6, 2, 10], [4, 5, 6], [8, 1, 2], [6, -4, -3], [-9, 7, -6], [5, 5, 8], [-8, -1, -1], [6, 8, -5], [-7, 10, 4], [-6, -4, -2], [6, -9, 10], [-9, -2, -3], [3, 0, 0], [-5, 3, -8], [9, -5, 3], [6, -5, 9], [0, 1, -7], [-9, -6, 1], [-3, -3, -4], [-5, -2, 4], [5, -8, -8], [10, -5, 2], [9, -7, 0], [1, 4, -7], [-9, -4, 7], [2, 1, 1], [-8, -8, -7], [3, 1, 4], [-5, -3, 4], [7, -4, -4], [-5, 4, 6], [6, 9, -7], [3, 3, 5], [-7, 5, -7], [-2, 4, 0], [6, 3, 5], [3, 1, -2], [5, 8, 9], [9, 1, 4], [9, -6, 3], [-2, 9, -9], [-3, 7, -1], [1, -1, 3], [-8, -5, 4], [-5, 8, 2], [3, -7, 2], [3, -2, 1], [-10, -7, -5], [6, 5, 0], [-2, -10, 1], [1, 3, -3], [4, -1, 7], [-9, -9, 10], [-2, -1, -9], [-8, -8, 5], [3, 5, 4], [-7, 6, -2], [8, 5, -6], [1, 3, -1], [1, 0, 2], [4, 9, 2], [-9, -4, 5], [-3, -1, 6], [-9, 5, 3], [-9, -7, 4], [-9, 8, -3], [-3, 7, -4], [-8, 1, -4], [0, -10, -2], [7, -5, -2], [8, -4, -10], [-5, 6, 7], [-9, 9, -8], [7, -5, -10], [10, -10, 5], [9, 4, 5], [-7, 9, -8], [-2, -6, 6], [-8, 5, -4], [-8, -1, -5], [7, -7, -2], [4, 7, -3], [-8, -8, -4], [-2, 6, 5], [-1, -6, -9], [1, 3, 3], [-3, -8, 5], [9, -8, -5], [5, -5, -2], [6, 3, -6], [6, 5, 7], [0, -6, 2], [-6, -2, -3], [-9, 1, 7], [5, 0, -6], [3, -6, -1], [-8, -8, -10], [-4, -5, 0], [7, -10, 5]]
classification_answers = [[12], [1], [5], [-8], [6], [-23], [17], [5], [-7], [6], [-20], [-23], [3], [-5], [15], [9], [13], [-10], [7], [8], [15], [-3], [-5], [-13], [-5], [-7], [-19], [-26], [-3], [12], [9], [-19], [14], [-6], [-19], [-23], [-6], [1], [6], [-8], [2], [-7], [12], [-4], [4], [7], [19], [7], [-14], [-2], [-16], [-3], [-1], [10], [4], [-2], [-1], [-1], [-3], [12], [-13], [-15], [-12], [0], [11], [7], [-2], [6], [-11], [1], [11], [-4], [-6], [3], [15], [-9], [17], [1], [1], [-7], [13], [-7], [-3], [-22], [-2], [23], [-11], [-15], [5], [8], [-8], [1], [-2], [0], [-2], [-2], [-18], [-11], [-15], [4], [-14], [-8], [-10], [10], [16], [-1], [-6], [12], [-2], [-11], [5], [-2], [14], [6], [-16], [-6], [18], [5], [-2], [15], [21], [14], [3], [16], [21], [-4], [0], [3], [-1], [11], [2], [-9], [2], [-6], [2], [-4], [1], [-12], [-14], [6], [6], [-8], [-9], [10], [-6], [-8], [2], [11], [4], [-6], [15], [-6], [-8], [-9], [-7], [-8], [13], [-20], [-2], [4], [-3], [15], [-1], [1], [-12], [10], [-2], [3], [21], [20], [-3], [13], [-4], [26], [20], [17], [-12], [19], [6], [-19], [-21], [6], [-8], [-1], [-6], [5], [0], [11], [6], [4], [1], [-3], [19], [-7], [-1], [22], [5], [-17], [16], [-13], [5], [-7], [-9], [-2], [11], [-8], [-13], [-9], [-10], [5], [8], [-10], [-3], [13], [-14], [-17], [-9], [13], [14], [4], [16], [-2], [-12], [8], [-8], [2], [11], [19], [-8], [-2], [2], [2], [-13], [14], [6], [5], [6], [-5], [-8], [9], [12], [-25], [-2], [-5], [-13], [5], [5], [18], [-8], [9], [-8], [8], [-8], [22], [-13], [12], [-13], [3], [-13], [12], [2], [4], [-8], [5], [19], [-4], [22], [1], [-21], [18], [6], [6], [13], [3], [1], [-19], [7], [12], [6], [-8], [2], [-14], [-1], [13], [12], [-18], [3], [-2], [-9], [19], [3], [-11], [25], [13], [7], [13], [10], [-4], [3], [7], [-13], [11], [17], [0], [1], [-14], [-17], [7], [5], [9], [8], [9], [3], [-13], [3], [13], [6], [-8], [4], [3], [-5], [3], [-8], [12], [-2], [4], [-11], [3], [8], [7], [1], [7], [-1], [16], [-2], [-20], [-13], [-12], [-12], [-9], [4], [-2], [7], [4], [20], [-13], [0], [1], [10], [1], [-2], [11], [-13], [-3], [3], [-17], [3], [3], [22], [4], [7], [-6], [-1], [17], [4], [1], [13], [0], [7], [4], [10], [2], [14], [5], [4], [3], [4], [8], [9], [8], [4], [-4], [17], [8], [-4], [9], [-5], [11], [12], [-10], [-5], [-3], [5], [8], [-3], [-3], [-2], [9], [-7], [-7], [18], [6], [15], [11], [-1], [-8], [18], [-10], [9], [7], [-12], [7], [-14], [3], [-10], [7], [10], [-6], [-14], [-10], [-3], [-11], [7], [2], [-2], [-6], [4], [-23], [8], [-4], [-1], [5], [8], [11], [-9], [2], [14], [2], [22], [14], [6], [-2], [3], [3], [-9], [5], [-2], [2], [-22], [11], [-11], [1], [10], [-8], [-12], [-11], [12], [-3], [7], [3], [3], [15], [-8], [2], [-1], [-12], [-4], [0], [-11], [-12], [0], [-6], [8], [-8], [-8], [5], [18], [-6], [-2], [-7], [-14], [-2], [8], [-20], [9], [-16], [7], [-6], [-4], [-2], [3], [18], [-4], [-11], [-1], [-1], [-4], [-26], [-9], [2]]

testing_data = [[8, -3, -7], [0, -7, -2], [1, 3, 6], [4, -4, 3], [1, 4, 7], [-4, 4, 7], [-5, -1, -5], [-7, 9, 2], [10, 9, -3], [0, -2, 2], [4, -3, 1], [-1, 9, 1], [9, -1, 6], [0, 6, -3], [-4, 2, -8], [1, 4, -2], [-2, 4, 7], [2, -3, -7], [5, -8, -7], [8, -7, 0], [7, 1, -1], [-4, 5, 9], [1, -10, 4], [-8, -7, 9], [-8, 9, 5], [-7, -8, -6], [9, 7, 7], [-4, -3, -3], [-4, 5, 10], [-7, -1, -10], [6, 8, 1], [0, 10, 1], [-3, 7, 9], [-7, 3, -2], [9, 5, 7], [-9, -4, 5], [-6, 3, -9], [5, 1, 7], [-10, -2, -5], [2, 5, 9], [7, 5, 3], [3, 2, -1], [6, -3, -2], [-5, 9, -9], [-8, -6, 1], [8, -3, -5], [-7, 2, 6], [-9, -1, -9], [-6, 0, -10], [-2, -6, -3]]
testing_answers = [[-2], [-9], [10], [3], [12], [7], [-11], [4], [16], [0], [2], [9], [14], [3], [-10], [3], [9], [-8], [-10], [1], [7], [10], [-5], [-6], [6], [-21], [23], [-10], [11], [-18], [15], [11], [13], [-6], [21], [-8], [-12], [13], [-17], [16], [15], [4], [1], [-5], [-13], [0], [1], [-19], [-16], [-11]]


class NN:
    def __init__(self, input_size, inner_layers_number, height, output_size, inner_layer_activation, last_layer_activation):
        self.inner_layer_activation = inner_layer_activation
        self.last_layer_activation = last_layer_activation
        self.layers = [[]] * (inner_layers_number + 2) 
        self.layers[0] = Layer(input_size, height)
        self.layers[-1] = Layer(height, output_size)
        for i in range(inner_layers_number):
            self.layers[i+1] = Layer(height, height)
    
    def train(self, epochs, learning_rate, data_train, data_output, is_testing, batch_size):
        current_epoch = 0
        current_batch = 0
        for i in range(epochs):
            current_epoch_loss = 0
            for j in range(len(data_train)):
                self.layers[0].forward(data_train[j])
                self.layers[0].activation_function(self.inner_layer_activation, False)
                for k in range(len(self.layers)-2):
                    self.layers[k+1].forward(self.layers[k].outputs)
                    self.layers[k+1].activation_function(self.inner_layer_activation, False)
                self.layers[-1].forward(self.layers[-2].outputs)
                self.layers[-1].activation_function(self.last_layer_activation, True)
                # print(f"Outputted: {self.layers[-1].outputs}")
                # print(f"Actual: {data_output[j]}")
                #now for loss
                if is_testing == False:
                    self.layers[-1].loss(self.layers[-1].outputs, data_output[j],"mse")
                    print(f"Loss: {self.layers[-1].mean_loss}")
                    #now for back prop
                    self.layers[-1].back_prop(self.layers[-1].d_loss)
                    for l in range(len(self.layers)-1):
                        self.layers[-l-2].back_prop(self.layers[-l-1].loss_to_pass)
                    current_batch += 1
                    if current_batch == batch_size:
                        current_batch = 0
                        for i in range(len(self.layers)):
                            self.layers[i].update_w_and_b(batch_size, learning_rate)
                elif is_testing == True:
                    print("----- TESTING -----")
                    if self.last_layer_activation != "Softmax":
                        self.layers[-1].loss(self.layers[-1].outputs, data_output[j],"mse")
                        current_epoch_loss
                    elif self.last_layer_activation == "Softmax":
                        #fix to use max value instead of rounded
                        print(self.layers[-1].outputs)
                        predicted_ans = [0] * len(self.layers[-1].outputs)
                        index = predicted_ans.index(max(predicted_ans))
                        for i in range(len(predicted_ans)):
                            if i != index:
                                predicted_ans[i] = 0
                            else:
                                predicted_ans[i] = 1
                        # print(f"predicted: {predicted_ans}")
                        # print(f"actual: {data_output[j]}")
                        if predicted_ans == data_output[j]:
                            pass #fix this
            
            current_epoch += 1
            print(f"Epochs completed: {current_epoch}/{epochs}\nAverage epoch loss: {current_epoch_loss/len(data_train)}")

    def test(self, testing_data, testing_answers):
        self.train(1, 0, testing_data, testing_answers, True, 1000000000000000)


neural = NN(3, 1, 1, 1, "Leaky_ReLU", "Leaky_ReLU")
neural.train(10000, 0.01, classification_data, classification_answers, False, 32)
neural.test(testing_data, testing_answers)