import math
import random
E = math.e
random.seed(1)


class Layer:
    def __init__(self, previous_height, height):
        self.biases = []
        self.biases = [(random.random())*2-1 for n in range(height)]
        self.weights = []
        self.weights = [[(random.random())*2-1 for n in range(previous_height)] for m in range(height)]
    
    def forward(self, previous_layer_outputs):
        self.outputs = [0]*len(self.biases)
        for j in range(len(self.biases)):
            for k in range(len(self.weights[j])):
                self.outputs[j] += ((previous_layer_outputs[k]) * (self.weights[j][k])) + self.biases[j]
    
    def activation_function(self, activation_function_type):
        self.activation_function_type = activation_function_type
        if self.activation_function_type == "ReLU":
            self.outputs = [max(0, n) for n in self.outputs]
        if self.activation_function_type == "Leaky_ReLU":
            self.outputs = [0.1*n if n<0 else n for n in self.outputs]
        if self.activation_function_type == "Softmax":
            maximum_output = max(self.outputs)
            self.outputs = [self.outputs[n]-maximum_output for n in range(len(self.outputs))]
            sum = 0
            for i in range(len(self.outputs)):
                sum += E**(self.outputs[i])
            for i in range(len(self.outputs)):
                self.outputs[i] = (E**(self.outputs[i]))/sum

    def loss(self, prediced_list, expected_list):
        self.mean_loss = 0
        for i in range(len(prediced_list)):
            self.mean_loss += (prediced_list[i] - expected_list[i])**2
        self.mean_loss /= 2
        self.d_loss = [prediced_list[i] - expected_list[i] for i in range(len(prediced_list))]
    
    def back_prop(self, inputted_loss_array, learning_rate):
        self.passed_on_loss_array = inputted_loss_array
        if self.activation_function_type == "ReLU":
            for i in range(len(self.passed_on_loss_array)):
                if self.passed_on_loss_array[i]<1:
                    self.passed_on_loss_array[i] = 0
        if self.activation_function_type == "Leaky_ReLU":
            for i in range(len(self.passed_on_loss_array)):
                if self.passed_on_loss_array[i] < 0:
                    self.passed_on_loss_array[i] *= 0.1
        if self.activation_function_type == "Softmax":
            for i in range(len(self.passed_on_loss_array)):
                self.passed_on_loss_array[i] *= self.passed_on_loss_array[i]
        for i in range(len(self.biases)):
            self.biases[i] -= self.biases[i] * learning_rate * self.passed_on_loss_array[i]
            d_w = []
            for j in range(len(self.weights[0])):
                d_w.append(self.weights[i][j]*self.passed_on_loss_array[i])
            for k in range(len(self.weights[0])):
                self.weights[i][k] -= self.weights[i][k]*learning_rate
        # calcs for next loss passed array
        loss_to_pass = [0] * len(self.weights[0])
        for i in range(len(loss_to_pass)):
            for j in range(len(self.weights)):
                loss_to_pass[i] += loss_to_pass[i]*self.weights[j][i]
        self.loss_to_pass = loss_to_pass


#Data for checking if multiplication is odd or even
classification_data = [[1,2],
              [3,3],
              [6,1],
              [3,7],
              [2,3],
              [8,4],
              [5,5]]
classification_answers = [[0,1],
                 [1,0],
                 [0,1],
                 [1,0],
                 [0,1],
                 [0,1],
                 [1,0]]

class NN:
    def __init__(self, input_size, inner_layers_number, height, output_size, inner_layer_activation, last_layer_activation):
        self.inner_layer_activation = inner_layer_activation
        self.last_layer_activation = last_layer_activation
        self.nn = [[]] * (inner_layers_number + 2)   
        self.nn[0] = Layer(input_size, height)
        self.nn[-1] = Layer(height, output_size)
        for i in range(inner_layers_number):
            self.nn[i+1] = Layer(height, height)
    
    def train(self, epochs, learning_rate, data_train, data_output):
        current_epoch = 0
        recurring_avg_loss = 0
        for i in range(epochs):
            for j in range(len(data_train)):
                self.nn[0].forward(data_train[j])
                self.nn[0].activation_function(self.inner_layer_activation)
                for k in range(len(self.nn)-2):
                    self.nn[k+1].forward(self.nn[k].outputs)
                    self.nn[k+1].activation_function(self.inner_layer_activation)
                self.nn[-1].forward(self.nn[-2].outputs)
                self.nn[-1].activation_function(self.last_layer_activation)
                #now for loss
                self.nn[-1].loss(self.nn[-1].outputs, data_output[j])
                recurring_avg_loss += self.nn[-1].mean_loss
                #now for back prop
                self.nn[-1].back_prop(self.nn[-1].d_loss, learning_rate)
                for l in range(len(self.nn)-1):
                    self.nn[-l-2].back_prop(self.nn[-l-1].loss_to_pass, learning_rate)
            current_epoch += 1
            print(f"Current epoch: {current_epoch}, current avg_loss for batch: {recurring_avg_loss}")
            recurring_avg_loss = 0


neural = NN(2, 10, 20, 2, "Leaky_ReLU", "Softmax")
neural.train(1000, 0.05, classification_data, classification_answers)