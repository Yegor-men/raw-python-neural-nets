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


#literally the sum
classification_data = [[2, -1, 10], [3, -10, -2], [6, 3, -6], [-9, -1, 3], [2, -1, 0], [-3, -1, 9], [7, 8, 7], [-4, -9, -4], [1, 6, -2], [4, -3, -4], [-4, -6, 1], [3, -3, -7], [-10, -7, -8], [6, 3, 5], [3, -3, -7], [3, -1, 9], [-4, 10, 9], [7, -8, 4], [8, -8, 7], [3, 5, 0], [-7, 4, 9], [6, 7, -8], [-10, 3, 5], [-2, -5, -1], [-2, 0, 6], [-1, 0, 4], [5, 1, 10], [-10, 2, -10], [5, 9, 7], [0, 4, 0], [-5, -6, 9], [8, 4, 6], [1, -2, -9], [-3, -5, -8], [-7, 2, 5], [-4, 0, -10], [6, -3, 1], [3, -4, -2], [6, 4, -10], [-8, 2, 5], [8, -6, -9], [6, 1, 0], [-6, 8, -9], [-9, 4, -2], [9, 4, 0], [-9, 2, 9], [7, 8, 2], [9, -8, 7], [6, 10, 4], [10, -1, -4], [0, -5, 4], [-4, -4, -2], [1, 6, -6], [-6, 9, 5], [1, -8, 3], [-1, 4, 2], [2, 8, -6], [-8, -7, -4], [-2, -7, 2], [-6, -8, -2], [-3, -2, 9], [-4, -6, -7], [7, -7, 3], [-5, -2, 8], [2, 10, -7], [-4, -8, 1], [-9, -6, 6], [-8, 0, 8], [8, 7, -10], [-9, 9, 7], [-1, 5, -3], [-2, -5, -6], [9, -10, 6], [-6, 8, 5], [-3, -4, -8], [5, 10, -3], [-4, -5, 7], [2, -3, -3], [-7, 6, 9], [-9, -2, -6], [-10, 4, 0], [9, -1, -4], [3, 1, -5], [9, -9, -6], [1, -6, -7], [-2, 9, -9], [-2, -1, -7], [2, -5, 4], [-5, 6, 0], [0, -1, -2], [8, -7, -7], [5, 8, 2], [-3, -5, 10], [-4, -5, -1], [-7, 2, 9], [-4, -4, -8], [8, -7, -5], [3, -8, 4], [-2, 5, 8], [-6, 6, 10]]
classification_answers = [[11], [-9], [3], [-7], [1], [5], [22], [-17], [5], [-3], [-9], [-7], [-25], [14], [-7], [11], [15], [3], [7], [8], [6], [5], [-2], [-8], [4], [3], [16], [-18], [21], [4], [-2], [18], [-10], [-16], [0], [-14], [4], [-3], [0], [-1], [-7], [7], [-7], [-7], [13], [2], [17], [8], [20], [5], [-1], [-10], [1], [8], [-4], [5], [4], [-19], [-7], [-16], [4], [-17], [3], [1], [5], [-11], [-9], [0], [5], [7], [1], [-13], [5], [7], [-15], [12], [-2], [-4], [8], [-17], [-6], [4], [-1], [-6], [-12], [-2], [-10], [1], [1], [-3], [-6], [15], [2], [-10], [4], [-16], [-4], [-1], [11], [10]]

testing_data = [[-6, 1, 6], [2, 9, -10], [-9, 9, -7], [3, 3, -9], [0, 2, 1], [9, -5, -1], [9, 7, 3], [0, -1, -7], [4, 5, -2], [8, -7, -7], [-3, -6, 5], [-3, -8, 8], [-1, 4, 3], [6, -4, -7], [-4, -6, 4], [-5, 1, 1], [4, -1, 5], [-7, 7, -9], [3, -7, -4], [5, -9, -8], [-9, -1, -7], [-6, -6, 6], [-5, -5, 6], [4, 7, -4], [5, 4, 1], [2, 5, 6], [1, -6, -2], [8, 6, -7], [0, 0, 3], [-3, -1, 0]]
testing_answers = [[1], [1], [-7], [-3], [3], [3], [19], [-8], [7], [-6], [-4], [-3], [6], [-5], [-6], [-3], [8], [-9], [-8], [-12], [-17], [-6], [-4], [7], [10], [13], [-7], [7], [3], [-4]]

class NN:
    def __init__(self, input_size, inner_layers_number, height, output_size, inner_layer_activation, last_layer_activation):
        self.inner_layer_activation = inner_layer_activation
        self.last_layer_activation = last_layer_activation
        self.nn = [[]] * (inner_layers_number + 2)   
        self.nn[0] = Layer(input_size, height)
        self.nn[-1] = Layer(height, output_size)
        for i in range(inner_layers_number):
            self.nn[i+1] = Layer(height, height)
    
    def train(self, epochs, learning_rate, data_train, data_output, is_testing):
        current_epoch = 0
        current_accuracy_measure = 0

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
                if is_testing == False:
                    self.nn[-1].loss(self.nn[-1].outputs, data_output[j])
                    #current_accuracy_measure += self.nn[-1].mean_loss
                    #now for back prop
                    self.nn[-1].back_prop(self.nn[-1].d_loss, learning_rate)
                    for l in range(len(self.nn)-1):
                        self.nn[-l-2].back_prop(self.nn[-l-1].loss_to_pass, learning_rate)
                elif is_testing == True:
                    if self.last_layer_activation != "Softmax":
                        self.nn[-1].loss(self.nn[-1].outputs, data_output[j])
                        current_accuracy_measure += self.nn[-1].mean_loss
                    elif self.last_layer_activation == "Softmax":
                        print(self.nn[-1].outputs)
                        predicted_ans = [round(self.nn[-1].outputs[n]) for n in range(len(self.nn[-1].outputs))]
                        print(f"predicted: {predicted_ans}")
                        print(f"actual: {data_output[j]}")
                        if predicted_ans == data_output[j]:
                            current_accuracy_measure += 1
            current_epoch += 1
            print(f"Epochs completed: {round(100*current_epoch/epochs,2)}%")
        
        if is_testing == True:
            if self.last_layer_activation == "Softmax":
                print(f"Percentage correct: {100*current_accuracy_measure/len(data_train)}")
            else:
                print(f"Average loss: {current_accuracy_measure/len(data_train)}")

    def test(self, testing_data, testing_answers):
        self.train(1, 0, testing_data, testing_answers, True)


neural = NN(3, 2, 16, 1, "Leaky_ReLU", "Leaky_ReLU")
neural.train(100, 0.01, classification_data, classification_answers, False)
neural.test(testing_data, testing_answers)