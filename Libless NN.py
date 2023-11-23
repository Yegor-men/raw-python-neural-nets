import math
import random
E = math.e
random.seed(31415)


class Layer:
    def __init__(self, previous_height, height):
        self.biases = [(1*(random.random())*2-1) for n in range(height)]
        self.weights = [[(1*(random.random())*2-1) for n in range(previous_height)] for m in range(height)]
    
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
            loss = [math.log(prediced_list[i] - expected_list[i]) for i in range(len(prediced_list))]
            self.mean_loss = 0
            for i in range(len(loss)):
                self.mean_loss += loss[i]
        
    def back_prop(self, inputted_loss_array, learning_rate):
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
            self.biases[i] -= learning_rate * self.passed_on_loss_array[i]

        for i in range(len(self.weights)):
            for j in range(len(self.weights[0])):
                self.weights[i][j] -= self.passed_on_loss_array[i]*self.previous_layer_outputs[j]*learning_rate

        loss_to_pass = [0] * len(self.weights[0])
        for i in range(len(loss_to_pass)):
            for j in range(len(self.weights)):
                loss_to_pass[i] += self.passed_on_loss_array[j] * self.weights[j][i]
        self.loss_to_pass = loss_to_pass


#literally the sum


classification_data = [[1,2,3],[0,2,3]]
classification_answers = [[6],[5]]

testing_data = [[1,2,3],[0,2,3]]
testing_answers = [[6],[5]]

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
                self.nn[0].activation_function(self.inner_layer_activation, False)
                for k in range(len(self.nn)-2):
                    self.nn[k+1].forward(self.nn[k].outputs)
                    self.nn[k+1].activation_function(self.inner_layer_activation, False)
                self.nn[-1].forward(self.nn[-2].outputs)
                self.nn[-1].activation_function(self.last_layer_activation, True)
                print(f"Outputted: {self.nn[-1].outputs}")
                print(f"Actual: {data_output[j]}")
                #now for loss
                if is_testing == False:
                    self.nn[-1].loss(self.nn[-1].outputs, data_output[j],"mse")
                    #now for back prop
                    self.nn[-1].back_prop(self.nn[-1].d_loss, learning_rate)
                    for l in range(len(self.nn)-1):
                        self.nn[-l-2].back_prop(self.nn[-l-1].loss_to_pass, learning_rate)
                elif is_testing == True:
                    if self.last_layer_activation != "Softmax":
                        self.nn[-1].loss(self.nn[-1].outputs, data_output[j],"mse")
                        current_accuracy_measure += self.nn[-1].mean_loss
                    elif self.last_layer_activation == "Softmax":
                        #fix to use max value instead of rounded
                        print(self.nn[-1].outputs)
                        predicted_ans = [0] * len(self.nn[-1].outputs)
                        index = predicted_ans.index(max(predicted_ans))
                        for i in range(len(predicted_ans)):
                            if i != index:
                                predicted_ans[i] = 0
                            else:
                                predicted_ans[i] = 1
                        print(f"predicted: {predicted_ans}")
                        print(f"actual: {data_output[j]}")
                        if predicted_ans == data_output[j]:
                            current_accuracy_measure += 1
            print(f"Epochs completed: {current_epoch}/{epochs}")
            current_epoch += 1
            
        
        if is_testing == True:
            if self.last_layer_activation == "Softmax":
                print(f"Percentage correct: {100*current_accuracy_measure/len(data_train)}")
            else:
                print(f"Average loss: {current_accuracy_measure/len(data_train)}")

    def test(self, testing_data, testing_answers):
        self.train(1, 0, testing_data, testing_answers, True)


neural = NN(3, 1, 1, 1, "Leaky_ReLU", "Leaky_ReLU")
neural.train(3300, 0.01, classification_data, classification_answers, False)
neural.test(testing_data, testing_answers)