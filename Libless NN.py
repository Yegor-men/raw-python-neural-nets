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
                #self.passed_on_loss_array[i] *= E**self.passed_on_loss_array[i]

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


#Sin
classification_data = [[-2.86], [-0.1], [2.56], [0.6], [-0.5], [0.05], [-0.25], [-2.84], [1.84], [-0.44], [-2.5], [1.81], [1.06], [-1.81], [-0.4], [1.37], [-1.41], [-0.11], [-1.95], [-2.96], [2.02], [2.14], [-1.74], [1.5], [2.58], [1.84], [-0.06], [-0.56], [-2.91], [-2.04], [-0.84], [-1.14], [-0.8], [-2.23], [-1.21], [1.48], [0.85], [2.47], [0.87], [-1.98], [0.02], [-2.27], [1.52], [-2.73], [-1.29], [1.39], [-2.03], [-1.06], [-1.04], [-1.5], [1.71], [-1.78], [2.73], [1.37], [1.58], [0.45], [-1.48], [-1.39], [-1.0], [1.11], [-0.78], [2.58], [1.29], [-1.59], [0.87], [-0.21], [-2.73], [0.8], [-2.7], [-1.11], [0.64], [2.39], [1.17], [0.36], [-0.0], [-0.12], [-2.16], [-2.21], [2.29], [-2.12], [-1.09], [2.89], [-2.59], [0.12], [2.06], [-0.56], [1.07], [0.86], [0.44], [1.68], [1.87], [2.11], [-0.01], [0.73], [1.43], [-2.47], [0.22], [-0.39], [1.52], [0.65], [-2.78], [1.67], [1.37], [1.39], [-1.34], [-0.61], [-1.23], [-1.84], [-0.69], [-0.22], [1.38], [-1.67], [-1.06], [0.53], [-1.4], [-0.27], [1.46], [-0.99], [1.63], [2.29], [-0.74], [1.4], [0.19], [-0.09], [-2.32], [-1.29], [0.19], [-1.52], [0.75], [-1.57], [1.42], [-2.1], [0.17], [-2.74], [-2.56], [-0.44], [-2.48], [-2.51], [-1.46], [2.54], [-0.08], [0.57], [-1.39], [-1.52], [1.4], [1.25], [1.41], [-2.44], [-2.47], [-2.01], [1.14], [-2.82], [1.91], [0.74], [1.04], [-0.05], [2.62], [-2.63], [1.0], [2.62], [-0.09], [2.62], [-0.65], [-2.03], [1.72], [0.19], [-1.24], [0.32], [-0.35], [2.89], [-1.82], [-1.35], [-0.91], [1.9], [-2.92], [-0.59], [1.42], [-1.17], [1.16], [2.57], [-2.7], [-2.8], [-0.39], [-2.59], [1.32], [1.65], [-1.75], [-0.52], [1.75], [1.82], [-1.82], [0.39], [-2.3], [-0.55], [2.26], [-2.46], [-0.57], [-0.96], [-0.69], [-0.76], [0.18], [0.44], [0.08], [-2.57], [-2.86], [-1.16], [2.99], [2.74], [-1.0], [-2.86], [2.19], [-2.26], [-1.42], [1.31], [-1.38], [-1.19], [-1.9], [-0.5], [2.0], [-1.59], [1.18], [-1.25], [-0.14], [-2.07], [-2.85], [0.87], [-2.81], [1.94], [1.06], [0.32], [0.77], [0.34], [-0.76], [1.34], [1.26], [-0.0], [1.36], [-0.46], [2.36], [1.65], [-2.58], [2.43], [2.82], [-1.58], [1.79], [0.55], [-0.88], [-2.15], [0.3], [2.3], [-1.3], [0.69], [-2.53], [-1.29], [2.69], [1.0], [0.05], [-1.87], [-2.14], [2.07], [-1.59], [-2.59], [2.2], [2.89], [-1.71], [-2.62], [1.75], [0.08], [0.76], [2.98], [-1.64], [-2.21], [-0.11], [2.02], [-1.94], [2.59], [0.58], [0.88], [1.49], [-0.56], [-0.24], [2.82], [2.43], [1.1], [1.85], [-2.87], [-1.91], [2.9], [-1.87], [0.42], [-2.25], [2.09], [-2.57], [-2.03], [0.32], [2.82], [0.47], [-1.37], [-1.33], [1.34], [-2.41], [0.54], [0.11], [-1.68], [-0.65], [1.31], [0.55], [2.77], [-1.3], [-2.47], [2.49], [-2.89], [-1.44], [-2.63], [1.56], [-0.58], [-0.82], [-2.68], [-0.88], [-2.63], [-2.47], [1.13], [2.73], [0.4], [0.88], [-1.86], [-2.25], [-1.01], [0.32], [0.73], [0.61], [0.49], [-2.15], [-0.92], [1.17], [0.57], [-2.5], [1.33], [-0.39], [-1.33], [0.17], [-1.77], [-0.35], [0.54], [-2.51], [2.48], [-0.8], [2.4], [2.64], [1.42], [2.61], [-2.16], [-2.7], [2.64], [0.43], [2.03], [-2.93], [1.45], [2.33], [-0.17], [-0.75], [2.9], [-1.85], [-2.96], [-1.32], [-1.22], [2.19], [-2.57], [1.56], [-0.13], [-2.4], [1.7], [-1.46], [-1.2], [0.58], [1.22], [2.61], [2.43], [-1.03], [1.96], [2.98], [0.97], [2.18], [-0.96], [-1.27], [0.86], [-1.46], [0.57], [0.99], [-2.99], [-0.01], [-1.68], [1.68], [-1.07], [2.07], [-0.24], [2.23], [0.62], [2.35], [-2.64], [-1.21], [1.75], [-0.95], [-0.35], [-2.95], [-0.74], [-2.74], [1.41], [2.42], [-1.67], [-1.79], [0.45], [0.81], [-0.6], [0.95], [1.77], [-1.67], [-0.6], [2.34], [1.04], [2.73], [0.43], [-1.49], [2.12], [-0.62], [-2.27], [-1.94], [-2.68], [1.12], [-0.94], [2.47], [-2.66], [0.29], [0.91], [-2.37], [2.18], [-0.56], [-0.77], [2.75], [-0.97], [-2.56], [0.85], [-0.99], [-0.97], [2.53], [-0.36], [1.24], [1.1], [0.59], [-2.63], [0.91], [-2.59], [2.92], [-2.08], [-0.62], [-0.65], [0.49], [0.54], [1.06], [-1.72], [1.97], [0.72], [-2.28], [-1.19], [1.39], [1.83], [-1.36], [2.32], [-0.97], [-0.46], [-0.99], [0.39], [1.78], [-0.91], [2.77], [-2.16], [-1.1], [-2.12], [-0.62], [1.59], [-2.65], [-2.8], [2.88], [-2.45], [2.94], [-1.35], [-0.73], [-0.45], [0.35], [1.74], [-0.67], [2.38], [-0.96], [1.99], [1.71], [-1.42], [-0.45], [-0.98], [-1.03], [-1.55]]
classification_answers =[[-2.78], [-1.0], [5.49], [5.65], [-4.79], [0.5], [-2.47], [-2.97], [9.64], [-4.26], [-5.98], [9.72], [8.72], [-9.72], [-3.89], [9.8], [-9.87], [-1.1], [-9.29], [-1.81], [9.01], [8.42], [-9.86], [9.97], [5.33], [9.64], [-0.6], [-5.31], [-2.3], [-8.92], [-7.45], [-9.09], [-7.17], [-7.9], [-9.36], [9.96], [7.51], [6.22], [7.64], [-9.17], [0.2], [-7.65], [9.99], [-4.0], [-9.61], [9.84], [-8.96], [-8.72], [-8.62], [-9.97], [9.9], [-9.78], [4.0], [9.8], [10.0], [4.35], [-9.96], [-9.84], [-8.41], [8.96], [-7.03], [5.33], [9.61], [-10.0], [7.64], [-2.08], [-4.0], [7.17], [-4.27], [-8.96], [5.97], [6.83], [9.21], [3.52], [-0.0], [-1.2], [-8.31], [-8.03], [7.52], [-8.53], [-8.87], [2.49], [-5.24], [1.2], [8.83], [-5.31], [8.77], [7.58], [4.26], [9.94], [9.56], [8.58], [-0.1], [6.67], [9.9], [-6.22], [2.18], [-3.8], [9.99], [6.05], [-3.54], [9.95], [9.8], [9.84], [-9.73], [-5.73], [-9.42], [-9.64], [-6.37], [-2.18], [9.82], [-9.95], [-8.72], [5.06], [-9.85], [-2.67], [9.94], [-8.36], [9.98], [7.52], [-6.74], [9.85], [1.89], [-0.9], [-7.32], [-9.61], [1.89], [-9.99], [6.82], [-10.0], [9.89], [-8.63], [1.69], [-3.91], [-5.49], [-4.26], [-6.14], [-5.9], [-9.94], [5.66], [-0.8], [5.4], [-9.84], [-9.99], [9.85], [9.49], [9.87], [-6.45], [-6.22], [-9.05], [9.09], [-3.16], [9.43], [6.74], [8.62], [-0.5], [4.98], [-4.9], [8.41], [4.98], [-0.9], [4.98], [-6.05], [-8.96], [9.89], [1.89], [-9.46], [3.15], [-3.43], [2.49], [-9.69], [-9.76], [-7.9], [9.46], [-2.2], [-5.56], [9.89], [-9.21], [9.17], [5.41], [-4.27], [-3.35], [-3.8], [-5.24], [9.69], [9.97], [-9.84], [-4.97], [9.84], [9.69], [-9.69], [3.8], [-7.46], [-5.23], [7.72], [-6.3], [-5.4], [-8.19], [-6.37], [-6.89], [1.79], [4.26], [0.8], [-5.41], [-2.78], [-9.17], [1.51], [3.91], [-8.41], [-2.78], [8.14], [-7.72], [-9.89], [9.66], [-9.82], [-9.28], [-9.46], [-4.79], [9.09], [-10.0], [9.25], [-9.49], [-1.4], [-8.78], [-2.87], [7.64], [-3.26], [9.33], [8.72], [3.15], [6.96], [3.33], [-6.89], [9.73], [9.52], [-0.0], [9.78], [-4.44], [7.04], [9.97], [-5.33], [6.53], [3.16], [-10.0], [9.76], [5.23], [-7.71], [-8.37], [2.96], [7.46], [-9.64], [6.37], [-5.74], [-9.61], [4.36], [8.41], [0.5], [-9.56], [-8.42], [8.78], [-10.0], [-5.24], [8.08], [2.49], [-9.9], [-4.98], [9.84], [0.8], [6.89], [1.61], [-9.98], [-8.03], [-1.1], [9.01], [-9.33], [5.24], [5.48], [7.71], [9.97], [-5.31], [-2.38], [3.16], [6.53], [8.91], [9.61], [-2.68], [-9.43], [2.39], [-9.56], [4.08], [-7.78], [8.68], [-5.41], [-8.96], [3.15], [3.16], [4.53], [-9.8], [-9.71], [9.73], [-6.68], [5.14], [1.1], [-9.94], [-6.05], [9.66], [5.23], [3.63], [-9.64], [-6.22], [6.06], [-2.49], [-9.91], [-4.9], [10.0], [-5.48], [-7.31], [-4.45], [-7.71], [-4.9], [-6.22], [9.04], [4.0], [3.89], [7.71], [-9.58], [-7.78], [-8.47], [3.15], [6.67], [5.73], [4.71], [-8.37], [-7.96], [9.21], [5.4], [-5.98], [9.71], [-3.8], [-9.71], [1.69], [-9.8], [-3.43], [5.14], [-5.9], [6.14], [-7.17], [6.75], [4.81], [9.89], [5.07], [-8.31], [-4.27], [4.81], [4.17], [8.96], [-2.1], [9.93], [7.25], [-1.69], [-6.82], [2.39], [-9.61], [-1.81], [-9.69], [-9.39], [8.14], [-5.41], [10.0], [-1.3], [-6.75], [9.92], [-9.94], [-9.32], [5.48], [9.39], [5.07], [6.53], [-8.57], [9.25], [1.61], [8.25], [8.2], [-8.19], [-9.55], [7.58], [-9.94], [5.4], [8.36], [-1.51], [-0.1], [-9.94], [9.94], [-8.77], [8.78], [-2.38], [7.9], [5.81], [7.11], [-4.81], [-9.36], [9.84], [-8.13], [-3.43], [-1.9], [-6.74], [-3.91], [9.87], [6.61], [-9.95], [-9.76], [4.35], [7.24], [-5.65], [8.13], [9.8], [-9.95], [-5.65], [7.18], [8.62], [4.0], [4.17], [-9.97], [8.53], [-5.81], [-7.65], [-9.33], [-4.45], [9.0], [-8.08], [6.22], [-4.63], [2.86], [7.9], [-6.97], [8.2], [-5.31], [-6.96], [3.82], [-8.25], [-5.49], [7.51], [-8.36], [-8.25], [5.74], [-3.52], [9.46], [8.91], [5.56], [-4.9], [7.9], [-5.24], [2.2], [-8.73], [-5.81], [-6.05], [4.71], [5.14], [8.72], [-9.89], [9.21], [6.59], [-7.59], [-9.28], [9.84], [9.67], [-9.78], [7.32], [-8.25], [-4.44], [-8.36], [3.8], [9.78], [-7.9], [3.63], [-8.31], [-8.91], [-8.53], [-5.81], [10.0], [-4.72], [-3.35], [2.59], [-6.38], [2.0], [-9.76], [-6.67], [-4.35], [3.43], [9.86], [-6.21], [6.9], [-8.19], [9.13], [9.9], [-9.89], [-4.35], [-8.3], [-8.57], [-10.0]]

testing_data =[[-0.33], [0.51], [-0.38], [-2.72], [0.68], [-0.39], [1.29], [1.83], [2.08], [1.71], [-2.22], [-0.74], [-0.04], [-0.54], [-2.62], [1.42], [-1.66], [-1.91], [-2.56], [-1.12], [-2.13], [-1.29], [0.79], [1.09], [2.23], [2.99], [1.65], [2.17], [-1.35], [-1.47], [-0.04], [-2.56], [-1.63], [-0.55], [1.79], [-2.96], [0.25], [0.98], [1.0], [-2.47], [-0.57], [2.48], [-2.72], [1.54], [2.67], [1.6], [-2.58], [0.48], [0.36], [0.38]]
testing_answers = [[-3.24], [4.88], [-3.71], [-4.09], [6.29], [-3.8], [9.61], [9.67], [8.73], [9.9], [-7.97], [-6.74], [-0.4], [-5.14], [-4.98], [9.89], [-9.96], [-9.43], [-5.49], [-9.0], [-8.48], [-9.61], [7.1], [8.87], [7.9], [1.51], [9.97], [8.26], [-9.76], [-9.95], [-0.4], [-5.49], [-9.98], [-5.23], [9.76], [-1.81], [2.47], [8.3], [8.41], [-6.22], [-5.4], [6.14], [-4.09], [10.0], [4.54], [10.0], [-5.33], [4.62], [3.52], [3.71]]

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
            batch_loss = 0
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
                    batch_loss += self.layers[-1].mean_loss
                    #now for back prop
                    self.layers[-1].back_prop(self.layers[-1].d_loss)
                    current_epoch_loss += self.layers[-1].mean_loss
                    for l in range(len(self.layers)-1):
                        self.layers[-l-2].back_prop(self.layers[-l-1].loss_to_pass)
                    current_batch += 1
                    if current_batch == batch_size:
                        current_batch = 0
                        print(f"{round(batch_loss/batch_size,3)}")
                        for i in range(len(self.layers)):
                            self.layers[i].update_w_and_b(batch_size, learning_rate)
                        batch_loss = 0
                elif is_testing == True:
                    #print("----- TESTING -----")
                    if self.last_layer_activation != "Softmax":
                        self.layers[-1].loss(self.layers[-1].outputs, data_output[j],"mse")
                        current_epoch_loss += self.layers[-1].mean_loss
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


neural = NN(1, 2, 100, 1, "ReLU", "ReLU")
neural.train(100, 0.01, classification_data, classification_answers, False, 10)
neural.test(testing_data, testing_answers)