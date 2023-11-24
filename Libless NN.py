import math
import random
E = math.e
random.seed(0)

#--------------------------------------------------------------------------------------------------------------------------------
class Layer:
    def __init__(self, previous_height, height, activation_function_type):
        self.biases = [(1*(random.random())*2-1) for n in range(height)]
        self.weights = [[(1*(random.random())*2-1) for n in range(previous_height)] for m in range(height)]
       
        self.delta_biases = [0] * height
        self.delta_weights = [[0] * previous_height for _ in range(height)]

        self.activation_function_type = activation_function_type
    
    def forward(self, previous_layer_outputs):
        self.previous_layer_outputs = previous_layer_outputs
        self.outputs = [0]*len(self.biases)
        for j in range(len(self.biases)):
            for k in range(len(self.weights[0])):
                self.outputs[j] += ((previous_layer_outputs[k]) * (self.weights[j][k]))
            self.outputs[j] += self.biases[j]

    def activation_function(self):
        if self.activation_function_type == "None":
            self.post_activation_outputs = self.outputs
        if self.activation_function_type == "ReLU":
            self.post_activation_outputs = [max(0, n) for n in self.outputs]
        if self.activation_function_type == "Leaky_ReLU":
            self.post_activation_outputs = [0.1*n if n<0 else n for n in self.outputs]
        if self.activation_function_type == "Softmax":
            self.post_activation_outputs = self.outputs
            maximum_output = max(self.post_activation_outputs)
            sum = 0
            for i in range(len(self.post_activation_outputs)):
                self.post_activation_outputs[i] -= maximum_output
                self.post_activation_outputs[i] = E**self.post_activation_outputs[i]
                sum += self.post_activation_outputs[i]
            for i in range(len(self.post_activation_outputs)):
                self.post_activation_outputs[i] /= sum


    def loss(self, prediced_list, expected_list, type):
        self.loss_type = type
        if self.loss_type == "mse":
            self.mean_loss = 0
            for i in range(len(prediced_list)):
                self.mean_loss += 0.5*((prediced_list[i] - expected_list[i])**2)
            self.mean_loss /= len(prediced_list)
            self.d_loss = [(prediced_list[i] - expected_list[i]) for i in range(len(prediced_list))]
        elif self.loss_type == "log":
            self.mean_loss = 0
            for i in range(len(prediced_list)):
                self.mean_loss += 0.5*((prediced_list[i] - expected_list[i])**2)
            self.mean_loss /= len(prediced_list)
            self.d_loss = [(prediced_list[i] - expected_list[i]) for i in range(len(prediced_list))]
        
    def back_prop(self, inputted_loss_array):
        self.passed_on_loss_array = inputted_loss_array
        if self.activation_function_type == "ReLU":
            for i in range(len(inputted_loss_array)):
                if self.outputs[i] < 0:
                    self.passed_on_loss_array[i] *= 0
                else:
                    self.passed_on_loss_array[i] *= 1
        elif self.activation_function_type == "Leaky_ReLU":
            for i in range(len(inputted_loss_array)):
                if self.outputs[i] < 0:
                    self.passed_on_loss_array[i] *= 0.1
                else:
                    self.passed_on_loss_array[i] *= 1
        elif self.activation_function_type == "Softmax":
            for i in range(len(self.passed_on_loss_array)):
                self.passed_on_loss_array[i] *= (1 - self.outputs[i]) * self.outputs[i]
                #self.passed_on_loss_array[i] *= E**self.passed_on_loss_array[i]

        for i in range(len(self.biases)):
            self.delta_biases[i] += self.passed_on_loss_array[i]

        for i in range(len(self.weights)):
            for j in range(len(self.weights[0])):
                self.delta_weights[i][j] += self.passed_on_loss_array[i]*self.previous_layer_outputs[j]

        self.loss_to_pass = [0] * len(self.weights[0])
        for i in range(len(self.loss_to_pass)):
            for j in range(len(self.weights)):
                self.loss_to_pass[i] += self.passed_on_loss_array[j] * self.weights[j][i]

    def update_w_and_b(self, batch_size, learning_rate):
        for i in range(len(self.delta_weights)):
            for j in range(len(self.delta_weights[0])):
                self.delta_weights[i][j] /= batch_size
                self.weights[i][j] -= self.delta_weights[i][j]*learning_rate
        for i in range(len(self.biases)):
            self.delta_biases[i] /= batch_size
            self.biases[i] -= self.delta_biases[i]*learning_rate

        self.delta_biases = [0] * len(self.biases)
        self.delta_weights = [[0] * len(self.weights[0]) for _ in range(len(self.weights))]

#--------------------------------------------------------------------------------------------------------------------------------
class NN:
    def __init__(self, input_size, inner_layers_number, height, output_size, inner_layer_activation, last_layer_activation):
        self.inner_layer_activation = inner_layer_activation
        self.last_layer_activation = last_layer_activation
        self.layers = [[]] * (inner_layers_number + 2)
        self.layers[0] = Layer(input_size, height, inner_layer_activation)
        self.layers[-1] = Layer(height, output_size, last_layer_activation)
        for i in range(inner_layers_number):
            self.layers[i+1] = Layer(height, height, inner_layer_activation)
    
    def train(self, epochs, learning_rate, training_data, training_answers, batch_size):
        current_epoch = 0
        current_batch = 0
        batch_loss = 0
        for i in range(epochs):
            current_epoch_loss = 0
            combined_data = list(zip(training_data, training_answers))
            # Shuffle the combined data
            random.shuffle(combined_data)
            # Split the shuffled data back into training_data and training_answers
            training_data, training_answers = zip(*combined_data)

            for j in range(len(training_data)):
                current_batch += 1
                #start layer forward and activ
                self.layers[0].forward(training_data[j])
                self.layers[0].activation_function()
                #middle layer forward and activ
                for k in range(len(self.layers)-2):
                    self.layers[k+1].forward(self.layers[k].post_activation_outputs)
                    self.layers[k+1].activation_function()
                #last layer forward and activ
                self.layers[-1].forward(self.layers[-2].post_activation_outputs)
                self.layers[-1].activation_function()
                #now for loss
                loss_function = "log" if self.last_layer_activation == "Softmax" else "mse"
                self.layers[-1].loss(self.layers[-1].post_activation_outputs, training_answers[j],loss_function)
                batch_loss += self.layers[-1].mean_loss
                current_epoch_loss += self.layers[-1].mean_loss
                #now for back prop
                self.layers[-1].back_prop(self.layers[-1].d_loss)
                for l in range(len(self.layers)-1):
                    self.layers[-l-2].back_prop(self.layers[-l-1].loss_to_pass)
                
                if current_batch == batch_size:
                    current_batch = 0
                    print(f"{round(batch_loss/batch_size,3)}")
                    for i in range(len(self.layers)):
                        self.layers[i].update_w_and_b(batch_size, learning_rate)
                    batch_loss = 0
            
            if current_batch != 0:
                for i in range(len(self.layers)):
                    self.layers[i].update_w_and_b(batch_size, learning_rate)
            
            current_epoch += 1
            print(f"Epochs completed: {current_epoch}/{epochs}\nAverage epoch loss: {current_epoch_loss/len(training_data)}")

    def predict(self, data_to_predict):
        self.prediction_outputs = []
        for i in range(len(data_to_predict)):
            self.layers[0].forward(data_to_predict[i])
            self.layers[0].activation_function()
            #middle layer forward and activ
            for k in range(len(self.layers)-2):
                self.layers[k+1].forward(self.layers[k].post_activation_outputs)
                self.layers[k+1].activation_function()
            #last layer forward and activ
            self.layers[-1].forward(self.layers[-2].post_activation_outputs)
            self.layers[-1].activation_function()
            print(f"Predicting")
            self.prediction_outputs.append(self.layers[-1].post_activation_outputs)

#--------------------------------------------------------------------------------------------------------------------------------
class Training_data:
    def __init__(self, amount_to_gen):
        self.training_inputs = []
        self.training_outputs = []
        for i in range(amount_to_gen):
            x1 = random.random()*20-5
            x2 = random.random()*20-5
            self.training_inputs.append([x1, x2])
            if x1**2 + x2**2 <= 25:
                self.training_outputs.append([0,1])
            else:
                self.training_outputs.append([1,0])
    def get_training_inputs(self):
        return(self.training_inputs)
    def get_training_outputs(self):
        return(self.training_outputs)
        
#--------------------------------------------------------------------------------------------------------------------------------
class Prediction_data:
    def __init__(self, amount_to_gen):
        self.prediction_inputs = []
        self.prediction_outputs = []
        for i in range(amount_to_gen):
            x1 = random.random()*10-5
            x2 = random.random()*10-5
            self.prediction_inputs.append([x1, x2])
            if x1**2 + x2**2 <= 25:
                self.prediction_outputs.append([0,1])
            else:
                self.prediction_outputs.append([1,0])
    def get_prediction_inputs(self):
        return(self.prediction_inputs)
    def get_prediction_outputs(self):
        return(self.prediction_outputs)


#--------------------------------------------------------------------------------------------------------------------------------
training_data = Training_data(1000) #amount to gen
prediction_data = Prediction_data(50) #amount to gen
#--------------------------------------------------------------------------------------------------------------------------------

neural = NN(2, 3, 30, 2, "Leaky_ReLU", "Softmax")
neural.train(300, 0.001, training_data.get_training_inputs(), training_data.get_training_outputs(), 50)
neural.predict(prediction_data.get_prediction_inputs())

#print(neural.prediction_outputs)

#use compare() for classification tasks
def compare(prediction, actual):
    total_correct = 0
    for i in range(len(prediction)):
        for j in range(len(prediction[0])):
            prediction[i][j] = round(prediction[i][j])
    for i in range(len(prediction)):
        if prediction[i] == actual[i]:
            total_correct += 1
    print(f"Total accuracy: {round((total_correct/len(prediction))*100,5)} %")

compare(neural.prediction_outputs,prediction_data.get_prediction_outputs())