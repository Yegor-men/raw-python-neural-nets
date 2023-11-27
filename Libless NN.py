import math
import random
E = math.e
random.seed(0)
import csv

#--------------------------------------------------------------------------------------------------------------------------------
class Layer:
    def __init__(self, previous_height, height, activation_function_type):
        self.biases = [1 * (random.random() * 2 - 1) for n in range(height)]
        self.weights = [[(1 * (random.random()) * 2 - 1) for n in range(previous_height)] for m in range(height)]
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
        elif self.activation_function_type == "ReLU":
            self.post_activation_outputs = [max(0, n) for n in self.outputs]
        elif self.activation_function_type == "Leaky_ReLU":
            self.post_activation_outputs = [0.1 * n if n < 0 else n for n in self.outputs]
        elif self.activation_function_type == "Softmax":
            self.maximum_output = max(self.outputs)
            self.exp_outputs = [E**(n-self.maximum_output) for n in self.outputs]
            self.sum_exp_outputs = sum(self.exp_outputs)
            self.post_activation_outputs = [n / self.sum_exp_outputs for n in self.exp_outputs]

    def loss(self, prediced_list, expected_list, type):
        self.loss_type = type
        self.mean_loss = 0
        if self.loss_type == "mse":
            for i in range(len(prediced_list)):
                self.mean_loss += 0.5*((prediced_list[i] - expected_list[i])**2)
            self.mean_loss /= len(prediced_list)
            self.d_loss = [(prediced_list[i] - expected_list[i]) for i in range(len(prediced_list))]
        elif self.loss_type == "log":
            self.d_loss = self.post_activation_outputs[:]
            for i in range(len(self.post_activation_outputs)):
                self.d_loss[i] = -expected_list[i]*math.log(self.post_activation_outputs[i] + 1e-15)
            for i in range(len(self.d_loss)):
                self.mean_loss += self.d_loss[i]
                self.d_loss[i] = (self.post_activation_outputs[i] - expected_list[i])

    def back_prop(self, inputted_loss_array):
        self.passed_on_loss_array = inputted_loss_array
        if self.activation_function_type == "ReLU":
            for i in range(len(inputted_loss_array)):
                if self.outputs[i] < 0:
                    self.passed_on_loss_array[i] *= 0
        elif self.activation_function_type == "Leaky_ReLU":
            for i in range(len(inputted_loss_array)):
                if self.outputs[i] < 0:
                    self.passed_on_loss_array[i] *= 0.1

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
        for i in range(epochs):
            current_epoch_loss = 0
            batch_loss = 0
            combined_data = list(zip(training_data, training_answers))
            # Shuffle the combined data
            random.shuffle(combined_data)
            # Split the shuffled data back into training_data and training_answers
            training_data, training_answers = zip(*combined_data)
            for j in range(len(training_data)):
                current_batch += 1
                #start layer forward and activation
                self.layers[0].forward(training_data[j])
                self.layers[0].activation_function()
                #middle layer forward and activation
                for k in range(len(self.layers)-2):
                    self.layers[k+1].forward(self.layers[k].post_activation_outputs)
                    self.layers[k+1].activation_function()
                #last layer forward and activation
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
                    # print(f"{round(batch_loss/batch_size,3)}")
                    for g in range(len(self.layers)):
                        self.layers[g].update_w_and_b(batch_size, learning_rate)
                    batch_loss = 0
            if current_batch != 0:
                for k in range(len(self.layers)):
                    self.layers[k].update_w_and_b(batch_size, learning_rate)
            current_epoch += 1
            print(f"Epochs completed: {current_epoch}/{epochs} |Average epoch loss: {current_epoch_loss/len(training_data)}")

    def predict(self, data_to_predict):
        self.prediction_outputs = []
        for i in range(len(data_to_predict)):
            self.layers[0].forward(data_to_predict[i])
            self.layers[0].activation_function()
            #middle layer forward and activation
            for k in range(len(self.layers)-2): 
                self.layers[k+1].forward(self.layers[k].post_activation_outputs)
                self.layers[k+1].activation_function()
            #last layer forward and activation
            self.layers[-1].forward(self.layers[-2].post_activation_outputs)
            self.layers[-1].activation_function()
            print(f"Predicting")
            self.prediction_outputs.append(self.layers[-1].post_activation_outputs)

    def export_weights(self):
        all_weights = []
        for i in range(len(self.layers)):
            all_weights.append(self.layers[i].weights)
        return(all_weights)
    
    def export_biases(self):
        all_biases = []
        for i in range(len(self.layers)):
            all_biases.append(self.layers[i].biases)
        return(all_biases)

#--------------------------------------------------------------------------------------------------------------------------------
# class Training_data:
#     def __init__(self, amount_to_gen):
#         self.training_inputs = []
#         self.training_outputs = []
#         for i in range(amount_to_gen):
#             x1 = random.random()*20
#             self.training_inputs.append([x1])
#             self.training_outputs.append([1,0,0] if x1<5 else [0,1,0] if 5<=x1<13 else [0,0,1])
#     def get_t_q(self):
#         return(self.training_inputs)
#     def get_t_a(self):
#         return(self.training_outputs)
        
# #--------------------------------------------------------------------------------------------------------------------------------
# class Prediction_data:
#     def __init__(self, amount_to_gen):
#         self.prediction_inputs = []
#         self.prediction_outputs = []
#         for i in range(amount_to_gen):
#             x1 = random.random()*20
#             self.prediction_inputs.append([x1])
#             self.prediction_outputs.append([1,0,0] if x1<5 else [0,1,0] if 5<=x1<13 else [0,0,1])
#     def get_p_q(self):
#         return(self.prediction_inputs)
#     def get_p_a(self):
#         return(self.prediction_outputs)

# #--------------------------------------------------------------------------------------------------------------------------------
# training_data = Training_data(1000)
# prediction_data = Prediction_data(100)
#--------------------------------------------------------------------------------------------------------------------------------
def one_hot_encoding(data, data_types):
    output = []
    for i in range(len(data)):
        output.append([])
        for j in range(data_types):
            if j == data[i]:
                output[i].append(1)
            else:
                output[i].append(0)
    return(output)
#-----IRIS-DATA------------------------------------------------------------------------------------------------------------------
questions = []
labels = []

with open("Iris.csv", "r") as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        feature_row = list(map(float, row[1:5]))
        label = row[5]
        questions.append(feature_row)
        labels.append(label)

for i in range(len(questions)):
    if labels[i] == "Iris-setosa":
        labels[i] = 0
    elif labels[i] == "Iris-versicolor":
        labels[i] = 1
    elif labels[i] == "Iris-virginica":
        labels[i] = 2


class QuestionsAndAnswers():
    def __init__(self, questions, answers, amount):
        combined_data = list(zip(questions, answers))
        # Shuffle the combined data
        random.shuffle(combined_data)
        # Split the shuffled data back into training_data and training_answers
        questions, answers = zip(*combined_data)
        
        self.training_data_questions = questions[:amount]
        self.training_data_answers = answers[:amount]
        self.prediction_data_questions = questions[amount:]
        self.prediction_data_answers = answers[amount:]
    
    def get_t_q(self):
        return self.training_data_questions
    def get_t_a(self):
        return self.training_data_answers
    def get_p_q(self):
        return self.prediction_data_questions
    def get_p_a(self):
        return self.prediction_data_answers

iris_data = QuestionsAndAnswers(questions, one_hot_encoding(labels,3), 99)

#--------------------------------------------------------------------------------------------------------------------------------
def classification_compare(prediction, actual):
        total_correct = 0
        print(prediction)
        for i in range(len(prediction)):
            index_of_max = prediction[i].index(max(prediction[i]))
            for j in range(len(prediction[0])):
                if j == index_of_max:
                    prediction[i][j] = 1
                else:
                    prediction[i][j] = 0
        print(prediction)
        for i in range(len(prediction)):
            if prediction[i] == actual[i]:
                total_correct += 1
        print(f"Total accuracy: {round((total_correct/len(prediction))*100,5)} %")

def train_and_test(input_size, inner_layers_amount, neurons_per_layer, output_size, inner_neuron_activation, last_layer_activation, epochs, training_step, training_questions, training_answers, batch_size, predict_questions, predict_answers, is_classification):
    neural = NN(input_size, inner_layers_amount, neurons_per_layer, output_size, inner_neuron_activation, last_layer_activation)
    neural.train(epochs, training_step, training_questions, training_answers, batch_size)
    neural.predict(predict_questions)
    if is_classification == False:
        print(neural.prediction_outputs)
    if is_classification == True:
        classification_compare(neural.prediction_outputs, predict_answers)
    # print(neural.export_biases())
    # print(neural.export_weights())

train_and_test(input_size = 4, 
               inner_layers_amount = 10, 
               neurons_per_layer = 10, 
               output_size = 3, 
               inner_neuron_activation = "ReLU", 
               last_layer_activation = "Softmax", 
               epochs = 1000,
               training_step = 0.01,
               training_questions = iris_data.get_t_q(),
               training_answers = iris_data.get_t_a(),
               batch_size = 1000,
               predict_questions = iris_data.get_p_q(),
               predict_answers = iris_data.get_p_a(),
               is_classification = True)