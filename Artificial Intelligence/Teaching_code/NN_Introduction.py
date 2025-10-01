#practical introduction to artifitial neural networks
import numpy as np

# A perceptron is one of the earliest and simplest types of artificial neural networks, introduced by Frank Rosenblatt in 1958. 
#It’s a binary linear classifier that decides whether an input belongs to one class or another.
   
inputs, weights = [], []

preguntas = [
    "Glucose level: ",
    "BMI: ",
    "Age: "
]

for pregunta in preguntas:
    i = float(input(pregunta))
    w = float(input("\nWeight: "))
    inputs.append(i)
    weights.append(w)
threshold = int(input("\nthreshold: "))

print("The perceptron as a weighted sum with threshold\n")

class WeightedSum():
    def __init__(self, inputs, weights):
        self.inputs = np.array(inputs)
        self.weights = np.array(weights)
  
    def predict(self, threshold):
        return (self.inputs @ self.weights) >= threshold

linear_clissifier = WeightedSum(inputs, weights)
print(linear_clissifier.predict(threshold))


print("The percectron - jump function \n")

class Perceptron():
    def __init__(self, inputs, weights):
        self.inputs = np.array(inputs)
        self.weights = np.array(weights)
  
    def predict(self, threshold):
    	b = -threshold
        return (self.inputs @ self.weights) + b > 0
        
linear_clissifier = Perceptron(inputs, weights)
print(linear_clissifier.predict(threshold))

print("Change the Activation function to a sigmode")

class SigmoidNeuron():
    def __init__(self, inputs, weights):
        self.inputs = np.array(inputs)
        self.weights = np.array(weights)
  
    def predict(self, bias):
        z = (self.weights @ self.inputs) + bias
        return 1. / (1. + np.exp(-z))

linear_clissifier = SigmoidNeuron(inputs, weights)
print(linear_clissifier.predict(threshold))



















