from abc import ABC, abstractmethod
import numpy as np
from layers import Layer

class ActivationLayer(Layer, ABC):
    def forward_propagation(self, input, training):
        self.input = input
        self.output = self.activation_function(input)
        return self.output

    def backward_propagation(self, output_error, learning_rate=None):
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input):
        pass

    @abstractmethod
    def derivative(self, input):
        pass

    def output_shape(self):
        return self._input_shape

    def parameters(self):
        return 0

class SigmoidActivation(ActivationLayer):
    def activation_function(self, input):
        input = np.clip(input, -50, 50)  # Evita overflow
        self.output = 1 / (1 + np.exp(-input))  
        return self.output  

    def derivative(self, input):
        return self.output * (1 - self.output)  # Usa output já calculado para eficiência

class ReLUActivation(ActivationLayer):
    def __init__(self):
        pass  # Removido alpha desnecessário

    def activation_function(self, input):
        self.output = np.maximum(0, input)
        return self.output

    def derivative(self, input):
        return np.where(input > 0, 1, 0)  # Derivada correta (0 para input <= 0)

class LeakyReLUActivation(ActivationLayer):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def activation_function(self, input):
        self.output = np.where(input > 0, input, self.alpha * input)
        return self.output

    def derivative(self, input):
        return np.where(input > 0, 1, self.alpha)