from abc import ABC, abstractmethod
import numpy as np
import copy

class Layer(ABC):
    def __init__(self):
        self.seed = None

    def set_seed(self, seed):
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

    @abstractmethod
    def forward_propagation(self, inputs, training=True):
        raise NotImplementedError
    
    @abstractmethod
    def backward_propagation(self, error):
        raise NotImplementedError
    
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError
    
    @abstractmethod
    def parameters(self):
        raise NotImplementedError
    
    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape
    
    def layer_name(self):
        return self.__class__.__name__


class DenseLayer(Layer):
    def __init__(self, n_units, input_shape=None, l2=0.0):
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape
        self.l2 = l2  # Regularização L2
        
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None
        
    def initialize(self, optimizer):
        # Set seed before random initialization
        if hasattr(self, '_seed') and self._seed is not None:
            np.random.seed(self._seed)
        elif hasattr(optimizer, 'seed') and optimizer.seed is not None:
            np.random.seed(optimizer.seed)
        n_inputs = self.input_shape()[0] if self.input_shape() else 1
        self.weights = np.random.randn(n_inputs, self.n_units) * np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        return self
    
    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)
    
    def forward_propagation(self, inputs, training=True):
        inputs = np.array(inputs, dtype=np.float32)
        self.input = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
 
    def backward_propagation(self, output_error):
        input_error = np.dot(output_error, self.weights.T)
    
        weights_error = np.dot(self.input.T, output_error) + self.l2 * self.weights
        bias_error = np.sum(output_error, axis=0, keepdims=True)
    
        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        return input_error
 
    def output_shape(self):
        return (self.n_units,)
    

class EmbeddingLayer(Layer):
    def __init__(self, vocab_size, embedding_dim, input_shape=None):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self._input_shape = input_shape
        self.embeddings = None  
        self.emb_opt = None     
        self.input = None
        self.output = None

    def initialize(self, optimizer):
        # Set seed before random initialization
        if hasattr(self, '_seed') and self._seed is not None:
            np.random.seed(self._seed)
        elif hasattr(optimizer, 'seed') and optimizer.seed is not None:
            np.random.seed(optimizer.seed)
        self.embeddings = np.random.randn(self.vocab_size, self.embedding_dim) * 0.01
        self.emb_opt = copy.deepcopy(optimizer)
        return self

    def forward_propagation(self, inputs, training=True):
        if np.any(inputs >= self.vocab_size) or np.any(inputs < 0):
            raise ValueError("Índices de entrada fora do intervalo do vocabulário.")
        self.input = inputs
        self.output = self.embeddings[inputs]
        return self.output

    def backward_propagation(self, output_error):
        grad = np.zeros_like(self.embeddings)
        indices = self.input.flatten()
        error_flat = output_error.reshape(-1, self.embedding_dim)
        np.add.at(grad, indices, error_flat)
        self.embeddings = self.emb_opt.update(self.embeddings, grad)
        return None

    def output_shape(self):
        if self._input_shape is None:
            return (self.embedding_dim,)
        return (*self._input_shape, self.embedding_dim)

    def parameters(self):
        return self.vocab_size * self.embedding_dim
    

class FlattenLayer(Layer):
    def __init__(self):
        super().__init__()
        self._input_shape = None

    def forward_propagation(self, inputs, training=True):
        self.input = inputs
        batch_size = inputs.shape[0]
        self.output = inputs.reshape(batch_size, -1)
        return self.output

    def backward_propagation(self, output_error):
        return output_error.reshape(self.input.shape)

    def output_shape(self):
        if self._input_shape is None:
            raise ValueError("Input shape não definida para FlattenLayer.")
        return (np.prod(self._input_shape),)

    def parameters(self):
        return 0


class DropoutLayer(Layer):
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward_propagation(self, inputs, training=True):
        self.input = inputs
        if training:
            if hasattr(self, '_seed') and self._seed is not None:
                np.random.seed(self._seed)
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=inputs.shape) / (1 - self.dropout_rate)
            return inputs * self.mask
        return inputs

    def backward_propagation(self, output_error):
        return output_error * self.mask if self.mask is not None else output_error

    def output_shape(self):
        return self._input_shape

    def parameters(self):
        return 0

class BatchNormalizationLayer(Layer):
    def __init__(self, momentum=0.9, epsilon=1e-5):
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None
        self.input = None 
        self.output = None
        
    def initialize(self, optimizer):
        self._input_shape = self.input_shape()  # Obtém a forma da camada anterior
        self.gamma = np.ones(self._input_shape)
        self.beta = np.zeros(self._input_shape)
        self.running_mean = np.zeros(self._input_shape)
        self.running_var = np.ones(self._input_shape)
        self.gamma_opt = copy.deepcopy(optimizer)
        self.beta_opt = copy.deepcopy(optimizer)
        return self
    
    def forward_propagation(self, inputs, training=True):
        self.input = inputs
        if training:
            batch_mean = np.mean(inputs, axis=0)
            batch_var = np.var(inputs, axis=0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            self.norm = (inputs - batch_mean) / np.sqrt(batch_var + self.epsilon)
        else:
            self.norm = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        self.output = self.gamma * self.norm + self.beta
        return self.output
    
    def backward_propagation(self, output_error):
        batch_size = self.input.shape[0]
        batch_mean = np.mean(self.input, axis=0)
        batch_var = np.var(self.input, axis=0)
        
        dgamma = np.sum(output_error * self.norm, axis=0)
        dbeta = np.sum(output_error, axis=0)

        self.gamma = self.gamma_opt.update(self.gamma, dgamma)
        self.beta = self.beta_opt.update(self.beta, dbeta)

        dnorm = output_error * self.gamma
        dvar = np.sum(dnorm * (self.input - batch_mean) * -0.5 * (batch_var + self.epsilon) ** -1.5, axis=0)
        dmean = np.sum(dnorm * -1 / np.sqrt(batch_var + self.epsilon), axis=0) + dvar * np.mean(-2 * (self.input - batch_mean), axis=0)

        input_error = (dnorm / np.sqrt(batch_var + self.epsilon)) + (dvar * 2 * (self.input - batch_mean) / batch_size) + (dmean / batch_size)
        return input_error
    
    def output_shape(self):
        return self._input_shape
    
    def parameters(self):
        return np.prod(self.gamma.shape) + np.prod(self.beta.shape)


class GlobalAveragePoolingLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward_propagation(self, inputs, training=True):
        self.input = inputs
        # Faz a média ao longo do axis 1 (normalmente a dimensão da sequência)
        self.output = np.mean(inputs, axis=1)
        return self.output

    def backward_propagation(self, output_error):
        # O gradiente de cada elemento é dividido igualmente pelo tamanho da sequência
        seq_len = self.input.shape[1]
        # Expande a dimensão para repartir o erro de forma uniforme
        return np.repeat(output_error[:, np.newaxis, :], seq_len, axis=1) / seq_len

    def output_shape(self):
        if self._input_shape is None:
            raise ValueError("Input shape não definida para GlobalAveragePoolingLayer.")
        if len(self._input_shape) < 2:
            raise ValueError("GlobalAveragePoolingLayer requer uma entrada com pelo menos 2 dimensões.")
        # Se a input_shape for (seq_len, features), a saída será (features,)
        return self._input_shape[1:]

    def parameters(self):
        return 0
    
class GlobalAveragePooling1D(Layer):
    def __init__(self):
        super().__init__()

    def forward_propagation(self, inputs, training=True):
        """
        Calcula a média ao longo da dimensão temporal (axis=1).
        inputs: tensor de forma (batch_size, seq_len, features)
        Retorna: tensor de forma (batch_size, features)
        """
        self.input = inputs
        self.output = np.mean(inputs, axis=1)  # Média sobre seq_len
        return self.output

    def backward_propagation(self, output_error):
        """
        Distribui o erro uniformemente pela sequência.
        output_error: gradiente da camada seguinte, de forma (batch_size, features)
        Retorna: gradiente com a mesma forma que os inputs (batch_size, seq_len, features)
        """
        seq_len = self.input.shape[1]
        return np.repeat(output_error[:, np.newaxis, :], seq_len, axis=1) / seq_len

    def output_shape(self):
        if self._input_shape is None:
            raise ValueError("Input shape não definida para GlobalAveragePooling1D.")
        # Se a _input_shape foi definida sem a dimensão do batch, ela deve ter 2 dimensões: (seq_len, features)
        if len(self._input_shape) == 2:
            return (self._input_shape[1],)
        # Se a _input_shape incluir a dimensão do batch (3 dimensões), ignora o batch
        elif len(self._input_shape) == 3:
            return (self._input_shape[2],)
        else:
            raise ValueError("GlobalAveragePooling1D requer entrada de 2 ou 3 dimensões (seq_len, features) ou (batch, seq_len, features).")

    def parameters(self):
        return 0  # Camada sem parâmetros treináveis
