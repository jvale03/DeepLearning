from abc import ABC, abstractmethod
import numpy as np
import copy

'''
layers.py
Defines the layers that can be used in the neural network
'''


class Layer(ABC):

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
        
        dgamma = np.sum(output_error * self.norm, axis=0)
        dbeta = np.sum(output_error, axis=0)
        
        self.gamma = self.gamma_opt.update(self.gamma, dgamma)
        self.beta = self.beta_opt.update(self.beta, dbeta)
        
        dnorm = output_error * self.gamma
        dvar = np.sum(dnorm * (self.input - np.mean(self.input, axis=0)) * -0.5 * (self.running_var + self.epsilon) ** -1.5, axis=0)
        dmean = np.sum(dnorm * -1 / np.sqrt(self.running_var + self.epsilon), axis=0) + dvar * np.mean(-2 * (self.input - np.mean(self.input, axis=0)), axis=0)
        
        input_error = dnorm / np.sqrt(self.running_var + self.epsilon) + dvar * 2 * (self.input - np.mean(self.input, axis=0)) / batch_size + dmean / batch_size
        return input_error
    
    def output_shape(self):
        return self._input_shape
    
    def parameters(self):
        return np.prod(self.gamma.shape) + np.prod(self.beta.shape)


class RNNLayer(Layer):
    def __init__(self, n_units, input_shape=None, return_sequences=False, bptt_trunc=None):
        '''
        n_units: Number of units in the RNN layer, represents the hidden state size i.e. how much information to remember
        input_shape: Shape of the input data
        return_sequences: Whether to return the output for each timestep or just the last one
        bptt_trunc: The number of time steps to backpropagate through time (i.e. the number of time steps to unroll the RNN).
        '''

        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape
        self.return_sequences = return_sequences
        self.bptt_trunc = bptt_trunc

        # Weights
        self.W = None # Input data 
        self.U = None # Recurrent data
        self.b = None # Bias

        # For backpropagation
        self.inputs = None # Input sequence
        self.states = None # Hidden states
        self.outputs = None # Output sequence

    def initialize(self, optimizer):
        # Getting the input dimension
        n_inputs = self.input_shape()[-1] if self.input_shape() else 1

        '''
        2 initialization methods under -> choose 1 only
        '''
        '''
        # Initializing weights randomly
        self.W = np.random.randn(self.n_units, n_inputs)
        self.U = np.random.randn(self.n_units, self.n_units)
        self.b = np.random.randn(self.n_units)
        '''

        # Initializing weights 'randomly' but with Xavier/Glorot initialization (helps preventing vanishing/exploding gradients)
        self.W = np.random.randn(self.n_units, n_inputs) * np.sqrt(2.0 / (n_inputs + self.n_units))
        self.U = np.random.randn(self.n_units, self.n_units) * np.sqrt(2.0 / (self.n_units + self.n_units))
        self.b = np.zeros((self.n_units,))

        # Creating optimizers for each parameter
        self.W_opt = copy.deepcopy(optimizer)
        self.U_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)

        return self
    
    def forward_propagation(self, inputs, training=True):
        
        # Handling 4D input from EmbeddingLayer (batch_size, seq_len, feature_dim, embedding_dim)
        if len(inputs.shape) == 4:
            batch_size, sequence_length, feature_dim, embedding_dim = inputs.shape
            # Reshaping to 3D by combining feature_dim and embedding_dim
            inputs = inputs.reshape(batch_size, sequence_length, feature_dim * embedding_dim)
        
        batch_size, sequence_length, n_features = inputs.shape
        self.inputs = inputs
        self.states = [np.zeros((batch_size, self.n_units))] # Initial hidden state
        self.outputs = []

        for t in range(sequence_length):
                x_t = inputs[:, t, :] 
                h_prev = self.states[-1]
                
                h_t = np.tanh(
                    np.dot(x_t, self.W.T) +
                    np.dot(h_prev, self.U.T) +
                    self.b
                )
                self.states.append(h_t)
                self.outputs.append(h_t)

        if self.return_sequences:
            # Returning the output for each timestep
            return np.stack(self.outputs, axis=1)
        else:
            # Returning only the last output
            return self.outputs[-1]
        
    def backward_propagation(self, output_error):
        '''
        output_error: Error from the next layer with shape (batch_size, sequence_length, n_units) if return_sequences=True
            or (batch_size, n_units) if return_sequences=False
        '''
        batch_size, sequence_length, n_features = self.inputs.shape

        # Determine how many timesteps to go backwards
        # should be the minimum between bptt_trunc and sequence_length
        if self.bptt_trunc is not None:
            backpropg_steps = min(sequence_length, self.bptt_trunc)
            start_t = max(0, sequence_length - backpropg_steps)
        else:
            backpropg_steps = sequence_length
            start_t = 0

        # Initializing gradients
        dW = np.zeros_like(self.W)
        dU = np.zeros_like(self.U)
        db = np.zeros_like(self.b)

        # Initializing input errors
        dx = np.zeros_like(self.inputs)

        # If return_sequences=False, the error is only from the last time step (t)
        dh_next = np.zeros((batch_size, self.n_units))
        if not self.return_sequences:
            dh_next = output_error

        # If return_sequences=True, the error is from the last time step (t) to the first (0)
        for t in reversed(range(start_t, sequence_length)):
            if self.return_sequences:
                # current timestep gradient plus accumulated future gradient
                dh = output_error[:, t, :] + dh_next
            else:
                # using only accumulated future gradient
                dh = dh_next
            
            # Getting current and previous state
            h_t = self.states[t+1]
            h_prev = self.states[t]

            # Getting current input
            x_t = self.inputs[:, t, :]

            # Calculating the gradient for each batch
            for i in range(batch_size):
                dtanh = (1 - h_t[i]**2) * dh[i]

                dW += np.outer(dtanh, x_t[i])
                dU += np.outer(dtanh, h_prev[i])
                db += dtanh

                # Calculating input error for this timestep
                dx[i, t, :] = np.dot(self.W.T, dtanh)

                # Calculate the error to pass to previous timestep
                dh_next[i] = np.dot(self.U.T, dtanh)

        # Updating the weights
        self.W = self.W_opt.update(self.W, dW)
        self.U = self.U_opt.update(self.U, dU)
        self.b = self.b_opt.update(self.b, db)

        return dx

    # Returns the shape of the output
    def output_shape(self):
        if self.return_sequences:
            return (self._input_shape[0], self.n_units)
        else:
            return (self.n_units,)
        
    # Calculates the number of parameters
    def parameters(self):
        return(np.prod(self.W.shape) +
               np.prod(self.U.shape) + 
               np.prod(self.b.shape))
    
    # Returns the weights in a dictionary
    def get_weights(self):
        return {
            'W' : self.W.copy(),
            'U' : self.U.copy(),
            'b' : self.b.copy()
        }
    
    # Updates the weights with the ones in the dictionary
    def set_weights(self, weights):
        self.W = weights['W'].copy()
        self.U = weights['U'].copy()
        self.b = weights['b'].copy()