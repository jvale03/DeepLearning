import numpy as np

class Optimizer:

    def __init__(self, learning_rate = 0.01,  momentum = 0.90, seed=None):
        self.retained_gradient = {}
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)
 
    def update(self, w, grad_loss_w):
        w_id = id(w)

        if w_id not in self.retained_gradient:
            self.retained_gradient[w_id] = np.zeros_like(w)

        self.retained_gradient[w_id] = self.momentum * self.retained_gradient[w_id] + (1 - self.momentum) * grad_loss_w
        return np.copy(w - self.learning_rate * self.retained_gradient[w_id])
    

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, seed=None):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # Dicionários para armazenar os momentos de primeira e segunda ordem e a contagem de passos para cada parâmetro
        self.m = {}
        self.v = {}
        self.t = {}
        if seed is not None:
            np.random.seed(seed)

    def update(self, w, grad_loss_w):
        w_id = id(w)
        # Inicializa os momentos e o contador se ainda não existirem para o parâmetro w
        if w_id not in self.m:
            self.m[w_id] = np.zeros_like(w)
            self.v[w_id] = np.zeros_like(w)
            self.t[w_id] = 0

        # Incrementa o contador de passos
        self.t[w_id] += 1

        # Atualiza os momentos de primeira (m) e segunda ordem (v)
        self.m[w_id] = self.beta1 * self.m[w_id] + (1 - self.beta1) * grad_loss_w
        self.v[w_id] = self.beta2 * self.v[w_id] + (1 - self.beta2) * (grad_loss_w ** 2)

        # Correção de viés para os momentos
        m_hat = self.m[w_id] / (1 - self.beta1 ** self.t[w_id])
        v_hat = self.v[w_id] / (1 - self.beta2 ** self.t[w_id])

        # Atualiza os pesos utilizando a regra do Adam
        w_updated = w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return np.copy(w_updated)