import numpy as np

class Optimizer:

    def __init__(self, learning_rate = 0.01,  momentum = 0.90):
        self.retained_gradient = {}
        self.learning_rate = learning_rate
        self.momentum = momentum
 
    def update(self, w, grad_loss_w):
        w_id = id(w)

        if w_id not in self.retained_gradient:
            self.retained_gradient[w_id] = np.zeros_like(w)

        self.retained_gradient[w_id] = self.momentum * self.retained_gradient[w_id] + (1 - self.momentum) * grad_loss_w
        return np.copy(w - self.learning_rate * self.retained_gradient[w_id])