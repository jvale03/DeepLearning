import numpy as np
from layers import DenseLayer, EmbeddingLayer, FlattenLayer, DropoutLayer, BatchNormalizationLayer
from losses import BinaryCrossEntropy
from optimizer import Optimizer
from metrics import accuracy
from data import read_parquet
from activation import SigmoidActivation, ReLUActivation
import pickle
from visualization import plot_history

class DeepNeuralNetwork:
    def __init__(self, epochs=100, batch_size=128, optimizer=None,
                 learning_rate=0.01, momentum=0.90, verbose=True, 
                 loss=BinaryCrossEntropy, metric: callable = accuracy):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = Optimizer(learning_rate=learning_rate, momentum=momentum)
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric

        self.layers = []
        self.history = {}

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)
        self.layers.append(layer)
        return self

    def get_mini_batches(self, X, y=None, shuffle=True):
        n_samples = X.shape[0]
        assert self.batch_size <= n_samples, "Batch size cannot be greater than the number of samples"
        
        indices = np.random.permutation(n_samples) if shuffle else np.arange(n_samples)
        
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            if y is not None:
                yield X[indices[start:end]], y[indices[start:end]]
            else:
                yield X[indices[start:end]], None

    def forward_propagation(self, X, training):
        for layer in self.layers:
            X = layer.forward_propagation(X, training)
        return X

    def backward_propagation(self, output_error):
        for layer in reversed(self.layers):
            output_error = layer.backward_propagation(output_error)
        return output_error

    def fit(self, dataset, validation_data=None, patience=3):
        X, y = dataset.X, dataset.y
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        self.history = {'train_loss': [], 'train_metric': [], 'val_loss': [], 'val_metric': []}

        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None

        output_x_all = np.zeros((X.shape[0], 1))
        y_all = np.zeros((X.shape[0], 1))

        for epoch in range(1, self.epochs + 1):
            idx = 0
            for X_batch, y_batch in self.get_mini_batches(X, y):
                output = self.forward_propagation(X_batch, training=True)
                error = self.loss.derivative(y_batch, output)
                self.backward_propagation(error)
                
                batch_size = len(y_batch)
                output_x_all[idx:idx + batch_size] = output
                y_all[idx:idx + batch_size] = y_batch
                idx += batch_size

            train_loss = self.loss.loss(y_all, output_x_all)
            train_metric = self.metric(y_all, output_x_all)

            self.history['train_loss'].append(train_loss)
            self.history['train_metric'].append(train_metric)

            val_loss, val_metric = None, None
            if validation_data:
                val_output = self.predict(validation_data)
                val_loss = self.loss.loss(validation_data.y, val_output)
                val_metric = self.metric(validation_data.y, val_output)

                self.history['val_loss'].append(val_loss)
                self.history['val_metric'].append(val_metric)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = [layer.get_weights() for layer in self.layers if hasattr(layer, 'get_weights')]
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"\nEarly stopping ativado na epoch {epoch}. Melhor val_loss: {best_val_loss:.4f}")
                    for layer, weights in zip(self.layers, best_weights):
                        if hasattr(layer, 'set_weights'):
                            layer.set_weights(weights)
                    break

            if self.verbose:
                print(f"Epoch {epoch}/{self.epochs} - loss: {train_loss:.4f} - accuracy: {train_metric:.4f} "
                      f"- val_loss: {val_loss:.4f} - val_accuracy: {val_metric:.4f}")
        return self

    def predict(self, dataset):
        return self.forward_propagation(dataset.X, training=False)

    def score(self, dataset, predictions):
        if self.metric is not None:
            return self.metric(dataset.y, predictions)
        raise ValueError("No metric specified for the neural network.")

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Modelo salvo em {file_path}")

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Modelo carregado de {file_path}")
        return model

if __name__ == '__main__':
    train_data = read_parquet('../../datasets/train/train_sample_processed.parquet')
    test_data = read_parquet('../../datasets/test/test_sample_processed.parquet')

    net = DeepNeuralNetwork(epochs=20, batch_size=32, learning_rate=0.005, verbose=True,
                            loss=BinaryCrossEntropy, metric=accuracy)

    n_features = train_data.X.shape[1]
    net.add(EmbeddingLayer(vocab_size=10000, embedding_dim=16, input_shape=(n_features,)))
    net.add(DropoutLayer(dropout_rate=0.3))
    net.add(FlattenLayer())

    net.add(DenseLayer(32, l2=0.01))  
    net.add(BatchNormalizationLayer())  
    net.add(ReLUActivation())  
    net.add(DropoutLayer(dropout_rate=0.4))  

    net.add(DenseLayer(16, l2=0.02))  
    net.add(BatchNormalizationLayer())  
    net.add(ReLUActivation())  
    net.add(DropoutLayer(dropout_rate=0.6))  

    net.add(DenseLayer(1, l2=0.02))  
    net.add(SigmoidActivation())

    net.fit(train_data, validation_data=test_data, patience=6)
    plot_history(net.history)

    while True:
        opt = input("Queres guardar? [y/n]")
        if opt == "y":
            net.save("../../models/modelo_treinado.pkl")
            break
        elif opt == "n":
            break
