import numpy as np
from layers import DenseLayer, EmbeddingLayer, FlattenLayer, DropoutLayer, BatchNormalizationLayer, GlobalAveragePoolingLayer, GlobalAveragePooling1D
from losses import BinaryCrossEntropy   
from optimizer import Optimizer
from metrics import accuracy
from data import read_parquet, read_csv
from activation import SigmoidActivation, ReLUActivation
import pickle
# from visualization import plot_history  # Removido, pois o histórico não será armazenado

train_file = '../../../datasets/dataset_train.csv'
test_file = '../../../datasets/dataset_test.csv'
final_file = '../../../datasets/dataset_processed.csv'


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
        # Histórico removido

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

        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None

        for epoch in range(1, self.epochs + 1):
            train_loss, train_metric = 0, 0
            batch_count = 0
            for X_batch, y_batch in self.get_mini_batches(X, y):
                output = self.forward_propagation(X_batch, training=True)
                error = self.loss.derivative(y_batch, output)
                self.backward_propagation(error)
                
                train_loss += self.loss.loss(y_batch, output)
                train_metric += self.metric(y_batch, output)
                batch_count += 1

            train_loss /= batch_count
            train_metric /= batch_count

            val_loss, val_metric = None, None
            if validation_data:
                val_output = self.predict(validation_data)
                val_loss = self.loss.loss(validation_data.y, val_output)
                val_metric = self.metric(validation_data.y, val_output)

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
                if validation_data:
                    print(f"Epoch {epoch}/{self.epochs} - loss: {train_loss:.4f} - accuracy: {train_metric:.4f} "
                          f"- val_loss: {val_loss:.4f} - val_accuracy: {val_metric:.4f}")
                else:
                    print(f"Epoch {epoch}/{self.epochs} - loss: {train_loss:.4f} - accuracy: {train_metric:.4f}")
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
    # train_data = read_csv(train_file)
    # test_data = read_csv(test_file)
    final_data = read_csv(final_file)
    

    net = DeepNeuralNetwork(epochs=12, batch_size=32, learning_rate=0.003, verbose=True,
                            loss=BinaryCrossEntropy, metric=accuracy)

    n_features = final_data.X.shape[1]

    net.add(EmbeddingLayer(vocab_size=7500, embedding_dim=16, input_shape=(n_features,)))
    net.add(GlobalAveragePooling1D())
    

    net.add(DenseLayer(32, l2=0.01))
    net.add(BatchNormalizationLayer())
    net.add(ReLUActivation())
    net.add(DropoutLayer(0.3))

    net.add(DenseLayer(16, l2=0.02))
    net.add(ReLUActivation())
    net.add(DropoutLayer(0.3))

    net.add(DenseLayer(1))  
    net.add(SigmoidActivation())

    net.fit(final_data, patience=20)
    # plot_history(net.history)  # Removido, pois o histórico não é mais utilizado

    while True:
        opt = input("Queres guardar? [y/n]")
        if opt == "y":
            net.save("../../../models/modelo_dnn.pkl")
            break
        elif opt == "n":
            break
