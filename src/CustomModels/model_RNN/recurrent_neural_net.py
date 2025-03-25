import numpy as np
from layers import DenseLayer, RNNLayer, DropoutLayer, BatchNormalizationLayer, EmbeddingLayer
from activation import SigmoidActivation, ReLUActivation
from losses import BinaryCrossEntropy
from optimizer import Optimizer
from metrics import accuracy
from data import read_csv
import pickle
from visualization import plot_history
from tokenizer import SimpleTokenizer

file = '../../../datasets/final_dataset.csv'


class RecurrentNeuralNetwork:
    def __init__(self, epochs=100, batch_size=32, optimizer=None, 
                 learning_rate=0.01, momentum=0.90, verbose=True,
                 loss=BinaryCrossEntropy, metric=accuracy, seed=None):
        
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = Optimizer(learning_rate=learning_rate, momentum=momentum, seed=self.seed)
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric

        self.layers = []
        self.history = {'train_loss' : [], 'train_metric' : [], 
                        'val_loss' : [], 'val_metric' : []}
        
    def add(self, layer):
            if hasattr(layer, 'set_seed') and self.seed is not None:
                layer.set_seed(self.seed)
                
            if self.layers:
                layer.set_input_shape(input_shape=self.layers[-1].output_shape())
                
            if hasattr(layer, 'initialize'):
                layer.initialize(self.optimizer)
                
            self.layers.append(layer)
            return self

    def get_mini_batches(self, X, y=None, shuffle=True):
        n_samples = X.shape[0]
        if self.batch_size > n_samples:
            self.batch_size = n_samples
        
        if self.seed is not None and shuffle:
            np.random.seed(self.seed)

        indices = np.random.permutation(n_samples) if shuffle else np.arange(n_samples)

        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch_indices = indices[start:end]
            if y is not None:
                yield X[batch_indices], y[batch_indices]
            else:
                yield X[batch_indices], None

    def forward_propagation(self, X, training):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def backward_propagation(self, output_error):
        error = output_error
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)
        return error
    
    def reshape_for_rnn(self, X):
        return X
    
    def fit(self, dataset, validation_data=None, patience=5):
        if self.seed is not None:
            np.random.seed(self.seed)
        X,y = dataset.X, dataset.y
        # reshaping X
        X = self.reshape_for_rnn(X)
        # resahping y
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)

        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = []

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")
            (train_loss, train_metric) = (0, 0)
            batch_count = 0

            # Training in each batch
            for X_batch, y_batch in self.get_mini_batches(X,y):

                print(f"Batch {batch_count + 1}/{len(X) // self.batch_size}", end='\r')
                # Forward pass to get a prediction
                output = self.forward_propagation(X_batch, training=True)

                # Calculating the gradient of the loss function
                error = self.loss.derivative(y_batch, output)

                # Backward pass to adjust the weights
                self.backward_propagation(error)

                # Calculating metric
                batch_loss = self.loss.loss(y_batch, output)
                batch_metric = self.metric(y_batch, output)

                train_loss += batch_loss
                train_metric += batch_metric
                batch_count += 1

            # Calculating average metric for the epoch
            train_loss = train_loss / batch_count
            train_metric = train_metric / batch_count
            
            # Storing training history
            self.history['train_loss'].append(train_loss)
            self.history['train_metric'].append(train_metric)

            # Validation phase
            (val_loss, val_metric) = (None, None)
            if validation_data:
                val_X, val_y = validation_data.X, validation_data.y

                # Reshaping validation X
                val_X = self.reshape_for_rnn(val_X)

                # Reshaping validation y
                if np.ndim(val_y) == 1:
                    val_y = np.expand_dims(val_y, axis=1)

                # Making predictions on the actual validation data
                val_output = self.predict(validation_data)

                # Calculating validation metrics
                val_loss = self.loss.loss(val_y, val_output)
                val_metric = self.metric(val_y, val_output)

                # Storing validation history
                self.history['val_loss'].append(val_loss)
                self.history['val_metric'].append(val_metric)

                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = [layer.get_weights() for layer in self.layers if hasattr(layer, 'get_weights')]
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"\nEarly stopping activated at epoch {epoch}. Best val_loss: {best_val_loss:.4f}")
                    # Restoring best weights
                    weight_index = 0
                    for layer in self.layers:
                        if hasattr(layer, 'set_weights'):
                            layer.set_weights(best_weights[weight_index])
                            weight_index += 1
                    break
            
            if self.verbose:
                val_loss_str = f"- val_loss: {val_loss:.4f}" if val_loss is not None else ""
                val_metric_str = f"- val_accuracy: {val_metric:.4f}" if val_metric is not None else ""
                print(f"Epoch {epoch}/{self.epochs} - loss: {train_loss:.4f} - accuracy: {train_metric:.4f} "
                      f"{val_loss_str} {val_metric_str}")
                
        return self

    # Convenience methods - not really being used in this project besides for save and load
    def predict(self, dataset):
        X = dataset.X
        X = self.reshape_for_rnn(X)
        return self.forward_propagation(X, training=False)
    
    def score(self, dataset, predictions):
        if self.metric is not None:
            return self.metric(dataset.y, predictions)
        raise ValueError("No metric specified for the neural network.")
    
    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {file_path}")

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {file_path}")
        return model

if __name__ == '__main__':
    print('Started')

    print("Tokenizing csv...")
    tokenizer = SimpleTokenizer(num_words=10000)
    print("CSV tokenized!")

    train_data, validation_data, test_data = read_csv(file, tokenizer)

    # Creating a RNN model
    rnn = RecurrentNeuralNetwork(
        epochs=3,
        batch_size=32,
        learning_rate=0.01,
        momentum=0.9,
        verbose=True
    )
    
    print('Created model architecture')
    n_features = train_data.X.shape[1]
    # Build RNN architecture
    rnn.add(EmbeddingLayer(vocab_size=10000, embedding_dim=8, input_shape=(n_features,)))
    rnn.add(RNNLayer(32, return_sequences=True, bptt_trunc=None))
    rnn.add(RNNLayer(16, return_sequences=False, bptt_trunc=None))
    rnn.add(BatchNormalizationLayer())
    rnn.add(ReLUActivation())
    rnn.add(DropoutLayer(dropout_rate=0.3))
    rnn.add(DenseLayer(8))
    rnn.add(ReLUActivation())
    rnn.add(DenseLayer(1))
    rnn.add(SigmoidActivation())
    print('Added layers to model')
    
    # Train the model
    rnn.fit(train_data, validation_data=validation_data, patience=5)
    print('Model trained')
    
    # Plot training history
    plot_history(rnn.history)

    test_predictions = rnn.predict(test_data)
    test_score = rnn.score(test_data, test_predictions)
    print(f"Accuracy no dataset de teste: {test_score:.4f}")

    
    save_model = input("Do you want to save the model? [y/n]: ")
    if save_model.lower() == 'y':
        rnn.save("../../../models/rnn_model.pkl")
        print('Model saved')