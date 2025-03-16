import numpy as np
from layers import DenseLayer, RNNLayer, DropoutLayer, BatchNormalizationLayer
from activation import SigmoidActivation, ReLUActivation
from losses import BinaryCrossEntropy
from optimizer import Optimizer
from metrics import accuracy
from data import read_parquet
import pickle
from visualization import plot_history

class RecurrentNeuralNetwork:
    def __init__(self, epochs=100, batch_size=32, optimizer=None, 
                 learning_rate=0.01, momentum=0.90, verbose=True,
                 loss=BinaryCrossEntropy, metric=accuracy):
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = Optimizer(learning_rate=learning_rate, momentum=momentum)
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric

        self.layers = []
        self.history = {'train_loss' : [], 'train_metric' : [], 
                        'val_loss' : [], 'val_metric' : []}
        
    def add(self, layer):
        if self.layers:
            # Setting the input shape to the output shape of the last added layer
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            # Calling the initialize method if the layer has it
            layer.initialize(self.optimizer)
        self.layers.append(layer)
        return self

    def get_mini_batches(self, X, y=None, shuffle=True):
        n_samples = X.shape[0]
        if self.batch_size > n_samples:
            self.batch_size = n_samples
        
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
        # If X is (batch_size, features), reshape to (batch_size, seq_len, features)
        if len(X.shape) == 2:
            return X.reshape(X.shape[0], X.shape[1], 1)
        return X
    
    def fit(self, dataset, validation_data=None, patience=5):
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
                    for i, layer in enumerate(self.layers):
                        if hasattr(layer, 'set_weights'):
                            layer.set_weights(best_weights[i])
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
    
    def score(self,dataset):
        predictions = self.predict(dataset)
        if self.metric is not None:
            return self.metric(dataset.y, predictions)
        raise ValueError("No metric specified for the neural network")
    
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

# Creating and training the RNN model    
if __name__ == '__main__':
    print('Started')

    train_data = read_parquet('datasets/new_data/Merged_Dataset_train.parquet')
    test_data = read_parquet('datasets/new_data/Merged_Dataset_test.parquet')

    version = 1
    learning_rates = [0.1, 0.01, 0.001]
    
    for l in learning_rates:
        # Creating a RNN model
        rnn = RecurrentNeuralNetwork(
            epochs=10,
            batch_size=16,
            learning_rate=l,
            momentum=0.9,
            verbose=True
        )
        
        print('Created model architecture')

        n_features = train_data.X.shape[1]

        # Build RNN architecture
        # In this architecture, each token is treated as a timestep with dim=1
        # Input shape: (batch_size, sequence_length, 1)
        rnn.add(RNNLayer(32, input_shape=(n_features, 1), return_sequences=True, bptt_trunc=None))
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
        rnn.fit(train_data, validation_data=test_data, patience=7)

        print('Model trained')
        
        # Plot training history
        #plot_history(rnn.history)
        
        # Save model prompt
        #save_model = input("Do you want to save the model? [y/n]: ")
        #if save_model.lower() == 'y':
        rnn.save(f"models/rnn_model_v{version}.pkl")
        print('Model saved')
        version+=1