import pandas as pd
import numpy as np

class Data:
    def __init__(self, X, y=None, features=None, label=None):
        if X is None:
            raise ValueError("X cannot be None")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if features is not None and X.shape[1] != len(features):
            raise ValueError("Number of features must match the number of columns in X")
        if features is None:
            features = [f"feat_{i}" for i in range(X.shape[1])]
        if y is not None and label is None:
            label = "y"
        self.X = X
        self.y = y
        self.features = features
        self.label = label

def read_csv(filename, tokenizer, sequence_length=128, train_ratio=0.7, val_ratio=0.15):
    data = pd.read_csv(filename)

    if data.shape[1] != 2:
        raise ValueError("O dataset deve ter exatamente duas colunas: uma independente e uma dependente.")
    
    texts = data.iloc[:, 0].astype(str).tolist()
    labels = data.iloc[:, 1].astype(np.float32).to_numpy()
    
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Padding para garantir que todas as sequências tenham o mesmo comprimento
    X = np.zeros((len(sequences), sequence_length), dtype=np.int32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), sequence_length)
        X[i, :length] = seq[:length]
    
    # Divisão dos dados
    total_samples = X.shape[0]
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size
    
    indices = np.random.permutation(total_samples)
    train_idx, val_idx, test_idx = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]
    
    X_train, y_train = X[train_idx], labels[train_idx]
    X_val, y_val = X[val_idx], labels[val_idx]
    X_test, y_test = X[test_idx], labels[test_idx]
    
    features = [f"feat_{i}" for i in range(sequence_length)]
    label = data.columns[1]
    
    return (Data(X=X_train, y=y_train, features=features, label=label),
            Data(X=X_val, y=y_val, features=features, label=label),
            Data(X=X_test, y=y_test, features=features, label=label))



def read_csv_once(data, tokenizer, sequence_length=128):

    if data.shape[1] != 2:
        raise ValueError("O dataset deve ter exatamente duas colunas: uma independente e uma dependente.")
    
    texts = data.iloc[:, 0].astype(str).tolist()
    labels = data.iloc[:, 1].astype(np.float32).to_numpy()
    
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    X = np.zeros((len(sequences), sequence_length), dtype=np.int32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), sequence_length)
        X[i, :length] = seq[:length]
    
    features = [f"feat_{i}" for i in range(sequence_length)]
    label = data.columns[1]
    
    return Data(X=X, y=labels, features=features, label=label)