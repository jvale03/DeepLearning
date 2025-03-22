import re
import numpy as np
from transformers import BertTokenizerFast
from recurrent_neural_net import RecurrentNeuralNetwork

# Carrega o tokenizer customizado
tokenizer = BertTokenizerFast.from_pretrained("custom_tokenizer", local_files_only=True)

# Função para limpeza do texto
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Função para processar textos
def process_texts(texts):
    tokenized = tokenizer(
        texts, padding="max_length", truncation=True, max_length=64
    )["input_ids"]
    return tokenized

# Load the trained RNN model
try:
    rnn = RecurrentNeuralNetwork.load("models/rnn_model.pkl")
    print("RNN model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please train and save the model first.")
    exit()

# Classe mock para compatibilidade com o modelo
class MockDataset:
    def __init__(self, X):
        self.X = X

# Loop de previsão
while True:
    opt = input("Insere o texto (ou escreve 'STOP' para sair): ")
    if opt.strip().upper() == "STOP":
        break
    
    opt_clean = clean_text(opt)
    opt_processed = process_texts([opt_clean])
    mock_dataset = MockDataset(np.array(opt_processed))
    prediction = rnn.predict(mock_dataset)
    prediction_label = "Humano" if prediction <= 0.5 else "IA"
    print(f"Previsão do modelo: {prediction_label} (Confiança: {prediction[0][0]:.4f})")