import re
import numpy as np
import pandas as pd
from transformers import BertTokenizerFast
from deep_neural_net import DeepNeuralNetwork

# Carrega o tokenizer customizado
tokenizer = BertTokenizerFast.from_pretrained("../data_processor/./custom_tokenizer")

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

# Carrega o modelo treinado
dnn = DeepNeuralNetwork()
dnn = dnn.load("../../models/modelo_treinado.pkl")

# Classe mock para compatibilidade com o modelo
class MockDataset:
    def __init__(self, X):
        self.X = X

# Função para prever um único texto
def predict_text(text):
    text_clean = clean_text(text)
    text_processed = process_texts([text_clean])
    mock_dataset = MockDataset(np.array(text_processed))
    prediction = dnn.predict(mock_dataset)
    prediction_label = "Humano" if prediction[0][0] >= 0.5 else "IA"
    print(f"Previsão do modelo: {prediction_label} (Confiança: {prediction[0][0]:.4f})")

# Função para prever a partir de um ficheiro de texto
def predict_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]
    
    processed_texts = process_texts([clean_text(t) for t in texts])
    mock_dataset = MockDataset(np.array(processed_texts))
    predictions = dnn.predict(mock_dataset)

    df = pd.DataFrame()
    df["previsao"] = ["Humano" if p[0] >= 0.5 else "IA" for p in predictions]
    df["confianca"] = [p[0] for p in predictions]

    output_file = "../../datasets/sores/predicoes.csv"
    df.to_csv(output_file, index=False)
    print(f"Previsões salvas em {output_file}")

# Menu interativo
while True:
    opt = input("Escolhe uma opção: (1) Texto manual (2) Carregar TXT (3) Sair: ").strip()
    
    if opt == "1":
        text = input("Insere o texto: ")
        predict_text(text)

    elif opt == "2":
        predict_from_txt("../../datasets/sores/dataset1_inputs.csv")

    elif opt == "3":
        break

    else:
        print("Opção inválida. Tenta novamente.")