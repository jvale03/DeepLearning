# csvs = [
#     "../../datasets/test/merged_1_test.parquet",
#     "../../datasets/test/merged_2_test.parquet",
#     "../../datasets/train/merged_1_train.parquet",
#     "../../datasets/train/merged_2_train.parquet",
# ]

import pandas as pd
import re
from transformers import BertTokenizerFast
from tokenizers import BertWordPieceTokenizer  # pip install tokenizers
import os

# Função de limpeza (opcional, dependendo do seu caso)
def clean_text(text: str) -> str:
    """Limpa o texto removendo caracteres especiais e deixa tudo em minúsculo."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Lista de arquivos de exemplo
csvs = [
    "../../datasets/test/test_sample.parquet",
    "../../datasets/train/train_sample.parquet",
]

# 1. Coleta dos textos do corpus e limpeza (se desejar)
corpus_texts = []
for file in csvs:
    df = pd.read_parquet(file)
    # Se desejar aplicar a limpeza antes do treinamento:
    texts = [clean_text(text) for text in df["text"].tolist()]
    corpus_texts.extend(texts)

# 2. Salva os textos em um arquivo temporário
corpus_file = "corpus.txt"
with open(corpus_file, "w", encoding="utf-8") as f:
    for line in corpus_texts:
        f.write(line.strip() + "\n")

# 3. Treina o tokenizer customizado com vocabulário de tamanho reduzido (por exemplo, 5000 tokens)
#    Aqui usamos o BertWordPieceTokenizer para manter compatibilidade com o formato BERT.
custom_tokenizer = BertWordPieceTokenizer(lowercase=True)
custom_tokenizer.train(
    files=[corpus_file],
    vocab_size=10000,
    min_frequency=2,
    show_progress=True,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# 4. Salva o tokenizer customizado em um diretório
tokenizer_directory = "./custom_tokenizer"
os.makedirs(tokenizer_directory, exist_ok=True)

custom_tokenizer.save_model(tokenizer_directory)

# 5. Carrega o tokenizer customizado usando o BertTokenizerFast da HuggingFace
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_directory)

# Função para processar os textos utilizando o tokenizer customizado
def process_texts(texts):
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
    )["input_ids"]
    return tokenized

# 6. Processa e salva os arquivos com os textos tokenizados
for file in csvs:
    df = pd.read_parquet(file)
    # Se os textos não foram limpos previamente, você pode aplicar a limpeza aqui também:
    texts = [clean_text(text) for text in df["text"].tolist()]
    df["text"] = process_texts(texts)
    
    new_file = file.replace(".parquet", "_processed.parquet")
    df.to_parquet(new_file, index=False)
    print(f"Arquivo processado e salvo: {new_file}")
