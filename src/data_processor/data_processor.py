
import pandas as pd
import re
from transformers import BertTokenizerFast
from tokenizers import BertWordPieceTokenizer
from sklearn.model_selection import train_test_split
import os


corpus_file = "./custom_tokenizer/corpus.txt"

file = "../../datasets/new_data/new.csv"

def reshape_df(df):
    category_mapping = {"Human": 0, "AI": 1, "student": 0}
    df["Label"] = df["Source"].map(category_mapping)
    df = df[["Text", "Label"]]
    return df

# def clean_text(text: str) -> str:
#     text = text.lower()
#     text = re.sub(r'[^a-z\s]', '', text)
#     return text
# este novo mantem a pontuação

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s.,]', '', text)  # Mantém letras, espaços, pontos e vírgulas
    return text

def load_tokenizer():
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    return tokenizer

def create_tokenizer(df):
    corpus_texts = []

    texts = [clean_text(text) for text in df["Text"].tolist()]
    corpus_texts.extend(texts)

    with open(corpus_file, "w", encoding="utf-8") as f:
        for line in corpus_texts:
            f.write(line.strip() + "\n")

    custom_tokenizer = BertWordPieceTokenizer(lowercase=True)
    custom_tokenizer.train(
        files=[corpus_file],
        vocab_size=5000,
        min_frequency=5,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    tokenizer_directory = "./custom_tokenizer"
    os.makedirs(tokenizer_directory, exist_ok=True)

    custom_tokenizer.save_model(tokenizer_directory)

    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_directory)

    return tokenizer

def process_text(df, tokenizer):
    texts = [clean_text(text) for text in df["Text"].tolist()]
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
    )["input_ids"]
    df["Text"] = tokenized
    return df

def split_data(df):
    df = df[["Text","Label"]]
    train_df, test_df = train_test_split(
        df, test_size=0.15, shuffle=True, random_state=42, stratify=df["Label"]
    )
    train_file = file.replace(".csv", "_train.csv")
    test_file = file.replace(".csv", "_test.csv")
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

if __name__ == "__main__":
    df = pd.read_csv(file)

    #df = reshape_df(df)

    tokenizer = create_tokenizer(df)

    df = process_text(df,tokenizer)

    split_data(df)

    print("Ta bala irmao!")