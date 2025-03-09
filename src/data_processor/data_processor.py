import pandas as pd
import re
from transformers import BertTokenizerFast
from tokenizers import BertWordPieceTokenizer
from sklearn.model_selection import train_test_split
import os

file = "../../datasets/new_data/Large_Physics_and_Science_Dataset.csv"

corpus_file = "./custom_tokenizer/corpus.txt"

def reshape_df(df):
    category_mapping = {"human": 0, "ai": 1, "student": 0}

    df["label"] = df["Source"].map(category_mapping)
    df["text"] = df["Text"]
    df = df[["text","label"]]

    return df


def clean_text(text: str) -> str:
    """Limpa o texto removendo caracteres especiais e deixa tudo em minúsculo."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


def create_tokenizer(df):
    corpus_texts = []

    texts = [clean_text(text) for text in df["text"].tolist()]
    corpus_texts.extend(texts)

    with open(corpus_file, "w", encoding="utf-8") as f:
        for line in corpus_texts:
            f.write(line.strip() + "\n")

    custom_tokenizer = BertWordPieceTokenizer(lowercase=True)
    custom_tokenizer.train(
        files=[corpus_file],
        vocab_size=5000,
        min_frequency=0,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    tokenizer_directory = "./custom_tokenizer"
    os.makedirs(tokenizer_directory, exist_ok=True)

    custom_tokenizer.save_model(tokenizer_directory)

    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_directory)

    return tokenizer

def process_text(df,tokenizer):
    texts = [clean_text(text) for text in df["text"].tolist()]

    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=64,
    )["input_ids"]

    df["text"] = tokenized

    return df

def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.25, shuffle=True, random_state=42)
    train_file = file.replace(".csv", "_train.parquet")
    test_file = file.replace(".csv", "_test.parquet")

    train_df.to_parquet(train_file, index=False)
    test_df.to_parquet(test_file, index=False)


if __name__ == "__main__":
    df = pd.read_csv(file)

    df = reshape_df(df)

    tokenizer = create_tokenizer(df)

    df = process_text(df,tokenizer)

    split_data(df)

    print("Ta bala irmao!")
