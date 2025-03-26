import re
import numpy as np
from collections import Counter
from itertools import chain

class SimpleTokenizer:
    def __init__(self, num_words=None, lower=True, split=" ", seed=None):
        self.num_words = num_words
        self.lower = lower
        self.split = split
        self.word_index = {}   # mapeamento palavra -> índice
        self.index_word = {}   # mapeamento índice -> palavra
        self.word_counts = {}  # contagem de ocorrência das palavras
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)

    def fit_on_texts(self, texts):
        if self.seed is not None:
            np.random.seed(self.seed)

        for text in texts:
            if self.lower:
                text = text.lower()
            words = text.split(self.split)
            for word in words:
                if word in self.word_counts:
                    self.word_counts[word] += 1
                else:
                    self.word_counts[word] = 1


        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)

        if self.num_words:
            sorted_words = sorted_words[:self.num_words - 1]  # índice 0 reservado para padding


        self.word_index = {word: i+1 for i, (word, count) in enumerate(sorted_words)}
        self.index_word = {i+1: word for i, (word, count) in enumerate(sorted_words)}

    def texts_to_sequences(self, texts):
        if self.seed is not None:
            np.random.seed(self.seed)

        sequences = []
        for text in texts:
            if self.lower:
                text = text.lower()
            words = text.split(self.split)
            seq = [self.word_index.get(word, 0) for word in words]  # 0 se a palavra não estiver no vocabulário
            sequences.append(np.array(seq, dtype=np.int32))
        return sequences



class RobustTokenizer:
    def __init__(self, num_words=None, lower=True, seed=None):
        self.num_words = num_words
        self.lower = lower
        self.word_index = {}   # mapeamento palavra -> índice
        self.index_word = {}   # mapeamento índice -> palavra
        self.word_counts = {}  # contagem de ocorrência das palavras
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)

    def tokenize_text(self, text):
        if self.seed is not None:
            np.random.seed(self.seed)
        """Tokeniza o texto utilizando expressão regular para capturar palavras."""
        if self.lower:
            text = text.lower()
        # A expressão abaixo captura palavras alfanuméricas; ela ignora pontuação.
        tokens = re.findall(r'\w+', text, flags=re.UNICODE)
        return tokens

    def fit_on_texts(self, texts):
        if self.seed is not None:
            np.random.seed(self.seed)
        """Constroi o vocabulário com base na frequência das palavras."""
        for text in texts:
            tokens = self.tokenize_text(text)
            for token in tokens:
                self.word_counts[token] = self.word_counts.get(token, 0) + 1

        # Ordena as palavras por frequência (maior para menor)
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)

        # Se num_words for definido, limita o vocabulário (o índice 0 é reservado para padding)
        if self.num_words:
            sorted_words = sorted_words[:self.num_words - 1]

        self.word_index = {word: i+1 for i, (word, count) in enumerate(sorted_words)}
        self.index_word = {i+1: word for i, (word, count) in enumerate(sorted_words)}

    def texts_to_sequences(self, texts):
        if self.seed is not None:
            np.random.seed(self.seed)
        """Converte os textos em sequências de inteiros com base no vocabulário construído."""
        sequences = []
        for text in texts:
            tokens = self.tokenize_text(text)
            seq = [self.word_index.get(token, 0) for token in tokens]  # 0 se a palavra não estiver no vocabulário
            sequences.append(np.array(seq, dtype=np.int32))
        return sequences


class AdvancedTokenizer:
    def __init__(self, num_words=None, lower=True, ngrams=(1, 2), remove_stopwords=False, stopwords_list=None, min_freq=5, seed=42):
        self.num_words = num_words
        self.lower = lower
        self.ngrams = ngrams  
        self.remove_stopwords = remove_stopwords
        self.word_counts = Counter()
        self.word_index = {}
        self.index_word = {}
        self.min_freq = min_freq  # Nova variável para frequência mínima
        self.seed = seed

        # Definir o seed globalmente uma vez no início
        np.random.seed(self.seed)

        # Stopwords personalizadas
        self.stopwords = set(stopwords_list) if stopwords_list else set()

    def tokenize_text(self, text):
        """Tokeniza o texto, removendo pontuação e aplicando regex aprimorado."""
        if self.lower:
            text = text.lower()

        tokens = re.findall(r"\b\w+(?:[-']\w+)?\b", text, re.UNICODE)

        # Remove stopwords, se necessário
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        return tokens

    def generate_ngrams(self, tokens):
        """Gera unigramas, bigramas, trigramas, etc."""
        return [' '.join(tokens[i:i+n]) for n in self.ngrams for i in range(len(tokens) - n + 1)]

    def fit_on_texts(self, texts):
        """Cria o vocabulário baseado na frequência das palavras e n-grams."""
        for text in texts:
            tokens = self.tokenize_text(text)
            tokens = self.generate_ngrams(tokens)
            self.word_counts.update(tokens)

        # Filtra os tokens com base na frequência mínima
        filtered_words = {word: count for word, count in self.word_counts.items() if count >= self.min_freq}

        # Ordena palavras por frequência decrescente e, em caso de empate, em ordem alfabética
        sorted_words = [word for word, _ in sorted(filtered_words.items(), key=lambda item: (-item[1], item[0]))]
        if self.num_words:
            sorted_words = sorted_words[:self.num_words]  # Limita o vocabulário ao número máximo de palavras

        self.word_index = {word: i+1 for i, word in enumerate(sorted_words)}
        self.index_word = {i+1: word for i, word in enumerate(sorted_words)}


    def texts_to_sequences(self, texts, max_len=100):
        """Converte textos em sequências numéricas e aplica padding/truncamento."""
        sequences = []
        for text in texts:
            tokens = self.tokenize_text(text)
            tokens = self.generate_ngrams(tokens)
            seq = [self.word_index.get(token, 0) for token in tokens]
            sequences.append(seq)

        # Padding/truncamento manual sem usar Keras
        padded_sequences = np.zeros((len(sequences), max_len), dtype=np.int32)
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_len)
            padded_sequences[i, :length] = seq[:length]

        return padded_sequences
    
    