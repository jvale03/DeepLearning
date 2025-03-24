import numpy as np

class SimpleTokenizer:
    def __init__(self, num_words=None, lower=True, split=" "):

        self.num_words = num_words
        self.lower = lower
        self.split = split
        self.word_index = {}   # mapeamento palavra -> índice
        self.index_word = {}   # mapeamento índice -> palavra
        self.word_counts = {}  # contagem de ocorrência das palavras

    def fit_on_texts(self, texts):

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

        sequences = []
        for text in texts:
            if self.lower:
                text = text.lower()
            words = text.split(self.split)
            seq = [self.word_index.get(word, 0) for word in words]  # 0 se a palavra não estiver no vocabulário
            sequences.append(np.array(seq, dtype=np.int32))
        return sequences


