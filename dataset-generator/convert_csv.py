import csv
import re

def read_sentences_from_txt(file_path):
    """Lê frases do arquivo TXT e retorna uma lista de frases válidas."""
    with open(file_path, mode='r', encoding='utf-8') as file:
        sentences = file.readlines()
    
    # Remover espaços extras e quebras de linha
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences

def filter_sentences(sentences, min_words=80, max_words=120):
    """Filtra frases que tenham entre min_words e max_words palavras e terminem com '.'"""
    filtered_sentences = [sent for sent in sentences if min_words <= len(sent.split()) <= max_words and sent.endswith('.')]
    return filtered_sentences

def write_sentences_to_csv(sentences, output_csv):
    """Salva as frases filtradas em um arquivo CSV."""
    with open(output_csv, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Text","Label"])
        for sentence in sentences:
            writer.writerow([sentence,0])

def main():
    input_txt = "teste.txt" 
    output_csv = "Human_Dataset.csv"

    sentences = read_sentences_from_txt(input_txt)
    filtered_sentences = filter_sentences(sentences, min_words=80, max_words=120)
    
    write_sentences_to_csv(filtered_sentences, output_csv)
    print(f"Frases filtradas salvas em {output_csv}")

if __name__ == '__main__':
    main()