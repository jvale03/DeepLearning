import csv
import re

def read_sentences_from_txt(file_path):
    """Reads the text file and splits it into real sentences."""
    with open(file_path, mode='r', encoding='utf-8') as file:
        text = file.read()  # Read the whole content

    # Split text into sentences using punctuation as delimiters
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Remove extra spaces
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences

def merge_small_sentences(sentences, min_words=80):
    """Merges small sentences until they reach the minimum word count."""
    merged_sentences = []
    current_sentence = ""

    for sentence in sentences:
        if current_sentence:  # If there's already a sentence being built
            combined = current_sentence + " " + sentence
            if len(combined.split()) >= min_words:
                merged_sentences.append(combined.strip())  # Save it as a full sentence
                current_sentence = ""  # Reset
            else:
                current_sentence = combined  # Keep merging
        else:
            if len(sentence.split()) >= min_words:
                merged_sentences.append(sentence.strip())  # Save it directly
            else:
                current_sentence = sentence  # Start merging

    # If there's a leftover sentence that never reached min_words, add it
    if current_sentence:
        merged_sentences.append(current_sentence.strip())

    return merged_sentences

def write_sentences_to_csv(sentences, output_csv):
    """Saves the processed sentences into a CSV file."""
    with open(output_csv, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Text", "Label"])  # Header
        for sentence in sentences:
            writer.writerow([sentence, 0])  # Save sentence with label 0

def main():
    input_txt = "all_sentences.txt"
    output_csv = "final.csv"

    sentences = read_sentences_from_txt(input_txt)  # Read and split into sentences
    processed_sentences = merge_small_sentences(sentences, min_words=120)  # Merge smaller ones

    write_sentences_to_csv(processed_sentences, output_csv)  # Save results
    print(f"Processed dataset saved to {output_csv}")

if __name__ == '__main__':
    main()