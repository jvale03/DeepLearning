import pandas as pd
import spacy
import re
from tqdm import tqdm

# Load spaCy NLP model
nlp = spacy.load("en_core_web_lg")

# Define keywords for scientific categories
KEYWORDS = {
    "Physics": ["quantum", "atom", "energy", "mass", "force", "particle", "relativity", "wave", "thermodynamics",
                "cosmology", "experiment", "nuclear", "astrophysics", "optics", "mechanics", "gravitation"],
    "Mathematics": ["algebra", "geometry", "topology", "calculus", "equation", "theorem", "proof", "matrix",
                    "statistics", "probability", "optimization", "tensor", "graph theory", "set theory"],
    "Computer Science": ["algorithm", "data structure", "programming", "artificial intelligence",
                         "machine learning", "neural network", "database", "cybersecurity", "blockchain",
                         "cloud computing", "quantum computing", "robotics"],
    "Quantitative Biology": ["bioinformatics", "computational biology", "systems biology", "genomic sequencing",
                              "neuroscience", "biostatistics", "epidemiology", "synthetic biology"],
    "Quantitative Finance": ["financial modeling", "risk management", "portfolio optimization",
                             "stochastic processes", "derivatives pricing", "market microstructure",
                             "high-frequency trading", "asset pricing"],
    "Statistics": ["Bayesian inference", "hypothesis testing", "regression analysis", "data visualization",
                   "sampling theory", "variance", "standard deviation", "time series analysis", "clustering"],
    "Electrical Engineering": ["circuits", "signal processing", "control systems", "digital signal processing",
                               "wireless communication", "microelectronics", "power electronics",
                               "antenna design", "IoT", "automation"],
    "Economics": ["macroeconomics", "microeconomics", "game theory", "economic modeling", "supply and demand",
                  "inflation", "GDP", "monetary policy", "economic growth", "taxation", "income distribution"]
}

def clean_text(text):
    """Clean text by removing numbers, non-ASCII characters, and extra spaces."""
    text = re.sub(r'\([0-9]+(\.[0-9]+)?\)', '', text)  # Remove citations like (2023)
    text = re.sub(r'\b[0-9]+(\.[0-9]+)?\b', '', text)  # Remove standalone numbers
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def extract_relevant_sentences(text):
    """Extract sentences containing keywords."""
    doc = nlp(text)
    relevant_sentences = []
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        for keywords in KEYWORDS.values():
            if any(keyword in sentence_text.lower() for keyword in keywords):
                relevant_sentences.append(sentence_text)
                break  # Avoid duplicate matches
    return relevant_sentences

def process_parquet(input_file, output_txt, batch_size=1000, start_batch=100):
    """Process Parquet file in batches, starting from the specified batch, and append relevant sentences to output file."""
    reader = pd.read_parquet(input_file, engine="pyarrow", columns=["text"])  # Read only 'text' column

    total_batches = (len(reader) + batch_size - 1) // batch_size  # Compute total batches
    tqdm.pandas(desc="Processing Text Entries")

    # Open file in append mode
    with open(output_txt, "a", encoding="utf-8") as f:
        for batch in tqdm(range(start_batch, total_batches), desc="Processing Batches"):
            batch_start = batch * batch_size
            batch_end = batch_start + batch_size
            chunk = reader.iloc[batch_start:batch_end].copy()

            chunk["text"] = chunk["text"].astype(str).apply(clean_text)
            chunk_sentences = chunk["text"].progress_apply(extract_relevant_sentences)

            # Flatten sentences and write to file
            sentences_to_write = [sentence for sentences in chunk_sentences for sentence in sentences]
            if sentences_to_write:
                f.write("\n".join(sentences_to_write) + "\n")

    print(f"Extracted sentences from batch {start_batch} onwards appended to {output_txt}")

if __name__ == "__main__":
    input_parquet = "filtered_data.parquet"
    output_file = "all_sentences.txt"
    start_batch = 1  # Start from batch 100

    process_parquet(input_parquet, output_file, batch_size=1000, start_batch=start_batch)
