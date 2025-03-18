import pandas as pd
import spacy
import re
import csv
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

def contains_relevant_keywords(text):
    """Check if the text contains any relevant keywords."""
    doc = nlp(text)
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        for keywords in KEYWORDS.values():
            if any(keyword in sentence_text.lower() for keyword in keywords):
                return True  # If any sentence contains a keyword, return True
    return False  # If no sentence contains a keyword, return False

def process_csv(input_file, output_csv, batch_size=1000, start_batch=0):
    """Process CSV file in batches, starting from the specified batch, and append relevant sentences to output CSV."""
    reader = pd.read_csv(input_file, usecols=["text", "generated"])  # Read 'text' and 'generated' columns

    total_batches = (len(reader) + batch_size - 1) // batch_size  # Compute total batches
    tqdm.pandas(desc="Processing Text Entries")

    # Open output CSV in append mode
    with open(output_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        
        # Write header if the output CSV is empty
        f.seek(0, 2)  # Move to the end of the file
        if f.tell() == 0:  # Check if the file is empty
            writer.writerow(["Text", "Label"])  # Write header row if file is empty

        for batch in tqdm(range(start_batch, total_batches), desc="Processing Batches"):
            batch_start = batch * batch_size
            batch_end = batch_start + batch_size
            chunk = reader.iloc[batch_start:batch_end].copy()

            # Only process rows where 'generated' is 1
            chunk = chunk[chunk['generated'] == 1]

            chunk["text"] = chunk["text"].astype(str).apply(clean_text)

            # Check if the cleaned text contains relevant keywords
            for text, label in zip(chunk["text"], chunk["generated"]):
                if contains_relevant_keywords(text):
                    writer.writerow([text, label])

    print(f"Processed sentences from batch {start_batch} onwards appended to {output_csv}")

if __name__ == "__main__":
    input_csv = "AI_Human.csv"  # Path to your input CSV file
    output_csv = "output_data.csv"  # Path to your output CSV file
    start_batch = 0  # Start from batch 0

    process_csv(input_csv, output_csv, batch_size=1000, start_batch=start_batch)
