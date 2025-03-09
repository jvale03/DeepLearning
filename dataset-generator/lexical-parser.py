import pandas as pd
import spacy
from tqdm import tqdm
import numpy as np

def extract_science_terms(text):
    doc = nlp(str(text))
    
    # Be more selective about entity types
    science_entity_types = ["CHEMICAL", "GENE", "DISEASE", "SPECIES", "PROCEDURE", "TECHNOLOGY"]
    terms = [ent.text for ent in doc.ents if ent.label_ in science_entity_types]
    
    # If no scientific entities found, check for scientific keywords
    if not terms:
        technical_terms = [token.text for token in doc if token.pos_ == "NOUN" and token.is_alpha 
                          and len(token.text) > 3 and not token.is_stop]
        
        if technical_terms:
            if len(technical_terms) >= 2:
                terms = technical_terms
    
    return ", ".join(terms) if terms else None

# Function to calculate the scientific confidence score based on text similarity
def is_scientific_text(text):
    doc = nlp(str(text))
    
    # Reference scientific concepts
    science_concepts = ["research", "experiment", "theory", "science", "physics", "chemistry", "biology"]
    science_docs = [nlp(concept) for concept in science_concepts]
    
    # Calculate similarity to scientific concepts
    similarities = [doc.similarity(science_doc) for science_doc in science_docs]
    avg_similarity = np.mean(similarities)
    
    return avg_similarity > 0.6  # Return True if similarity score is above threshold


def process_dataset(input_file, output_file):
    
    # Load the dataset
    df = pd.read_csv(input_file)
    tqdm.pandas(desc="Processing Text")

    # Apply the first filtering approach
    df["science_terms"] = df["text"].progress_apply(extract_science_terms)

    # Filter rows based on both approaches
    df_filtered = df[df["science_terms"].notna()]

    df_filtered.to_csv(output_file, index=False)
    print(f"Filtered dataset saved to {output_file}")


if __name__ == "__main__":
    nlp = spacy.load("en_core_sci_lg")
    
    input_file = "new.csv"
    output_file = "final.csv"
    
    process_dataset(input_file, output_file)
