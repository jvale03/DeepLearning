import spacy
import PyPDF2
import re
import os

# Carregar modelo de NLP do spaCy para inglês
nlp = spacy.load("en_core_web_md")

# Definição das palavras-chave para cada área
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

# Função para ler e extrair texto do PDF
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + ' '
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Função para limpar o texto
def clean_text(text):
    text = re.sub(r'\([0-9]+(\.[0-9]+)?\)', '', text)
    text = re.sub(r'\b[0-9]+(\.[0-9]+)?\b', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[-–—]\s+', '', text)
    return text

# Função para extrair frases contendo palavras-chave
def extract_relevant_sentences(text):
    doc = nlp(text)
    relevant_sentences = []
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        for category, keywords in KEYWORDS.items():
            if any(keyword in sentence_text.lower() for keyword in keywords):
                relevant_sentences.append(sentence_text)
                break
    return relevant_sentences

# Função para salvar frases em um arquivo
def write_sentences_to_txt(sentences, output_txt):
    with open(output_txt, mode='w', encoding='utf-8') as file:
        for sentence in sentences:
            file.write(sentence + '\n')

# Função principal para processar todos os PDFs
def process_all_pdfs(folder_path, output_txt):
    all_sentences = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processando: {filename}")
            text = read_pdf(pdf_path)
            cleaned_text = clean_text(text)
            sentences = extract_relevant_sentences(cleaned_text)
            all_sentences.extend(sentences)
    
    write_sentences_to_txt(all_sentences, output_txt)
    print(f"Frases relevantes de todos os PDFs salvas em {output_txt}")

if __name__ == '__main__':
    pdf_folder = "scraper/allPapers"  # Pasta contendo os PDFs
    output_file = "all_sentences.txt"  # Arquivo de saída
    process_all_pdfs(pdf_folder, output_file)
