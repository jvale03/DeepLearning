import re
import numpy as np
from transformers import BertTokenizerFast
from recurrent_neural_net import RecurrentNeuralNetwork

'''
Used to test professors' inputs and outputs automatically
To use this script, the files dataset1_inputs.csv and dataset1_outputs.csv must be in the datasets folder
'''

# Carrega o tokenizer customizado
tokenizer = BertTokenizerFast.from_pretrained("src/data_processor/custom_tokenizer", local_files_only=True)

# Função para limpeza do texto
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Função para processar textos
def process_texts(texts):
    tokenized = tokenizer(
        texts, padding="max_length", truncation=True, max_length=64
    )["input_ids"]
    return tokenized

# Load the trained RNN model
try:
    rnn = RecurrentNeuralNetwork.load("models/rnn_model.pkl")
    print("RNN model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please train and save the model first.")
    exit()

# Classe mock para compatibilidade com o modelo
class MockDataset:
    def __init__(self, X):
        self.X = X

# Load the input and output files
input_file = "datasets/dataset1_inputs.csv"
output_file = "datasets/dataset1_outputs.csv"

# Read input data
input_data = {}
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            # Assuming format is "ID [tab] Text"
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                doc_id, text = parts
                input_data[doc_id] = text

# Read output data (expected classifications)
expected_outputs = {}
with open(output_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            # Assuming format is "ID Human/AI"
            parts = line.strip().split()
            if len(parts) == 2:
                doc_id, label = parts
                expected_outputs[doc_id] = 1.0 if label.upper() == "HUMAN" else 0.0

# Processing and evaluating each entry
correct_predictions = 0
total_predictions = 0

for doc_id, text in input_data.items():
    # Skipping the first line (header)
    if doc_id == "ID":
        continue

    if doc_id in expected_outputs:
        # Clean and process the text
        text_clean = clean_text(text)
        text_processed = process_texts([text_clean])
        
        # Create dataset and predict
        mock_dataset = MockDataset(np.array(text_processed))
        prediction = rnn.predict(mock_dataset)
        predicted_label = "Human" if prediction >= 0.5 else "AI"
        expected_label = "Human" if expected_outputs[doc_id] >= 0.5 else "AI"
        
        # Compare prediction with expected output
        is_correct = (predicted_label == expected_label)
        if is_correct:
            correct_predictions += 1
        total_predictions += 1
        
        # Print results for each document
        print(f"Document {doc_id}:")
        print(f"  Predicted: {predicted_label} (Confidence: {prediction[0][0]:.4f})")
        print(f"  Expected: {expected_label}")
        print(f"  Correct: {is_correct}")
        print("-" * 50)

# Calculate and print overall accuracy
if total_predictions > 0:
    accuracy = correct_predictions / total_predictions * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
else:
    print("No predictions were made. Check your input files.")

# Get prediction distribuition - how many AI and Human predictions were made
ai_predictions = sum(1 for label in expected_outputs.values() if label < 0.5)
human_predictions = total_predictions - ai_predictions

print(f"The model predicted AI {ai_predictions} times")
print(f"The model predicted Human {human_predictions} times")