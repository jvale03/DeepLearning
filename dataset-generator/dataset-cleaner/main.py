import pandas as pd

# Load the dataset
file_path = "datasets/initial_dataset.csv"
df = pd.read_csv(file_path)

# Drop 'title' and 'ai_generated' columns
df_cleaned = df.drop(columns=['title', 'ai_generated'])

# Rename columns
df_cleaned = df_cleaned.rename(columns={'abstract': 'Text', 'is_ai_generated': 'Label'})

# Ensure that each entry in "Text" is a single line by replacing newlines and excessive spaces
df_cleaned["Text"] = df_cleaned["Text"].str.replace(r"\s+", " ", regex=True).str.strip()

# Save the cleaned dataset to a new CSV file
df_cleaned.to_csv("datasets/final_dataset.csv", index=False, quoting=1, encoding="utf-8")  # quoting=1 ensures quotes around text fields

print("final_dataset.csv has been saved successfully!")
