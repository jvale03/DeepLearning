import pandas as pd

# Load the dataset with semicolon delimiter
file_path = "datasets/initial_validation_dataset.csv"
df = pd.read_csv(file_path, delimiter=";")  # Specify semicolon as the delimiter

# Ensure that each entry in "Text" is a single line by replacing newlines and excessive spaces
df["Text"] = df["Text"].str.replace(r"\s+", " ", regex=True).str.strip()

# Save the cleaned dataset to a new CSV file with semicolon delimiter
df.to_csv("datasets/validation_dataset.csv", index=False, sep=";", quoting=1, encoding="utf-8")  # quoting=1 ensures quotes around text fields

print("a.csv has been saved successfully!")
