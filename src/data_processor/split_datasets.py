import pandas as pd
from sklearn.model_selection import train_test_split

file= "../../datasets/new_data/Large_Physics_and_Science_Dataset_Processed.csv"
df = pd.read_csv(file)

train_df, test_df = train_test_split(df, test_size=0.25, shuffle=True, random_state=42)

train_file = file.replace(".csv", "_train.parquet")
test_file = file.replace(".csv", "_test.parquet")

train_df.to_parquet(train_file, index=False)
test_df.to_parquet(test_file, index=False)