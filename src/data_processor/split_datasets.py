import pandas as pd
from sklearn.model_selection import train_test_split

parquet_files = [
    "../datasets/merged_1.parquet",
    "../datasets/merged_2.parquet"
]

for file in parquet_files:
    df = pd.read_parquet(file)

    train_df, test_df = train_test_split(df, test_size=0.25, shuffle=True, random_state=42)

    train_file = file.replace(".parquet", "_train.parquet")
    test_file = file.replace(".parquet", "_test.parquet")

    train_df.to_parquet(train_file, index=False)
    test_df.to_parquet(test_file, index=False)