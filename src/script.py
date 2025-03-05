import pandas as pd

# df1 = pd.read_parquet("../datasets/merged_1.parquet")
#df2 = pd.read_parquet("../datasets/merged_2.parquet")

# print(df1.head())
#print(df2.head())

df = pd.read_csv("../datasets/model_training_dataset.csv")
print(df.columns)
