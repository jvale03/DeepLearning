import pandas as pd

df1 = pd.read_parquet("../datasets/train/merged_1_train.parquet")
df2 = pd.read_parquet("../datasets/train/merged_2_train.parquet")

print(df1.shape)
print(df2.shape)

count1 = df1["label"].value_counts()
count2 = df2["label"].value_counts()

print(count1)
print(count2)

