import pandas as pd
import gc

new_df = pd.DataFrame(columns=["text","label"])


parquet_files = [
    "../datasets/train-00000-of-00007-bc5952582e004d67.parquet",
    "../datasets/train-00001-of-00007-71c80017bc45f30d.parquet",
    "../datasets/train-00002-of-00007-ee2d43f396e78fbc.parquet",
    "../datasets/train-00003-of-00007-529931154b42b51d.parquet",
    "../datasets/train-00004-of-00007-b269dc49374a2c0b.parquet",
    "../datasets/train-00005-of-00007-3dce5e05ddbad789.parquet",
    "../datasets/train-00006-of-00007-3d8a471ba0cf1c8d.parquet"
]

category_mapping = {"human": 0, "ai": 1, "student": 0}

for file in parquet_files:
    df = pd.read_parquet(file)

    df["label"] = df["source"].map(category_mapping)
    df = df[["text","label"]]

    new_df = pd.concat([new_df, df],ignore_index=True)

print(new_df.columns)
print("Parquets check!")

csv_files = [
    "../datasets/data_set.csv",
    "../datasets/LLM.csv",
    "../datasets/AI_Human.csv"
]

# csv_files[0]
data_sets = pd.read_csv(csv_files[0])
data_sets["label"] = data_sets["is_ai_generated"]
data_sets["text"] = data_sets["abstract"]
data_sets = data_sets[["text","label"]]
new_df = pd.concat([new_df,data_sets],ignore_index=True)

print(new_df.columns)
print("First csv check!")

# csv_files[1]
LLMs = pd.read_csv(csv_files[1])
LLMs["text"] = LLMs["Text"]
LLMs["label"] = LLMs["Label"].map(category_mapping)
LLMs = LLMs[["text","label"]]
new_df = pd.concat([new_df,LLMs],ignore_index=True)

print(new_df.columns)
print("Second csv check!")

# csv_files[2]
AI_Human = pd.read_csv(csv_files[2])
AI_Human["text"] = AI_Human["text"]
AI_Human["label"] = AI_Human["generated"]
AI_Human = AI_Human[["text","label"]]
new_df = pd.concat([new_df,AI_Human],ignore_index=True)

print(new_df.columns)
print("Second csv check!")

new_df.to_parquet("../datasets/merged_1.parquet")

del new_df
gc.collect()

# large file
model_training_dataset = pd.read_csv("../datasets/model_training_dataset.csv")
df_human = model_training_dataset[["human_text"]].rename(columns={"human_text": "text"})
df_human["label"] = 0
df_ai = model_training_dataset[["ai_text"]].rename(columns={"ai_text": "text"})
df_ai["label"] = 1
model_training_dataset = pd.concat([df_human,df_ai],ignore_index=True)
model_training_dataset.to_parquet("../datasets/merged_2.parquet")


print(model_training_dataset.columns)
print("Third csv check!")

