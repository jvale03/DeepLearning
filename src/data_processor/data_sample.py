import pandas as pd

# Carregar datasets
train_1 = pd.read_parquet("../../datasets/train/merged_1_train.parquet")
train_2 = pd.read_parquet("../../datasets/train/merged_2_train.parquet")
test_1 = pd.read_parquet("../../datasets/test/merged_1_test.parquet")
test_2 = pd.read_parquet("../../datasets/test/merged_2_test.parquet")

# Definir a coluna-alvo (ajusta conforme o teu dataset)
target_column = "label"  # Substitui pelo nome correto da tua variável alvo

# Amostragem estratificada para treino
train_1 = train_1.groupby(target_column).sample(n=50_000, random_state=42)
train_2 = train_2.groupby(target_column).sample(n=50_000, random_state=42)


# Concatenar e salvar
train = pd.concat([train_1, train_2], ignore_index=True)
train.to_parquet("../../datasets/train/train_sample.parquet")

# Amostragem estratificada para teste
test_1 = test_1.groupby(target_column).sample(n=5_000, random_state=42)
test_2 = test_2.groupby(target_column).sample(n=5_000, random_state=42)

# Concatenar e salvar
test = pd.concat([test_1, test_2], ignore_index=True)
test.to_parquet("../../datasets/test/test_sample.parquet")