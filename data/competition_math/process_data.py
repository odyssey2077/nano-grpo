import pandas as pd

original_file = "data/competition_math/competition_math.parquet"

df = pd.read_parquet(original_file)
#check the value distribution of column level
print(df["level"].value_counts() / len(df))
print(df["type"].value_counts() / len(df))

# shuffle randomly and split into train and test
df = df.sample(frac=1).reset_index(drop=True)
train_df = df.iloc[:int(len(df) * 0.6)]
test_df = df.iloc[int(len(df) * 0.6):]
print(train_df["level"].value_counts() / len(train_df))
print(train_df["type"].value_counts() / len(train_df))
print(test_df["level"].value_counts() / len(test_df))
print(test_df["type"].value_counts() / len(test_df))

train_df.to_parquet("data/competition_math/train.parquet")
test_df.to_parquet("data/competition_math/test.parquet")