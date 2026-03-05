import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "exp_dataset.csv"

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("Total samples:", len(df))
print("\nTool distribution:")
print(df["Tool"].value_counts())

# -----------------------------
# TRAIN / TEMP SPLIT (80/20)
# -----------------------------
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["Tool"],
    random_state=42
)

# -----------------------------
# EVAL / TEST SPLIT (10/10)
# -----------------------------
eval_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["Tool"],
    random_state=42
)

# -----------------------------
# SAVE
# -----------------------------
train_df.to_csv("train.csv", index=False)
eval_df.to_csv("eval.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("\nSaved files:")
print("train.csv:", len(train_df))
print("eval.csv:", len(eval_df))
print("test.csv:", len(test_df))

print("\nTrain distribution:")
print(train_df["Tool"].value_counts())