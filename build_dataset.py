import pandas as pd
import json
import random

DATA_PATH = "dataset/data/all_clean_data.csv"
DES_PATH = "dataset/plugin_des.json"

NUM_TOOLS = 5
MIN_COUNT = 200
SEED = 42

random.seed(SEED)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("Total samples:", len(df))

# -----------------------------
# TOOL FREQUENCY
# -----------------------------
tool_counts = df["Tool"].value_counts()

# keep tools with >= 200 samples
valid_tools = tool_counts[tool_counts >= MIN_COUNT].index.tolist()

print("Tools with >=200 samples:", len(valid_tools))

# -----------------------------
# RANDOMLY SELECT 5 TOOLS
# -----------------------------
selected_tools = random.sample(valid_tools, NUM_TOOLS)

print("Selected tools:", selected_tools)

# -----------------------------
# FILTER DATASET
# -----------------------------
df_final = df[df["Tool"].isin(selected_tools)].copy()

print("\nFinal distribution:")
print(df_final["Tool"].value_counts())

# -----------------------------
# LOAD TOOL DESCRIPTIONS
# -----------------------------
with open(DES_PATH) as f:
    tool_descriptions = json.load(f)

# keep only selected tool descriptions
exp_descriptions = {tool: tool_descriptions[tool] for tool in selected_tools}

# -----------------------------
# SAVE DATASET
# -----------------------------
df_final.to_csv("exp_dataset.csv", index=False)

# -----------------------------
# SAVE TOOL DESCRIPTIONS
# -----------------------------
with open("exp_descriptions.json", "w") as f:
    json.dump(exp_descriptions, f, indent=4)

print("\nSaved files:")
print("exp_dataset.csv")
print("exp_descriptions.json")