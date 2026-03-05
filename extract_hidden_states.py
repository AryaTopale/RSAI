import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

MODEL_NAME = "Qwen/Qwen2.5-1.5B"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    output_hidden_states=True
)

model.eval()

df = pd.read_csv("train.csv")

hidden_vectors = []
labels = []

for _, row in tqdm(df.iterrows(), total=len(df)):

    query = row["Query"]
    tool = row["Tool"]

    inputs = tokenizer(query, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # mean pool representation
    hidden = outputs.hidden_states[-1].mean(dim=1)

    hidden_vectors.append(hidden.cpu().numpy()[0])
    labels.append(tool)

X = np.array(hidden_vectors)
y = np.array(labels)

np.save("X_train.npy", X)
np.save("y_train.npy", y)

print("Saved probe training features")