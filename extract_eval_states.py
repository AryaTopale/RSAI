import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_NAME = "Qwen/Qwen2.5-1.5B"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    output_hidden_states=True
)

model.eval()

df = pd.read_csv("eval_results.csv")

queries = df["query"].tolist()
labels = df["pred_tool"].tolist()

hidden_states = []

print("Extracting eval hidden states...")

for q in tqdm(queries):

    inputs = tokenizer(q, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    h = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()[0]

    hidden_states.append(h)

X = np.array(hidden_states)
y = np.array(labels)

np.save("X_eval.npy", X)
np.save("y_eval.npy", y)

print("Saved eval probe dataset")