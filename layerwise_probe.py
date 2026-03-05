import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DATA_PATH = "train_results.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load model
# -------------------------

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    output_hidden_states=True
)

model.eval()

# -------------------------
# Load dataset
# -------------------------

df = pd.read_csv(DATA_PATH)

queries = df["query"].tolist()

# tools predicted by model earlier
labels = df["pred_tool"].tolist()

# encode tools
le = LabelEncoder()
y = le.fit_transform(labels)

# -------------------------
# Collect hidden states
# -------------------------

print("Extracting hidden states...")

layer_states = None

for query in tqdm(queries):

    inputs = tokenizer(query, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states

    # number of layers
    if layer_states is None:
        n_layers = len(hidden_states)
        hidden_dim = hidden_states[0].shape[-1]
        layer_states = [[] for _ in range(n_layers)]

    for i in range(n_layers):

        h = hidden_states[i].mean(dim=1).cpu().numpy()[0]
        layer_states[i].append(h)

# convert to numpy
layer_states = [np.array(l) for l in layer_states]

# -------------------------
# Probe model
# -------------------------

class Probe(nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

# -------------------------
# Train probe per layer
# -------------------------

layer_acc = []

print("Training probes...")

for layer in range(len(layer_states)):

    X = layer_states[layer]

    X = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y)

    probe = Probe(hidden_dim, len(le.classes_))

    optimizer = optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # train
    for epoch in range(20):

        logits = probe(X)

        loss = criterion(logits, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():

        logits = probe(X)
        preds = torch.argmax(logits, dim=1)

        acc = accuracy_score(y, preds.numpy())

    layer_acc.append(acc)

    print(f"Layer {layer} accuracy: {acc:.4f}")

# -------------------------
# Save results
# -------------------------

results = pd.DataFrame({
    "layer": list(range(len(layer_acc))),
    "accuracy": layer_acc
})

results.to_csv("layer_probe_accuracy.csv", index=False)

print("Saved: layer_probe_accuracy.csv")

# -------------------------
# Plot
# -------------------------

plt.figure(figsize=(8,5))

plt.plot(layer_acc, marker="o")

plt.xlabel("Layer")
plt.ylabel("Probe Accuracy")
plt.title("Layer-wise Tool Prediction Probe Accuracy")

plt.grid(True)

plt.savefig("layer_probe_plot.png")

print("Saved: layer_probe_plot.png")