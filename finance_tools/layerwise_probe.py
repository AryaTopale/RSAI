import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"
DATA_PATH = "/content/finance_train_results.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load model
# -------------------------

print(f"Loading model {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True,
    output_hidden_states=True,
)

model.eval()

# -------------------------
# Load and Clean dataset
# -------------------------

df = pd.read_csv(DATA_PATH)

# Filter out rows where the model failed to predict a tool
df = df.dropna(subset=["pred_tool"])
df = df[df["pred_tool"].str.strip() != ""]

queries = df["query"].tolist()
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
# Train/Test Split
# -------------------------

# Split indices so we can use them across all layers
train_idx, test_idx = train_test_split(
    np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
)

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
    X_layer = layer_states[layer]

    # Use split indices
    X_train = torch.tensor(X_layer[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx])
    X_test = torch.tensor(X_layer[test_idx], dtype=torch.float32)
    y_test = y[test_idx]

    probe = Probe(hidden_dim, len(le.classes_))

    optimizer = optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Increase epochs to 50 for finance domain
    for epoch in range(50):
        logits = probe(X_train)

        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits = probe(X_test)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        acc = accuracy_score(y_test, preds)

    layer_acc.append(acc)

    print(f"Layer {layer} Test accuracy: {acc:.4f}")

# -------------------------
# Save results
# -------------------------

results = pd.DataFrame({"layer": list(range(len(layer_acc))), "accuracy": layer_acc})

results.to_csv("layer_probe_accuracy.csv", index=False)

print("Saved: layer_probe_accuracy.csv")

# -------------------------
# Plot
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(layer_acc, marker="o")

plt.xlabel("Layer")
plt.ylabel("Probe Accuracy (Test Set)")
plt.title(f"Layer-wise Tool Prediction Probe Accuracy ({MODEL_NAME})")

plt.grid(True)

plt.savefig("layer_probe_plot.png")

print("Saved: layer_probe_plot.png")
