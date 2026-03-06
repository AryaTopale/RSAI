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

TRAIN_PATH = "train_results.csv"
EVAL_PATH = "eval_results.csv"

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
# Load datasets
# -------------------------

train_df = pd.read_csv(TRAIN_PATH)
eval_df = pd.read_csv(EVAL_PATH)

train_queries = train_df["query"].tolist()
train_labels = train_df["pred_tool"].tolist()

eval_queries = eval_df["query"].tolist()
eval_labels = eval_df["pred_tool"].tolist()


# encode labels using TRAIN set
le = LabelEncoder()
y_train = le.fit_transform(train_labels)
y_eval = le.transform(eval_labels)


# -------------------------
# Function: extract hidden states
# -------------------------

def extract_hidden_states(queries):

    layer_states = None

    for query in tqdm(queries):

        inputs = tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states

        if layer_states is None:

            n_layers = len(hidden_states)
            hidden_dim = hidden_states[0].shape[-1]

            layer_states = [[] for _ in range(n_layers)]

        for i in range(n_layers):

            h = hidden_states[i].mean(dim=1).float().cpu().numpy()[0]
            layer_states[i].append(h)

    layer_states = [np.array(l) for l in layer_states]

    return layer_states


# -------------------------
# Extract hidden states
# -------------------------

print("Extracting TRAIN hidden states...")
train_layer_states = extract_hidden_states(train_queries)

print("Extracting EVAL hidden states...")
eval_layer_states = extract_hidden_states(eval_queries)


hidden_dim = train_layer_states[0].shape[1]
num_classes = len(le.classes_)


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
# Train probes + Eval
# -------------------------

layer_acc = []

print("Training probes and evaluating on eval set...")

for layer in range(len(train_layer_states)):

    X_train = torch.tensor(
        train_layer_states[layer],
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(y_train)

    X_eval = torch.tensor(
        eval_layer_states[layer],
        dtype=torch.float32
    )

    probe = Probe(hidden_dim, num_classes)

    optimizer = optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(20):

        logits = probe(X_train)

        loss = criterion(logits, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # -------------------------
    # Evaluate probe
    # -------------------------

    with torch.no_grad():

        logits = probe(X_eval)

        preds = torch.argmax(logits, dim=1)

        acc = accuracy_score(
            y_eval,
            preds.numpy()
        )

    layer_acc.append(acc)

    print(f"Layer {layer} eval accuracy: {acc:.4f}")


# -------------------------
# Save results
# -------------------------

results = pd.DataFrame({
    "layer": list(range(len(layer_acc))),
    "eval_accuracy": layer_acc
})

results.to_csv(
    "layer_probe_accuracy_eval.csv",
    index=False
)

print("Saved: layer_probe_accuracy_eval.csv")


# -------------------------
# Plot
# -------------------------

plt.figure(figsize=(8,5))

plt.plot(layer_acc, marker="o")

plt.xlabel("Layer")
plt.ylabel("Eval Accuracy")
plt.title("Layer-wise Tool Prediction Probe Accuracy (Eval Set)")

plt.grid(True)

plt.savefig("layer_probe_plot_eval.png")

print("Saved: layer_probe_plot_eval.png")