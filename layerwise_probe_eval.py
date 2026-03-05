import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DATA_PATH = "eval_results.csv"
PROBE_PATH = "tool_probe.pt"

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
labels = df["pred_tool"].tolist()

le = LabelEncoder()
y = le.fit_transform(labels)


# -------------------------
# Extract hidden states
# -------------------------

print("Extracting hidden states...")

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

        # mean pooling across tokens
        h = hidden_states[i].mean(dim=1).squeeze(0).float().cpu().numpy()

        layer_states[i].append(h)


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
# Load probe checkpoint
# -------------------------

print("Loading saved probe...")

checkpoint = torch.load(
    PROBE_PATH,
    map_location="cpu",
    weights_only=False
)

# Handle multiple checkpoint formats

if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
elif "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

# Infer dimensions automatically

first_weight = list(state_dict.values())[0]
input_dim = first_weight.shape[1]
num_classes = first_weight.shape[0]

probe = Probe(input_dim, num_classes)
probe.load_state_dict(state_dict)

probe.to(device)
probe.eval()


# -------------------------
# Evaluate layer-wise
# -------------------------

layer_acc = []

print("Evaluating probes on eval set...")

for layer in range(len(layer_states)):

    X = torch.tensor(
        layer_states[layer],
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():

        logits = probe(X)
        preds = torch.argmax(logits, dim=1)

    acc = accuracy_score(
        y,
        preds.cpu().numpy()
    )

    layer_acc.append(acc)

    print(f"Layer {layer} accuracy: {acc:.4f}")


# -------------------------
# Save results
# -------------------------

results = pd.DataFrame({
    "layer": list(range(len(layer_acc))),
    "accuracy": layer_acc
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
plt.ylabel("Probe Accuracy")
plt.title("Layer-wise Tool Prediction Probe Accuracy (Eval Set)")

plt.grid(True)

plt.savefig("layer_probe_plot_eval.png")

print("Saved: layer_probe_plot_eval.png")