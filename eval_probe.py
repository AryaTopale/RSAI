import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("Loading probe...")

checkpoint = torch.load("tool_probe.pt")

classes = checkpoint["label_encoder"]
num_classes = len(classes)

# load features
X = np.load("X_probe.npy")
y = np.load("y_probe.npy")

le = LabelEncoder()
le.fit(classes)
y_encoded = le.transform(y)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y_encoded)

input_dim = X.shape[1]

class ToolProbe(nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

probe = ToolProbe(input_dim, num_classes)
probe.load_state_dict(checkpoint["model_state_dict"])
probe.eval()

with torch.no_grad():

    logits = probe(X)
    preds = torch.argmax(logits, dim=1)

acc = (preds == y).float().mean()

print("Probe accuracy:", acc.item())