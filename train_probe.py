import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Load probe dataset
# -------------------------
print("Loading probe dataset...")

X = np.load("X_probe.npy")
y = np.load("y_probe.npy")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

num_classes = len(le.classes_)
input_dim = X.shape[1]

print("Samples:", X.shape[0])
print("Hidden dim:", input_dim)
print("Tools:", le.classes_)

# convert to torch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y_encoded, dtype=torch.long)

# -------------------------
# Probe model
# -------------------------

class ToolProbe(nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


probe = ToolProbe(input_dim, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(probe.parameters(), lr=1e-3)

# -------------------------
# Training
# -------------------------

epochs = 50
batch_size = 64

dataset = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("Training probe...")

for epoch in range(epochs):

    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:

        logits = probe(batch_x)

        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)

        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    acc = correct / total

    print(f"Epoch {epoch+1} | Loss {total_loss:.4f} | Acc {acc:.4f}")

# -------------------------
# Save probe
# -------------------------

torch.save({
    "model_state_dict": probe.state_dict(),
    "label_encoder": le.classes_
}, "tool_probe.pt")

print("Probe saved → tool_probe.pt")