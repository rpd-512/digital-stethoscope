import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# pip3 install numpy scipy matplotlib scikit-learn
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


# ===================== CONFIG =====================

DATA_ROOT = "../../Spectrogram/DatasetGenerated/NewData/"

CLASS_MAP = {
    "Not_Heartbeat": 0,
    "Heartbeat": 1
}

FNAME_PREFIX = "HB"

BATCH_SIZE = 16
EPOCHS = 10
VAL_SPLIT = 0.2
SEED = 137

POS_WEIGHT = 1.0
NEG_WEIGHT = 1.23

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------ scan dataset ------------------

samples = []  # (spec_path, env_path, label)

spec_re = re.compile(rf"{FNAME_PREFIX}_spec_(\d+)\.npy")
env_re  = re.compile(rf"{FNAME_PREFIX}_env_(\d+)\.npy")

for class_name, label in CLASS_MAP.items():
    class_dir = os.path.join(DATA_ROOT, class_name)
    if not os.path.isdir(class_dir):
        continue

    spec_files = {}
    env_files = {}

    for f in os.listdir(class_dir):
        m = spec_re.match(f)
        if m:
            spec_files[m.group(1)] = f
        m = env_re.match(f)
        if m:
            env_files[m.group(1)] = f

    common_ids = sorted(set(spec_files) & set(env_files))

    for idx in common_ids:
        samples.append((
            os.path.join(class_dir, spec_files[idx]),
            os.path.join(class_dir, env_files[idx]),
            label
        ))

if not samples:
    raise RuntimeError("No valid samples found")

print(f"Total samples: {len(samples)}")

# ------------------ shuffle + split ------------------

np.random.shuffle(samples)
split = int(len(samples) * (1 - VAL_SPLIT))

train_samples = samples[:split]
val_samples   = samples[split:]

# ------------------ infer shapes ------------------

spec0 = np.load(train_samples[0][0])
env0  = np.load(train_samples[0][1])

F, T = spec0.shape
ENV_T = env0.shape[0]

print("Spectrogram shape:", spec0.shape)
print("Envelope length:", env0.shape)

# ===================== DATASET =====================

class HeartbeatDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        spec_path, env_path, label = self.samples[idx]

        spec = np.load(spec_path).astype(np.float32)
        env  = np.load(env_path).astype(np.float32)

        # shapes:
        # spec -> (1, F, T)
        # env  -> (T, 1)
        spec = torch.from_numpy(spec).unsqueeze(0)
        env  = torch.from_numpy(env).unsqueeze(-1)

        return spec, env, torch.tensor(label, dtype=torch.float32)

train_loader = DataLoader(
    HeartbeatDataset(train_samples),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    HeartbeatDataset(val_samples),
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False
)

# ===================== MODEL =====================

class CNNEnvFusion(nn.Module):
    def __init__(self):
        super().__init__()

        # -------- Spectrogram CNN --------
        self.spec_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.spec_fc = nn.Linear(64, 64)

        # -------- Envelope Conv1D --------
        self.env_net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.env_fc = nn.Linear(64, 64)

        # -------- Fusion --------
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 1)
        )

    def forward(self, spec, env):
        # spec: (B, 1, F, T)
        x = self.spec_net(spec)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.spec_fc(x))

        # env: (B, T, 1) → (B, 1, T)
        env = env.permute(0, 2, 1)
        y = self.env_net(env)
        y = y.view(y.size(0), -1)
        y = torch.relu(self.env_fc(y))

        z = torch.cat([x, y], dim=1)
        return torch.sigmoid(self.classifier(z)).squeeze(1)

model = CNNEnvFusion().to(DEVICE)

# ===================== LOSS =====================

def weighted_bce_loss(y_pred, y_true):
    eps = 1e-7
    y_pred = torch.clamp(y_pred, eps, 1 - eps)
    loss = (
        y_true * POS_WEIGHT * torch.log(y_pred) +
        (1 - y_true) * NEG_WEIGHT * torch.log(1 - y_pred)
    )
    return -loss.mean()

optimizer = optim.AdamW(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-4
)

# ===================== TRAIN =====================

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for spec, env, labels in train_loader:
        spec, env, labels = spec.to(DEVICE), env.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        preds = model(spec, env)
        loss = weighted_bce_loss(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for spec, env, labels in val_loader:
            spec, env, labels = spec.to(DEVICE), env.to(DEVICE), labels.to(DEVICE)
            preds = model(spec, env)
            predicted = (preds >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.numel()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {correct/total:.4f}")

# ------------------ save ------------------

torch.save(model.state_dict(), "model_006.pt")
print("Model saved.")

# ===================== EVALUATION =====================

THRESHOLD = 0.65

y_true = []
y_pred = []
probs  = []

model.eval()
with torch.no_grad():
    for spec, env, labels in val_loader:
        spec, env = spec.to(DEVICE), env.to(DEVICE)
        p = model(spec, env).cpu().numpy()

        probs.extend(p.tolist())
        y_true.extend(labels.numpy().tolist())
        y_pred.extend((p >= THRESHOLD).astype(int).tolist())

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix (threshold =", THRESHOLD, "):")
print(cm)

print(f"""
TN (Correct Not_Heartbeat): {tn}
FP (False Heartbeat):       {fp}
FN (Missed Heartbeat):      {fn}
TP (Correct Heartbeat):     {tp}
""")

print("Classification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Not_Heartbeat", "Heartbeat"]
))

# ------------------ Probability histogram ------------------

plt.hist(probs, bins=50)
plt.axvline(THRESHOLD, color="r")
plt.title("Prediction Confidence Distribution")
plt.xlabel("Heartbeat probability")
plt.ylabel("Count")
plt.show()
