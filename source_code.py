import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# -----------------------------
# 1. Configuration & Setup
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")

# Define paths (Adjust these if your data is in a different folder)
BASE_DIR = "da5401-2025-data-challenge"
# If running locally or in Colab, you might need: BASE_DIR = "./data"
TRAIN_PATH = os.path.join(BASE_DIR, "train_data.json")
TEST_PATH = os.path.join(BASE_DIR, "test_data.json")
METRIC_PATH = os.path.join(BASE_DIR, "metric_names.json")

# -----------------------------
# 2. Data Loading & Embedding Generation
# -----------------------------

embedding_dim = 1024  # Dimension for e5-large

def load_and_embed():
    # Load JSON
    print("Loading JSON data...")
    with open(TRAIN_PATH, "r", encoding="utf8") as f:
        train_raw = json.load(f)
    with open(TEST_PATH, "r", encoding="utf8") as f:
        test_raw = json.load(f)

    train_df = pd.DataFrame(train_raw)
    test_df = pd.DataFrame(test_raw)

    # --- Helper to combine text fields ---
    def combine_text(row):
        sp = str(row.get("system_prompt", "")) if row.get("system_prompt") else ""
        up = str(row.get("user_prompt", ""))
        rp = str(row.get("response", ""))
        return f"{sp} [SYS] {up} [USR] {rp} [RES]"

    train_df["full_text"] = train_df.apply(combine_text, axis=1)
    test_df["full_text"] = test_df.apply(combine_text, axis=1)

    # --- Embed Metrics ---
    # Metrics are repeated, so embed unique ones first to save time
    print("Embedding Metrics...")
    unique_metrics = list(set(train_df["metric_name"].astype(str).unique()) |
                          set(test_df["metric_name"].astype(str).unique()))

    metric_emb_map = {}
    # Batch encode unique metrics
    unique_embs = embedding_model.encode(unique_metrics, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
    for name, emb in zip(unique_metrics, unique_embs):
        metric_emb_map[name] = emb

    # Map back to dataframes
    train_metric_embs = np.vstack([metric_emb_map[n] for n in train_df["metric_name"].astype(str)])
    test_metric_embs = np.vstack([metric_emb_map[n] for n in test_df["metric_name"].astype(str)])

    # --- Embed Text ---
    print("Embedding Training Text...")
    train_text_embs = embedding_model.encode(train_df["full_text"].tolist(), batch_size=16, convert_to_numpy=True, show_progress_bar=True)

    print("Embedding Test Text...")
    test_text_embs = embedding_model.encode(test_df["full_text"].tolist(), batch_size=16, convert_to_numpy=True, show_progress_bar=True)

    # Save to disk
    np.save("train_metric_embs.npy", train_metric_embs)
    np.save("test_metric_embs.npy", test_metric_embs)
    np.save("train_text_embs.npy", train_text_embs)
    np.save("test_text_embs.npy", test_text_embs)

    # Get targets
    y_train = train_df["score"].values.astype(np.float32)

    return train_metric_embs, train_text_embs, y_train, test_metric_embs, test_text_embs

# Check if embeddings exist to skip computation
if os.path.exists("/content/drive/MyDrive/da5401-2025-data-challenge/train_text_embs.npy"):
    print("Loading pre-computed embeddings...")
    train_metric_embs = np.load("/content/drive/MyDrive/da5401-2025-data-challenge/train_metric_embs.npy")
    test_metric_embs = np.load("/content/drive/MyDrive/da5401-2025-data-challenge/test_metric_embs.npy")
    train_text_embs = np.load("/content/drive/MyDrive/da5401-2025-data-challenge/train_text_embs.npy")
    test_text_embs = np.load("/content/drive/MyDrive/da5401-2025-data-challenge/test_text_embs.npy")

    # Need y_train from dataframe
    with open(TRAIN_PATH, "r", encoding="utf8") as f:
        y_train = pd.DataFrame(json.load(f))["score"].values.astype(np.float32)
else:
    train_metric_embs, train_text_embs, y_train, test_metric_embs, test_text_embs = load_and_embed()

# -----------------------------
# 3. Data Augmentation
# -----------------------------
print("Performing Data Augmentation...")
rng = np.random.default_rng(42)
N = len(train_metric_embs)

# A. Shuffle Negatives: Random text with Real metric
perm = rng.permutation(N)
neg_metric_1 = train_metric_embs
neg_text_1   = train_text_embs[perm]
neg_y_1      = rng.integers(0, 4, size=N).astype(np.float32) # Labels 0-3

# B. Noise Negatives: Real text + Gaussian Noise
noise = rng.normal(scale=0.6, size=train_text_embs.shape).astype(np.float32)
neg_metric_2 = train_metric_embs
neg_text_2   = train_text_embs + noise
neg_y_2      = rng.integers(0, 4, size=N).astype(np.float32)

# C. Metric Swap Negatives: Real text + Random metric
perm2 = rng.permutation(N)
neg_metric_3 = train_metric_embs[perm2]
neg_text_3   = train_text_embs
neg_y_3      = rng.integers(0, 4, size=N).astype(np.float32)

# Combine everything
m_all = np.vstack([train_metric_embs, neg_metric_1, neg_metric_2, neg_metric_3])
t_all = np.vstack([train_text_embs,   neg_text_1,   neg_text_2,   neg_text_3])
y_all = np.concatenate([y_train,      neg_y_1,      neg_y_2,      neg_y_3])

print(f"Augmented Training Data Shape: {m_all.shape[0]} samples")

# -----------------------------
# 4. Feature Engineering
# -----------------------------
def build_features(metric_emb, text_emb):
    # 1. Absolute Difference
    abs_diff = np.abs(metric_emb - text_emb)

    # 2. Element-wise Product
    prod = metric_emb * text_emb

    # 3. Cosine Similarity
    # Normalize first
    m_norm = metric_emb / (np.linalg.norm(metric_emb, axis=1, keepdims=True) + 1e-9)
    t_norm = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-9)
    cosine = np.sum(m_norm * t_norm, axis=1, keepdims=True)

    # Concatenate: [Metric, Text, |M-T|, M*T, Cosine]
    # Dimensions: 1024 + 1024 + 1024 + 1024 + 1 = 4097
    X = np.hstack([metric_emb, text_emb, abs_diff, prod, cosine])
    return X.astype(np.float32)

print("Building features for training...")
X_train = build_features(m_all, t_all)
print("Building features for testing...")
X_test  = build_features(test_metric_embs, test_text_embs)

print(f"Feature Vector Size: {X_train.shape[1]}")

# -----------------------------
# 5. Model Definition
# -----------------------------
class DeepMLP(nn.Module):
    def __init__(self, input_dim):
        super(DeepMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# -----------------------------
# 6. Training & Validation (5-Fold CV)
# -----------------------------
N_FOLDS = 5
EPOCHS = 1000
BATCH_SIZE = 256
LEARNING_RATE = 1e-5

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X_train))
test_preds_accum = np.zeros((N_FOLDS, len(X_test)))

print("\nStarting 5-Fold Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\n=== Fold {fold+1}/{N_FOLDS} ===")

    # Split Data
    X_tr, y_tr = X_train[train_idx], y_all[train_idx]
    X_val, y_val = X_train[val_idx], y_all[val_idx]

    # Loaders
    train_ds = TabularDataset(X_tr, y_tr)
    val_ds = TabularDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model
    model = DeepMLP(input_dim=X_train.shape[1]).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_rmse = float('inf')

    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        # Validation
        model.eval()
        val_preds_fold = []
        val_targets_fold = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                p = model(xb).cpu().numpy()
                val_preds_fold.append(p)
                val_targets_fold.append(yb.numpy())

        val_preds_fold = np.concatenate(val_preds_fold)
        val_targets_fold = np.concatenate(val_targets_fold)

        rmse = np.sqrt(mean_squared_error(val_targets_fold, val_preds_fold))

        # Save best
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), f"best_model_fold{fold}.pt")

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}: Train Loss {train_loss/len(X_tr):.4f}, Val RMSE {rmse:.4f}")

    print(f"Fold {fold+1} Best RMSE: {best_rmse:.4f}")

    # -----------------------------
    # Inference (OOF & Test)
    # -----------------------------
    # Load best model
    model.load_state_dict(torch.load(f"best_model_fold{fold}.pt"))
    model.eval()

    # Generate OOF for this fold
    with torch.no_grad():
        # Process validation in batches to save memory
        fold_oof = []
        for xb, _ in val_loader:
            xb = xb.to(DEVICE)
            fold_oof.append(model(xb).cpu().numpy())
        oof_preds[val_idx] = np.concatenate(fold_oof)

        # Generate Test Predictions
        test_fold_preds = []
        # Create a simple loader for test data
        test_loader = DataLoader(torch.tensor(X_test), batch_size=BATCH_SIZE, shuffle=False)
        for xb in test_loader:
            xb = xb.to(DEVICE)
            test_fold_preds.append(model(xb).cpu().numpy())
        test_preds_accum[fold] = np.concatenate(test_fold_preds)

# -----------------------------
# 7. Post-Processing (Calibration)
# -----------------------------
print("\nCalibrating predictions...")

# Train Linear Regression on OOF predictions vs True Labels
calibrator = LinearRegression()
calibrator.fit(oof_preds.reshape(-1, 1), y_all)

print(f"Calibration Coeff: {calibrator.coef_[0]:.4f}, Intercept: {calibrator.intercept_:.4f}")

# Average test predictions across folds
avg_test_preds = test_preds_accum.mean(axis=0)

# Apply Calibration
final_test_preds = calibrator.predict(avg_test_preds.reshape(-1, 1))

# Clip to valid range [0, 10]
final_test_preds = np.clip(final_test_preds, 0, 10)

# -----------------------------
# 8. Save Submission
# -----------------------------
# Ensure ID starts from 1
ids = np.arange(1, len(final_test_preds) + 1)

submission = pd.DataFrame({
    "ID": ids,
    "score": final_test_preds
})

submission.to_csv("da5401-2025-data-challenge/submission.csv", index=False)
print("Successfully saved 'submission.csv'")