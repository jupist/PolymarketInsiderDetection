"""
autoencoder.py
Dense autoencoder for unsupervised anomaly detection on trader behavioral profiles.

Architecture:
    Input(n) → FC(64) → ReLU → Dropout(0.2) → FC(32) → ReLU
    Bottleneck(16)
    FC(32) → ReLU → FC(64) → ReLU → Output(n)

Training: MSE reconstruction loss on CONTROL traders only.
Scoring:  Per-trader mean squared reconstruction error (higher = more anomalous).

Usage:
    python src/models/autoencoder.py

Input:  data/processed/trader_profiles.parquet
Output: data/processed/ae_scores.parquet
"""

import pathlib
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROCESSED_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"

FEATURE_COLS = [
    "log_total_volume",
    "log_max_bet",
    "log_trade_count",
    "avg_bet_usdc",
    "bet_size_ratio",
    "direction_bias",
    "unique_markets",
    "market_concentration_hhi",
    "time_to_news_min",
    "time_span_hours",
]

# Training hyperparameters
BATCH_SIZE = 256
EPOCHS = 150
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 15
VAL_SPLIT = 0.2
DROPOUT = 0.2
LATENT_DIM = 16
HIDDEN_DIMS = [64, 32]


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int, dropout: float):
        super().__init__()

        # Encoder
        enc_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror of encoder)
        dec_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)


def train(
    model: Autoencoder,
    X_train: np.ndarray,
    X_val: np.ndarray,
    device: torch.device,
) -> list[float]:
    """Train with early stopping. Returns validation loss history."""
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    model.to(device)
    best_val = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                val_loss += criterion(recon, batch).item() * len(batch)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)
        history.append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best val={best_val:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def score_all(
    model: Autoencoder,
    X_all: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Per-sample mean squared reconstruction error."""
    model.eval()
    X_t = torch.tensor(X_all, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon = model(X_t)
    mse = ((X_t - recon) ** 2).mean(dim=1).cpu().numpy()
    return mse


def run(profiles: Optional[pd.DataFrame] = None, save: bool = True) -> pd.DataFrame:
    """Full train-and-score pipeline."""
    if profiles is None:
        path = PROCESSED_DIR / "trader_profiles.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Profiles not found at {path}. Run profile_builder.py first.")
        profiles = pd.read_parquet(path)

    X = profiles[FEATURE_COLS].values.astype(np.float32)

    # Fit scaler on control traders only
    control_mask = profiles["is_leaked_market"] == 0
    scaler = StandardScaler()
    scaler.fit(X[control_mask])
    X_scaled = scaler.transform(X).astype(np.float32)

    # Split control traders into train / val
    X_control = X_scaled[control_mask]
    X_train, X_val = train_test_split(X_control, test_size=VAL_SPLIT, random_state=42)

    input_dim = X_control.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  input_dim={input_dim}")

    model = Autoencoder(input_dim, HIDDEN_DIMS, LATENT_DIM, DROPOUT)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Autoencoder parameters: {total_params:,}")
    print(f"Training on {X_train.shape[0]:,} control traders (val={X_val.shape[0]:,}) ...")

    history = train(model, X_train, X_val, device)

    print("\nScoring all traders ...")
    mse = score_all(model, X_scaled, device)

    result = profiles[["taker", "is_leaked_market", "case_id"]].copy()
    result["anomaly_score_ae"] = mse
    result["rank_ae"] = result["anomaly_score_ae"].rank(ascending=False, method="min").astype(int)

    if save:
        out = PROCESSED_DIR / "ae_scores.parquet"
        result.to_parquet(out, index=False)
        print(f"AE scores saved → {out}")
        # Save model state + metadata
        torch.save(model.state_dict(), PROCESSED_DIR / "autoencoder_weights.pt")
        with open(PROCESSED_DIR / "autoencoder_meta.pkl", "wb") as f:
            pickle.dump(
                {
                    "scaler": scaler,
                    "features": FEATURE_COLS,
                    "input_dim": input_dim,
                    "hidden_dims": HIDDEN_DIMS,
                    "latent_dim": LATENT_DIM,
                    "val_history": history,
                },
                f,
            )

    return result


if __name__ == "__main__":
    result = run()
    top = result.nsmallest(20, "rank_ae")[["taker", "is_leaked_market", "case_id", "anomaly_score_ae", "rank_ae"]]
    print("\nTop-20 anomalies (Autoencoder):")
    print(top.to_string(index=False))
    leaked_in_top = result[result["rank_ae"] <= 100]["is_leaked_market"].sum()
    print(f"\nPrecision@100 (AE): {leaked_in_top}/100 = {leaked_in_top/100:.3f}")
