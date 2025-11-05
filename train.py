import os
import numpy as np
import pandas as pd
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import cfg
from dataset import TabularCSVDataset
from model import MLPRegressor
from utils import (
    set_seed, ensure_dir, save_json,
    mae, mse, rmse, r2, mape, wape, smape
)

SEED         = cfg.seed
DATA_DIR     = cfg.data_dir
TRAIN_CSV    = cfg.train_csv_name
OUT_DIR      = cfg.outputs_dir
MODEL_NAME   = cfg.model_name
BATCH_SIZE   = cfg.batch_size
NUM_WORKERS  = cfg.num_workers
VAL_FRACTION = cfg.val_size
HIDDEN_SIZES = cfg.hidden_sizes
DROPOUT      = cfg.dropout
LR           = cfg.lr
WEIGHT_DECAY = cfg.weight_decay
MAX_EPOCHS   = cfg.max_epochs
PATIENCE     = cfg.patience
AMP          = cfg.amp

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mse":   mse(y_true, y_pred),
        "rmse":  rmse(y_true, y_pred),
        "mae":   mae(y_true, y_pred),
        "mape":  mape(y_true, y_pred),
        "wape":  wape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2":    r2(y_true, y_pred),
    }

def run_epoch(model, loader, criterion, optimizer=None, device="cpu", amp=False, scaler=None) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    preds_all, y_all = [], []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).squeeze(1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        if amp and device.startswith("cuda"):
            with torch.cuda.amp.autocast():
                out = model(xb)
                loss = criterion(out, yb)
            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            out = model(xb)
            loss = criterion(out, yb)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds_all.append(out.detach().cpu().numpy())
        y_all.append(yb.detach().cpu().numpy())

    y_all = np.concatenate(y_all)
    preds_all = np.concatenate(preds_all)
    met = compute_metrics(y_all, preds_all)
    met["loss"] = total_loss / len(y_all)  
    return met

def main():
    set_seed(SEED)
    ensure_dir(OUT_DIR)

    train_csv_path = os.path.join(DATA_DIR, TRAIN_CSV)
    full_ds = TabularCSVDataset(train_csv_path)

    val_len = int(len(full_ds) * VAL_FRACTION)
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len],
                                    generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=NUM_WORKERS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = len([c.strip() for c in cfg.feature_cols])
    model = MLPRegressor(in_dim=in_dim, hidden_sizes=HIDDEN_SIZES, dropout=DROPOUT).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(AMP and device == "cuda"))

    best_val = float("inf")
    patience_left = PATIENCE
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, device, amp=AMP, scaler=scaler)
        va = run_epoch(model, val_loader, criterion, optimizer=None, device=device, amp=False)

        history.append({
            "epoch": epoch,
            
            "train_mse": tr["mse"], "train_rmse": tr["rmse"], "train_mae": tr["mae"],
            "train_mape": tr["mape"], "train_wape": tr["wape"], "train_smape": tr["smape"],
            "train_r2": tr["r2"],
            
            "val_mse": va["mse"], "val_rmse": va["rmse"], "val_mae": va["mae"],
            "val_mape": va["mape"], "val_wape": va["wape"], "val_smape": va["smape"],
            "val_r2": va["r2"],
        })

        improved = va["mse"] + 1e-9 < best_val
        if improved:
            best_val = va["mse"]
            patience_left = PATIENCE
            torch.save(model.state_dict(), os.path.join(OUT_DIR, MODEL_NAME))
        else:
            patience_left -= 1

        if epoch % 10 == 0 or improved:
            print(
                f"Epoch {epoch:03d} | "
                f"train: MSE {tr['mse']:.4f} RMSE {tr['rmse']:.3f} MAE {tr['mae']:.3f} "
                f"MAPE {tr['mape']:.2f}% WAPE {tr['wape']:.2f}% sMAPE {tr['smape']:.2f}% R² {tr['r2']:.3f} | "
                f"val:   MSE {va['mse']:.4f} RMSE {va['rmse']:.3f} MAE {va['mae']:.3f} "
                f"MAPE {va['mape']:.2f}% WAPE {va['wape']:.2f}% sMAPE {va['smape']:.2f}% R² {va['r2']:.3f}"
            )
        if patience_left == 0:
            print("Ранняя остановка.")
            break

    pd.DataFrame(history).to_csv(os.path.join(OUT_DIR, "history.csv"), index=False)
    save_json({"best_val_mse": best_val}, os.path.join(OUT_DIR, "metrics.json"))

if __name__ == "__main__":
    main()
