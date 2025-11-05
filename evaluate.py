import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import cfg
from dataset import TabularCSVDataset
from model import MLPRegressor
from utils import mae, mse, rmse, r2, mape, wape, smape

DATA_DIR   = cfg.data_dir
TEST_CSV   = cfg.test_csv_name
OUT_DIR    = cfg.outputs_dir
MODEL_NAME = cfg.model_name

def main():
    test_csv_path = os.path.join(DATA_DIR, TEST_CSV)
    ds = TabularCSVDataset(test_csv_path)
    loader = DataLoader(ds, batch_size=1024, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = len([c.strip() for c in cfg.feature_cols])
    model = MLPRegressor(in_dim=in_dim)

    ckpt_path = os.path.join(OUT_DIR, MODEL_NAME)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("Не найден чекпоинт модели. Сначала запусти make train.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            y_pred.append(pred)
            y_true.append(yb.numpy())

    y_true = np.concatenate(y_true).reshape(-1)
    y_pred = np.concatenate(y_pred).reshape(-1)

    metrics = {
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }

    print("TEST METRICS:")
    for k, v in metrics.items():
        if k in ["MAE", "RMSE"]:
            print(f"{k:7s}: {v:8.3f}")
        else:
            print(f"{k:7s}: {v:8.3f}")

if __name__ == "__main__":
    main()
