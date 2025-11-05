import argparse
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import cfg

def set_seed(seed: int):
    import random
    try:
        import torch
    except Exception:
        torch = None
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_table_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    return pd.read_excel(path, engine="openpyxl")

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    raw_target = cfg.target_col_raw.strip()
    if raw_target in df.columns and cfg.target_col_final not in df.columns:
        df.rename(columns={raw_target: cfg.target_col_final}, inplace=True)

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < before:
        print(f"Удалены дубликаты: {before - len(df)}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    nn_cols = [c.strip() for c in (cfg.feature_cols + [cfg.target_col_final]) if c.strip() in df.columns]
    if nn_cols:
        neg_mask = (df[nn_cols] < 0).any(axis=1)
        removed_neg = int(neg_mask.sum())
        if removed_neg > 0:
            print(f"Удалены строки с отрицательными значениями: {removed_neg}")
            df = df.loc[~neg_mask].reset_index(drop=True)

    feats = [c.strip() for c in cfg.feature_cols if c.strip() in df.columns]
    for c in feats:
        lo = df[c].quantile(0.01)
        hi = df[c].quantile(0.99)
        if pd.isna(lo) or pd.isna(hi) or hi <= lo:
            continue
        df[c] = df[c].clip(lower=lo, upper=hi)

    return df

def standardize_split_save(df: pd.DataFrame):
    feats = [c.strip() for c in cfg.feature_cols]
    target = cfg.target_col_final

    missing = [c for c in feats + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют ожидаемые признаки: {missing}")

    X = df[feats].astype("float32")
    y = df[target].astype("float32")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg.seed
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.values)
    X_test_s  = scaler.transform(X_test.values)

    train_df = pd.DataFrame(X_train_s, columns=feats, index=X_train.index)
    train_df[target] = y_train.values

    test_df = pd.DataFrame(X_test_s, columns=feats, index=X_test.index)
    test_df[target] = y_test.values

    ensure_dir(cfg.data_dir)
    train_path = os.path.join(cfg.data_dir, cfg.train_csv_name)
    test_path  = os.path.join(cfg.data_dir, cfg.test_csv_name)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    with open(os.path.join(cfg.data_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"Сохранен train -> {train_path} (rows={len(train_df)})")
    print(f"Сохранен test  -> {test_path} (rows={len(test_df)})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    args = parser.parse_args()

    set_seed(cfg.seed)

    input_path = args.input or os.path.join(cfg.data_dir, cfg.raw_xlsx_name)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Не найден исходный датасет: {input_path}")

    print(f"Считываем датасет: {input_path}")
    df_raw = read_table_any(input_path)

    print("Проводим чистку...")
    df_clean = clean_dataframe(df_raw)

    print("Стандартизируем, сплитуем, сохраняем csv...")
    standardize_split_save(df_clean)

    print("Препроцессинг завершен.")

if __name__ == "__main__":
    main()
