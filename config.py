from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    
    seed: int = 42
    data_dir: str = "data"
    raw_xls_name: str = "Concrete_Data.xls"  
    raw_xlsx_name: str = "Concrete_Data.xlsx" 
    train_csv_name: str = "train.csv"
    test_csv_name: str = "test.csv"
    outputs_dir: str = "outputs"
    model_name: str = "best_model.pt"
    zip_name: str = "concrete+compressive+strength.zip"
    download_url: str = (
    "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"
)
    feature_cols: List[str] = field(default_factory=lambda: [
        "Cement (component 1)(kg in a m^3 mixture)",
        "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
        "Fly Ash (component 3)(kg in a m^3 mixture)",
        "Water  (component 4)(kg in a m^3 mixture)",
        "Superplasticizer (component 5)(kg in a m^3 mixture)",
        "Coarse Aggregate  (component 6)(kg in a m^3 mixture)",
        "Fine Aggregate (component 7)(kg in a m^3 mixture)",
        "Age (day)",
    ])
    target_col_raw: str = "Concrete compressive strength(MPa, megapascals) "
    target_col_final: str = "strength_mpa"
    batch_size: int = 64
    num_workers: int = 0
    val_size: float = 0.2  
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.05
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 500
    patience: int = 50
    amp: bool = True  

cfg = Config()
