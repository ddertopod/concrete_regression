import os
import zipfile
import urllib.request
import pandas as pd
from config import cfg

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def download_dataset(url: str, dest_zip: str):
    print(f"Скачиваем датасет:\n{url}")
    urllib.request.urlretrieve(url, dest_zip)
    print(f"ZIP сохранен в {dest_zip}")

def extract_zip(zip_path: str, extract_to: str):
    print(f"Извлекаем {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    print(f"Извлечено в {extract_to}")

def convert_xls_to_xlsx(xls_path: str, xlsx_path: str):
    print(f"Конвертируем {xls_path} → {xlsx_path}")
    try:
        import xlrd  
        from openpyxl import Workbook

        book = xlrd.open_workbook(xls_path)
        sheet = book.sheet_by_index(0)

        wb = Workbook()
        ws = wb.active

        for r in range(sheet.nrows):
            row = []
            for c in range(sheet.ncols):
                row.append(sheet.cell_value(r, c))
            ws.append(row)

        wb.save(xlsx_path)
        os.remove(xls_path)
        print("Конвертировано успешно и удален старый .xls")
    except Exception as e:
        print(f"Ошибка конвертации: {e}")

def main():
    ensure_dir(cfg.data_dir)

    zip_path = os.path.join(cfg.data_dir, cfg.zip_name)
    xls_path = os.path.join(cfg.data_dir, cfg.raw_xls_name)
    xlsx_path = os.path.join(cfg.data_dir, cfg.raw_xlsx_name)

    if os.path.exists(xlsx_path):
        print(f"{xlsx_path} уже есть - пропускаем скачивание.")
        return

    download_dataset(cfg.download_url, zip_path)
    extract_zip(zip_path, cfg.data_dir)
    os.remove(zip_path)
    convert_xls_to_xlsx(xls_path, xlsx_path)
    print("Датасет готов!")

if __name__ == "__main__":
    main()
