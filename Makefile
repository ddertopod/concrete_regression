PYTHON := python3
DATA_DIR := data

DOWNLOAD_URL := $(shell $(PYTHON) -c "from config import cfg; print(cfg.download_url)")
RAW_XLS := $(shell $(PYTHON) -c "from config import cfg; print(cfg.raw_xls_name)")
RAW_XLSX := $(shell $(PYTHON) -c "from config import cfg; print(cfg.raw_xlsx_name)")
ZIP_NAME := $(notdir $(DOWNLOAD_URL))  

.PHONY: help start install prep train eval

help:
	@echo ""
	@echo "Make цели:"
	@echo "  make start    - установить зависимости из requirements.txt"
	@echo "  make install  - скачать датасет Concrete_Data.xls в папку ./data"
	@echo "  make prep     - запустить preprocessing.py (очистка, стандартизация, сплит)"
	@echo "  make help     - показать данное меню помощи"
	@echo ""


start:
	@echo "Установка зависимостей из requirements.txt..."
	@if [ -f requirements.txt ]; then \
		pip install -r requirements.txt; \
	else \
		echo "Не найден requirements.txt ! Пропускаем..."; \
	fi
	@echo "Зависимости установлены."


install:
	@echo "Скачиваем датасет ..."
	@$(PYTHON) install.py
	@echo "Датасет скачан в папку ./data."


prep:
	@echo "Запускаем препроцессинг..."
	@$(PYTHON) preprocessing.py --input "$(DATA_DIR)/$(RAW_XLSX)"
	@echo "Препроцессинг завершен."

train:
	@echo "Обучаем модель..."
	@$(PYTHON) train.py
	@echo "Обучение завершено. Выход в ./outputs"

eval:
	@echo "Проверяем модель на test.csv..."
	@$(PYTHON) evaluate.py
