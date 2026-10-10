# Будет расти по ходу курса. Правило: команду, которую приходится
# вспоминать по истории терминала, оформляем целью здесь.
.PHONY: help install check data prepare train test lint serve mlflow-server

ifeq ($(OS),Windows_NT)
VENV_BIN := .venv/Scripts
PY := python
else
VENV_BIN := .venv/bin
PY := python3
endif

export PATH := $(VENV_BIN):$(PATH) ## Выполнять команды в окружении

help:            ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:         ## Установить зависимости
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

check:           ## Проверить окружение
	$(PY) -m src.smoke_check

data:            ## Сгенерировать сырой датасет
	$(PY) -m src.data.generate

validate:        ## Провалидировать данные
	$(PY) -m src.data.validate

prepare:         ## Подготовить train/val/test
	$(PY) -m src.data.prepare

train:           ## Обучить модель
	$(PY) -m src.train

eval:            ## Оценить модель на тесте и проверить gate качества
	$(PY) -m src.evaluate

test:            ## Прогнать тесты
	pytest

lint:            ## Проверить стиль
	ruff check src tests

pipeline:        ## Воспроизвести пайплайн
	dvc repro

leaderboard:     ## Собрать таблицу лидеров из MLflow
	$(PY) -m scripts.leaderboard

serve:           ## Запустить сервис локально
	uvicorn src.service.app:app --host 0.0.0.0 --port 8000 --reload

mlflow-server:   ## Запустить MLflow локально
	mlflow server --host 127.0.0.1 --port 5000 \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlartifacts

docker-build:    ## Собрать образ
	docker build -f docker/Dockerfile -t churn-service:local .

docker-run:      ## Запустить контейнер
	docker run --rm -p 8000:8000 -v $(PWD)/models:/app/models:ro churn-service:local