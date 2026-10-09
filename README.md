![CI](https://github.com/punkrelique/mlops-2026-subbotin/actions/workflows/ci.yml/badge.svg)

[MLOPS2026](https://github.com/alsu124/mlops-deploy-course/blob/main/lessons/01-intro-project-setup/handout.md)

# Churn MLOps — стартовый шаблон

Это заготовка сквозного проекта курса. Здесь вы работаете весь семестр:
к декабрю из неё вырастет ML-сервис с пайплайном, тестами, CI/CD и мониторингом.

## Что делать прямо сейчас

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
make install
make check                         # должно напечатать environment: OK
make data                          # сгенерирует data/raw/churn.csv
```

Если `make check` печатает `FAIL` — читайте [../docs/setup.md](../docs/setup.md).

## Что уже готово, а что делаете вы

| Готово | Ваша работа |
|---|---|
| `src/config.py` — загрузка `params.yaml` | `src/data/prepare.py` |
| `src/logging_setup.py` — логирование | `src/features.py` |
| `src/data/generate.py` — генерация данных | `src/train.py` |
| `Makefile`, `params.yaml`, `.gitignore` | всё остальное, занятие за занятием |

Генерация данных отдана готовой намеренно: предмет курса — не выдумывание
датасета, а инженерия вокруг модели.

## Отправная точка

Файл [`notebooks/baseline_notebook.py`](notebooks/baseline_notebook.py) — это
реальный код «как из ноутбука»: работает у автора и больше нигде.
Первое занятие целиком про то, чтобы превратить его в воспроизводимый модуль.

## Правила проекта

1. **Никаких абсолютных путей.** Только `src.config.resolve()`.
2. **Никаких магических чисел в коде.** Всё числовое — в `params.yaml`.
3. **Данные и модели не коммитятся в Git.** С занятия 4 — в DVC.
4. **Секреты — только через переменные окружения.** `.env` уже в `.gitignore`.
5. **Работа идёт через ветки и Pull Request.** С занятия 2.

## Куда это вырастет

`../docs/syllabus.md` — план всех 18 занятий и артефакт каждого из них.

## Запуск MLFlow локально (lesson 6)
```sh
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

### Зачем тег `data_md5`, если уже есть `git_sha` (lesson 6)

`git_sha` фиксирует версию кода, но не версию данных. `generate` и кешируется DVC. Один и тот
же коммит кода можно прогнать на разных версиях `churn.csv` (другой `seed`). `git_sha` в этом случае не изменится,
а метрики будут другими. Тег `data_md5` это хеш файла данных из `dvc.lock`, он
привязывает run в MLFlow к конкретной версии данных, независимо от версии кода

## Выбор модели (lesson 6)

Модель: `logreg`: C=1.0, max_iter=1000

Метрика выбора: `ROC-AUC`. Она не зависит от порога классификации и от баланса классов

Почему`logreg` (ROC-AUC 0.8074): она обгоняет лучшие результаты `gradient_boosting`
(ROC-AUC 0.7999) и `random_forest` (0.7974). Внутри `logreg`
разница между `C=0.1` и `C=20.0` незначительна.

Порог:`0.245`. Высчитанный по формуле порог из предыдущих уроков для максимизации F1, если мы решаем,
что нам важно удержать клиентов, а не сэкономить на скидках

## Тесты и покрытие (lesson 8)

Покрытие (`pytest --cov=src --cov-report=term-missing`):
```sh
(.venv) punkrelique@DESKTOP-SHNMMS0:/mnt/c/Users/punkrelique/dev/mlops-2026-subbotin$ pytest --cov=src --cov-report=term-missing
...................                                                                                              [100%]

---------- coverage: platform linux, python 3.12.3-final-0 -----------
Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
src/__init__.py                   0      0   100%
src/config.py                    38     20    47%   33-34, 44-51, 56-65
src/data/__init__.py              3      0   100%
src/data/generate.py             53     17    68%   34-36, 97-111, 116
src/data/prepare.py              44     22    50%   42-78, 82
src/data/validate.py             46     46     0%   5-77
src/evaluate.py                  51     51     0%   5-77
src/features.py                  12      0   100%
src/logging_setup.py              8      0   100%
src/service/__init__.py           0      0   100%
src/service/model_loader.py      65     65     0%   1-99
src/smoke_check.py               33     33     0%   2-44
src/train.py                    109     87    20%   21-109, 133-138, 142-180, 184
-----------------------------------------------------------
TOTAL                           462    341    26%

19 passed in 6.74s
```

Цифра низкая, потому что считается по всему `src/`, а тестами осознанно покрыты только чистые
функции конвейера обучения (`generate`, `clean`, `split`, `build_preprocessor`, `build_pipeline`)

- `main()` в `prepare.py`, `train.py`, `generate.py` CLI обертки (argparse, чтение/запись
  CSV) вокруг уже протестированной логики. Такой тест `main()` проверял бы IO, а не поведение модели

- `log_to_mlflow()` в `train.py` требует поднятого MLflow-сервера, это задача не для юнит теста

- `src/data/validate.py`, `src/evaluate.py` CLI скрипты

- `src/smoke_check.py` ручная диагностика окружения, а не часть пайплайна обучения