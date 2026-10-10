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

## Настройте кэширование так, чтобы повторный прогон был быстрее первого (lesson 9)
Сократил (pipeline джобу)[https://github.com/punkrelique/mlops-2026-subbotin/actions/runs/37996209654/job/114042835647] с 1 минута 18 секунд до (1 минута 13 секунд)[https://github.com/punkrelique/mlops-2026-subbotin/actions/runs/38002233210/job/114062777698?pr=8]

## Почему проверка pytest в CI полезнее, чем та же проверка в pre-commit, хотя команда одна и та же? (lesson 9)
pre-commit можно случайно удалить или файл куда-то пропадет локально у разраба, а в CI явно видно, что было запущено и результат прозрачен

## API контракт (lesson 10)

### Сквозная трассировка запросов

Каждый ответ содержит заголовок `X-Request-ID`. Если он есть в запросе,
сервис возвращает то же значение, если нет, то генерирует новый.

### Ошибки

| Код | Когда | Тело |
|---|---|---|
| 422 | Тело запроса не прошло валидацию (неверный тип, значение вне диапазона, не входит в перечисленный список) | `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}` - стандартный формат FastAPI/Pydantic |
| 503 | Модель не загружена, либо (для `/reload`) не удалось загрузить новую версию | `{"detail": "Модель не загружена"}` |

### Объект `Customer`

Используется как тело запроса в `/predict`, `/explain` и как элемент списка `items` в `/predict/batch`.

| Поле | Тип | Ограничения |
|---|---|---|
| `tenure_months` | int | 0–200 |
| `monthly_charges` | float | 0–1000 |
| `total_charges` | float \| null | >=0, необязательное (по умолчанию `null`) |
| `contract_type` | string | `"month-to-month"` \| `"one_year"` \| `"two_year"` |
| `internet_service` | string | `"fiber"` \| `"dsl"` \| `"none"` |
| `payment_method` | string | `"electronic_check"` \| `"mailed_check"` \| `"bank_transfer"` \| `"credit_card"` |
| `num_support_calls` | int | 0–100 |
| `has_tech_support` | int | 0 или 1 |
| `is_senior` | int | 0 или 1 |
| `avg_monthly_gb` | float | 0–5000 |

Значение за пределами диапазона или не из перечисленного списка → `422`.

### `GET /health`

Проверка живости процесса и наличия загруженной модели. Без тела запроса.

Ответ `200`:
```json
{"status": "ok", "model_loaded": true, "model_version": "registry:churn_model/production"}
```
`status`: `"ok"`, если модель загружена, иначе `"degraded"`.

### `POST /predict`

Предсказание оттока для одного клиента. Тело запроса: объект `Customer`.

Ответ `200`:
```json
{
  "churn_probability": 0.54321,
  "churn": 1,
  "threshold": 0.245,
  "model_version": "registry:churn_model/production"
}
```
- `churn_probability` - вероятность оттока, округлена до 6 знаков после запятой
- `threshold` - порог классификации
`503`, если модель не загружена. `422` при некорректном теле

### `POST /predict/batch`

То же самое для нескольких клиентов за один вызов

Тело запроса:
```json
{"items": [ { "tenure_months": 3, "...": "..." }, { "...": "..." } ]}
```
`items` — список объектов `Customer`, от 1 до 1000 элементов.

Ответ `200`:
```json
{
  "items": [ { "churn_probability": 0.73, "churn": 1, "threshold": 0.245, "model_version": "..." }, { "...": "..." } ],
  "count": 2
}
```
Порядок `items` в ответе совпадает с порядком в запросе

`503`, если модель не загружена. `422`, если `items` пуст, содержит больше 1000
элементов или хотя бы один элемент не прошёл валидацию

### `POST /explain`

Объясняет вклад признаков в предсказание для одного клиента. Тело запроса: объект `Customer`

Ответ `200`:
```json
{
  "method": "linear_contribution",
  "contributions": {
    "cat__contract_type_month-to-month": 0.412,
    "num__monthly_charges": -0.187,
    "num__tenure_months": 0.093
  }
}
```
- `contributions` — топ 5 пар "признак : вклад", отсортированных по убыванию модуля вклада
- Ключи `contributions` — имена признаков **после препроцессинга** (категориальные
  поля разворачиваются в колонки вида `cat__contract_type_month-to-month`),
  а не названия полей `Customer`.
- `method` определяет смысл чисел в `contributions`, различать обязательно:
  - `"linear_contribution"` - вклад локальный: рассчитан для конкретного клиента
  - `"global_importance"` - вклад глобальный: это общая важность признака
    для модели в целом (`feature_importances_`), одинаковая для всех клиентов

`503`, если модель не загружена. `422` при некорректном теле.

### `POST /reload`

Перезагружает модель (из MLflow Registry либо из локального файла) без перезапуска процесса

Ответ `200`: тот же формат, что у `/health`, отражающий состояние после перезагрузки.

`503`, если перезагрузка не удалась. В этом случае сервис продолжает обслуживать предыдущую версию модели.

### 100 одиночных запросов против одного батча на 100 записе (lesson 10)

| Сценарий | Время (3 прогона) |
|---|---|
| 100x - `POST /predict` (последовательно) | 2336 мс / 4241 мс / 4398 мс - в среднем ~23-44 мс на запрос |
| 1x - `POST /predict/batch` (100 записей) | 64 мс / 18 мс / 18 мс |

- Фиксированные накладные расходы платятся один раз, а не 100. Каждый
  `POST /predict` - это отдельное HTTP соединение, отдельный разбор и валидация
  JSON через Pydantic, и отдельный вызов `load_params()` (`src/service/app.py`),
  который каждый раз заново читает и парсит `params.yaml` с диска. В батч-запросе
  все это происходит один раз для всех 100 записей

- Сетевой round-trip 100 последовательных запросов это 100 полных циклов
  "клиент ждёт ответ -> отправляет следующий". Один батч-запрос - это один
  round-trip независимо от размера батча

## Cколько уязвимостей нашлось и что вы с ними сделали. Правильный ответ не обязательно «починил все» — часть живёт в системных библиотеках базового образа (lesson 11)

Результат выполнения:
```sh
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image --severity HIGH,CRITICAL churn-service:local
```
```sh
churn-service:local (debian 13.7)
=================================
Total: 44 (HIGH: 44, CRITICAL: 0)

Python (python-pkg)
===================
Total: 2 (HIGH: 2, CRITICAL: 0)

┌───────────────────────────┬────────────────┬──────────┬────────┬───────────────────┬───────────────┬──────────────────────────────────────────────────────────────┐
│          Library          │ Vulnerability  │ Severity │ Status │ Installed Version │ Fixed Version │                            Title                             │
├───────────────────────────┼────────────────┼──────────┼────────┼───────────────────┼───────────────┼──────────────────────────────────────────────────────────────┤
│ jaraco.context (METADATA) │ CVE-2026-23949 │ HIGH     │ fixed  │ 5.3.0             │ 6.1.0         │ jaraco.context: jaraco.context: Path traversal via malicious │
│                           │                │          │        │                   │               │ tar archives                                                 │
│                           │                │          │        │                   │               │ https://avd.aquasec.com/nvd/cve-2026-23949                   │
├───────────────────────────┼────────────────┤          │        ├───────────────────┼───────────────┼──────────────────────────────────────────────────────────────┤
│ wheel (METADATA)          │ CVE-2026-24049 │          │        │ 0.45.1            │ 0.46.2        │ wheel: wheel: Privilege Escalation or Arbitrary Code         │
│                           │                │          │        │                   │               │ Execution via malicious wheel file...                        │
│                           │                │          │        │                   │               │ https://avd.aquasec.com/nvd/cve-2026-24049                   │
└───────────────────────────┴────────────────┴──────────┴────────┴───────────────────┴───────────────┴──────────────────────────────────────────────────────────────┘
```

Большинство уявзимостей находятся в базовых утилитах `Debian`, есть 2 уявзимости, которые идут из пакетов `Python`: `jaraco.context` и `wheel`. Это транзитивные зависимости установленных пакетов. В `jaraco.context` в версии 6.1.0 это починили, можно попробовать обновить пакет, а с `wheel` CVE говорит:"Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')". Вероятно проблема в валидации путей - можно обратить внимание на это, пока в библиотеке не починили