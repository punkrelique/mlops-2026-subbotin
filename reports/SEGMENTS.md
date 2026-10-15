# Качество по сегментам

Тест, общий ROC-AUC: **0.8074**. Порог gate по сегменту: 0.7 для сегментов больше 200 объектов.

| Срез | Сегмент | Размер | ROC-AUC | Отклонение от общего |
|---|---|---:|---:|---:|
| contract_type | month-to-month | 2015 | 0.7783 | -0.0292 |
| contract_type | one_year | 1095 | 0.8033 | -0.0042 |
| contract_type | two_year | 890 | 0.7941 | -0.0133 |
| internet_service | dsl | 1674 | 0.8042 | -0.0033 |
| internet_service | fiber | 1664 | 0.7779 | -0.0295 |
| internet_service | none | 662 | 0.7610 | -0.0464 |
| tenure_bucket | первый год (1-12) | 657 | 0.7494 | -0.0580 |
| tenure_bucket | второй год (13-24) | 697 | 0.7369 | -0.0705 |
| tenure_bucket | старожилы (25+) | 2646 | 0.7875 | -0.0200 |

Gate по сегментам: OK

# Дрейф по сегментам: train против data/raw/fresh.csv

Wasserstein (normed) для числовых, Jensen-Shannon для категориальных, порог 0.1. Внутри среза колонка, по которой режем, исключена. ROC-AUC на свежих данных показывает, во что дрейф обходится сегменту.

| Срез | Сегмент | train | current | Доля колонок с дрейфом | Колонки | ROC-AUC на current |
|---|---|---:|---:|---:|---|---:|
| все | все | 14000 | 5000 | 0.40 | internet_service, monthly_charges, num_support_calls, total_charges | 0.7885 |
| contract_type | month-to-month | 6891 | 3086 | 0.44 | internet_service, monthly_charges, num_support_calls, total_charges | 0.7756 |
| contract_type | one_year | 4143 | 1199 | 0.44 | internet_service, monthly_charges, num_support_calls, total_charges | 0.7683 |
| contract_type | two_year | 2966 | 715 | 0.44 | internet_service, monthly_charges, num_support_calls, total_charges | 0.7551 |
| internet_service | dsl | 5937 | 1523 | 0.33 | monthly_charges, num_support_calls, total_charges | 0.7489 |
| internet_service | fiber | 5862 | 2918 | 0.44 | contract_type, monthly_charges, num_support_calls, total_charges | 0.7782 |
| internet_service | none | 2201 | 559 | 0.44 | avg_monthly_gb, monthly_charges, num_support_calls, total_charges | 0.7419 |
| tenure_bucket | первый год (1-12) | 2372 | 797 | 0.44 | internet_service, monthly_charges, num_support_calls, total_charges | 0.7488 |
| tenure_bucket | второй год (13-24) | 2371 | 830 | 0.44 | internet_service, monthly_charges, num_support_calls, total_charges | 0.7580 |
| tenure_bucket | старожилы (25+) | 9257 | 3373 | 0.44 | internet_service, monthly_charges, num_support_calls, total_charges | 0.7521 |
