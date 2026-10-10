import pytest
from fastapi.testclient import TestClient

from src.service.schemas import Customer


@pytest.fixture(scope="module")
def client(tmp_path_factory, trained_pipeline):
    """Подсовываем сервису модель из фикстуры: тесты API не должны зависеть
    от того, лежит ли в models/ артефакт с прошлого запуска."""
    from src.service import app as app_module

    with TestClient(app_module.app) as c:
        app_module.holder.model = trained_pipeline
        app_module.holder.version = "test:fixture"
        yield c


VALID = Customer(
    tenure_months=3,
    monthly_charges=89.4,
    total_charges=268.2,
    contract_type="month-to-month",
    internet_service="fiber",
    payment_method="electronic_check",
    num_support_calls=4,
    has_tech_support=0,
    is_senior=0,
    avg_monthly_gb=61.3,
)


def test_predict_returns_valid_probability(client):
    body = client.post("/predict", json=dict(VALID)).json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["model_version"]


def test_unknown_category_returns_422(client):
    assert client.post("/predict", json=dict(VALID, contract_type="lifetime")).status_code == 422


def test_health_ok_when_model_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "test:fixture"


def test_predict_missing_field_returns_422(client):
    body = {k: v for k, v in dict(VALID).items() if k != "tenure_months"}
    r = client.post("/predict", json=body)
    assert r.status_code == 422


def test_predict_out_of_range_returns_422(client):
    assert client.post("/predict", json=dict(VALID, tenure_months=-1)).status_code == 422
    assert client.post("/predict", json=dict(VALID, tenure_months=999)).status_code == 422
    assert client.post("/predict", json=dict(VALID, monthly_charges=-5.0)).status_code == 422


def test_predict_batch_returns_all_items(client):
    r = client.post("/predict/batch", json={"items": [dict(VALID), dict(VALID)]})
    assert r.status_code == 200

    body = r.json()
    assert body["count"] == 2
    assert len(body["items"]) == 2
    for item in body["items"]:
        assert 0.0 <= item["churn_probability"] <= 1.0
        assert item["churn"] in (0, 1)
        assert item["model_version"] == "test:fixture"


def test_predict_empty_batch_returns_422(client):
    r = client.post("/predict/batch", json={"items": []})
    assert r.status_code == 422
