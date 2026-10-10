import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src.config import feature_columns, load_params
from src.logging_setup import request_id_var
from src.service.model_loader import holder, log
from src.service.schemas import (
    BatchPrediction,
    BatchRequest,
    Customer,
    Explanation,
    HealthResponse,
    Prediction,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Модель грузится ОДИН раз на старте. Если грузить в обработчике,
    # каждый запрос будет идти секундами.
    try:
        holder.load()
    except Exception as exc:
        log.error("не удалось загрузить модель: %s", exc)
    yield


app = FastAPI(
    title="Churn Prediction Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Жив ли процесс и загружена ли модель."""
    return HealthResponse(
        status="ok" if holder.loaded else "degraded",
        model_loaded=holder.loaded,
        model_version=holder.version,
    )


@app.post("/predict", response_model=Prediction, tags=["inference"])
def predict(customer: Customer) -> Prediction:
    params = load_params()
    threshold = params["evaluate"]["threshold"]
    if not holder.loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    df = pd.DataFrame([customer.model_dump()])[feature_columns(params)]
    proba = float(holder.model.predict_proba(df)[0, 1])
    return Prediction(
        churn_probability=round(proba, 6),
        churn=int(proba >= threshold),
        threshold=threshold,
        model_version=holder.version,
    )


@app.post("/predict/batch", response_model=BatchPrediction, tags=["inference"])
def predict_batch(customer: BatchRequest) -> BatchPrediction | None:
    params = load_params()
    threshold = params["evaluate"]["threshold"]
    if not holder.loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    df = pd.DataFrame([c.model_dump() for c in customer.items])[feature_columns(params)]
    probas = holder.model.predict_proba(df)[:, 1]

    items = [
        Prediction(
            churn_probability=round(float(p), 6),
            churn=int(p >= threshold),
            threshold=threshold,
            model_version=holder.version,
        )
        for p in probas
    ]

    return BatchPrediction(items=items, count=len(items))


@app.post("/explain", response_model=Explanation, tags=["inference"])
def explain(customer: Customer) -> Explanation:
    """Вклад каждого признака в предсказание.

    Для линейной модели это коэффициент, умноженный на значение признака
    после препроцессинга. Для деревьев — feature_importances_, они
    глобальные, а не для конкретного клиента, и это нужно честно писать
    в ответе, чтобы бизнес не принял одно за другое.
    """
    if not holder.loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    params = load_params()
    df = pd.DataFrame([customer.model_dump()])[feature_columns(params)]

    preprocessor = holder.model.named_steps["preprocess"]
    estimator = holder.model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "coef_"):
        transformed = preprocessor.transform(df)[0]
        values = estimator.coef_[0] * transformed
        method = "linear_contribution"
    else:
        values = estimator.feature_importances_
        method = "global_importance"

    top5 = sorted(zip(feature_names, values), key=lambda item: abs(item[1]), reverse=True)[:5]
    contributions = {name: round(float(value), 6) for name, value in top5}
    return Explanation(method=method, contributions=contributions)


@app.post("/reload", response_model=HealthResponse, tags=["service"])
def reload_model() -> HealthResponse:
    try:
        holder.load()
    except Exception as exc:
        log.error("не удалось перезагрузить модель: %s", exc)
        raise HTTPException(status_code=503, detail="Не удалось перезагрузить модель") from exc

    return HealthResponse(
        status="ok" if holder.loaded else "degraded", model_loaded=holder.loaded, model_version=holder.version
    )
