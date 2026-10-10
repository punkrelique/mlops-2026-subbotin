from typing import Literal, Optional

from pydantic import BaseModel, Field

ContractType = Literal["month-to-month", "one_year", "two_year"]
InternetService = Literal["fiber", "dsl", "none"]
PaymentMethod = Literal["electronic_check", "mailed_check", "bank_transfer", "credit_card"]
Method = Literal["linear_contribution", "global_importance"]


class Customer(BaseModel):
    tenure_months: int = Field(..., ge=0, le=200, description="Стаж клиента в месяцах")
    monthly_charges: float = Field(..., ge=0, le=1000)
    total_charges: Optional[float] = Field(None, ge=0)
    contract_type: ContractType
    internet_service: InternetService
    payment_method: PaymentMethod
    num_support_calls: int = Field(..., ge=0, le=100)
    has_tech_support: int = Field(..., ge=0, le=1)
    is_senior: int = Field(..., ge=0, le=1)
    avg_monthly_gb: float = Field(..., ge=0, le=5000)


class BatchRequest(BaseModel):
    # Верхняя граница обязательна: запрос на миллион строк положит сервис.
    # Понятный отказ лучше, чем OOM.
    items: list[Customer] = Field(..., min_length=1, max_length=1000)


class Prediction(BaseModel):
    churn_probability: float
    churn: int
    threshold: float
    model_version: str


class BatchPrediction(BaseModel):
    items: list[Prediction] = Field(..., min_length=1, max_length=1000)
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class Explanation(BaseModel):
    method: Method
    contributions: dict[str, float]
