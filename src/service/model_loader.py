import json
from pathlib import Path

import joblib
import mlflow

from src.config import feature_columns, load_params, resolve
from src.logging_setup import setup_logging

log = setup_logging()


class ModelHolder:
    def __init__(self) -> None:
        self.model = None
        self.version = "not-loaded"

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        params = load_params()
        if params["mlflow"]["enabled"] and self._try_registry(params):
            return
        self._load_local(params)

    def _try_registry(self, params) -> bool:
        try:
            import mlflow

            mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
            name = params["mlflow"]["registered_model_name"]
            stage = params["service"]["model_stage"]
            model_uri = f"models:/{name}@champion"
            self.check_signature(model_uri)

            self.model = mlflow.sklearn.load_model(model_uri)
            self.version = f"registry:{name}/{stage}"
            return True
        except Exception as exc:
            log.warning("MLflow Registry недоступен (%s), берём локальный файл", exc)
            return False

    def _load_local(self, params) -> None:
        self.model = joblib.load(resolve(params["service"]["model_path"]))

        models_dir = Path(resolve("models"))
        models_dir.mkdir(parents=True, exist_ok=True)
        meta_path = models_dir / "model_meta.json"

        meta = json.loads(meta_path.read_text())
        self.version = f"local:{meta['model']}"  # достаньте детали из models/model_meta.json

    import mlflow

    from src.config import feature_columns, load_params

    def check_signature(self, model_uri: str) -> None:
        signature = mlflow.models.get_model_info(model_uri).signature
        if signature is None:
            log.warning("у модели нет сигнатуры — обучите с input_example")
            return

        expected = set(feature_columns(load_params()))
        actual = {c.name for c in signature.inputs.inputs}
        if actual != expected:
            raise RuntimeError(
                f"сигнатура разошлась с конфигом: лишние {actual - expected}, недостающие {expected - actual}"
            )


holder = ModelHolder()
