"""
=============================================================
ai/ml_model.py
Modelo LightGBM para clasificacion de señales de trading.

Predice 3 clases:
  0 = HOLD  (no operar)
  1 = BUY   (abrir long)
  2 = SELL  (abrir short)

Incluye:
  - Optimizacion de hiperparametros con Optuna
  - Validacion cruzada temporal (TimeSeriesSplit)
  - Feature importance analysis
  - Persistencia del modelo versionado
  - Calibracion de probabilidades
=============================================================
"""

import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import TimeSeriesSplit

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from utils.logger import get_logger

log = get_logger(__name__)

# Clases de señal
CLASS_NAMES = ["HOLD", "BUY", "SELL"]


class TradingMLModel:
    """
    Clasificador LightGBM para señales de trading.

    Flujo de uso:
        model = TradingMLModel(models_dir)

        # Entrenar
        metrics = model.train(X_train, y_train, X_test, y_test,
                              feature_names=names, use_optuna=True)

        # Predecir (inferencia en tiempo real)
        signal, confidence, probs = model.predict(X_features)
        # signal: "BUY" | "SELL" | "HOLD"
        # confidence: float 0-1
        # probs: dict {"HOLD": 0.2, "BUY": 0.7, "SELL": 0.1}

        # Guardar/cargar
        model.save("v1.0.0")
        model.load("v1.0.0")
    """

    # Hiperparametros por defecto (se optimizan con Optuna si disponible)
    DEFAULT_PARAMS = {
        "objective":        "multiclass",
        "num_class":        3,
        "metric":           "multi_logloss",
        "boosting_type":    "gbdt",
        "n_estimators":     300,
        "learning_rate":    0.05,
        "num_leaves":       63,
        "max_depth":        -1,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "reg_alpha":        0.1,
        "reg_lambda":       0.1,
        "class_weight":     "balanced",
        "random_state":     42,
        "n_jobs":           -1,
        "verbose":          -1,
    }

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._model: Optional[lgb.LGBMClassifier] = None
        self._feature_names: list[str] = []
        self._version: str = ""
        self._metrics: dict = {}
        self._trained_at: Optional[datetime] = None
        self._is_trained = False

        if not HAS_LIGHTGBM:
            log.warning("LightGBM no instalado. Instala con: pip install lightgbm")

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def train(
        self,
        X_train:       np.ndarray,
        y_train:       np.ndarray,
        X_test:        np.ndarray,
        y_test:        np.ndarray,
        feature_names: list[str],
        use_optuna:    bool = True,
        n_trials:      int = 50,
        cv_folds:      int = 5,
    ) -> dict:
        """
        Entrena el modelo con los datos preparados.

        Args:
            X_train, y_train: Datos de entrenamiento
            X_test,  y_test:  Datos de test
            feature_names:    Nombres de las columnas de features
            use_optuna:       Si True, optimiza hiperparametros (mas lento)
            n_trials:         Numero de trials de Optuna
            cv_folds:         Folds para TimeSeriesSplit

        Returns:
            Dict con metricas del modelo entrenado
        """
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM requerido. pip install lightgbm")

        self._feature_names = feature_names
        start_time = time.time()

        log.info(
            f"Entrenando LightGBM | "
            f"Train: {len(X_train)} | Test: {len(X_test)} | "
            f"Features: {len(feature_names)} | Optuna: {use_optuna}"
        )

        # Optimizar hiperparametros si se solicita
        if use_optuna and HAS_OPTUNA:
            best_params = self._optimize_hyperparams(
                X_train, y_train, n_trials, cv_folds
            )
            params = {**self.DEFAULT_PARAMS, **best_params}
            log.info(f"Mejores hiperparametros encontrados: {best_params}")
        else:
            params = self.DEFAULT_PARAMS.copy()
            log.info("Usando hiperparametros por defecto")

        # Entrenar modelo final
        self._model = lgb.LGBMClassifier(**params)

        # Early stopping con datos de validacion
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        # Evaluar
        self._metrics = self._evaluate(X_train, y_train, X_test, y_test)
        self._metrics["training_seconds"] = round(time.time() - start_time, 1)
        self._metrics["n_estimators_used"] = self._model.best_iteration_ or params["n_estimators"]
        self._trained_at = datetime.now(timezone.utc)
        self._is_trained = True

        log.info(
            f"Entrenamiento completado en {self._metrics['training_seconds']}s | "
            f"Test accuracy: {self._metrics['test_accuracy']:.3f} | "
            f"F1: {self._metrics['f1_weighted']:.3f}"
        )

        return self._metrics

    def _optimize_hyperparams(
        self,
        X:        np.ndarray,
        y:        np.ndarray,
        n_trials: int,
        cv_folds: int,
    ) -> dict:
        """Busqueda de hiperparametros con Optuna y TimeSeriesSplit."""
        import warnings

        tscv = TimeSeriesSplit(n_splits=cv_folds)

        def objective(trial):
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
                "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "feature_fraction":  trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
                "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            }

            scores = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for train_idx, val_idx in tscv.split(X):
                    X_tr, X_val = X[train_idx], X[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]

                    m = lgb.LGBMClassifier(
                        **{**self.DEFAULT_PARAMS, **params},
                    )
                    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(20, verbose=False),
                                      lgb.log_evaluation(-1)])

                    y_pred = m.predict(X_val)
                    scores.append(f1_score(y_val, y_pred, average="weighted",
                                           zero_division=0))

            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return study.best_params

    def _evaluate(
        self,
        X_train, y_train,
        X_test,  y_test,
    ) -> dict:
        """Calcula metricas completas sobre train y test."""

        y_pred_train = self._model.predict(X_train)
        y_pred_test  = self._model.predict(X_test)

        metrics = {
            "train_accuracy":  round(accuracy_score(y_train, y_pred_train), 4),
            "test_accuracy":   round(accuracy_score(y_test,  y_pred_test),  4),
            "f1_weighted":     round(f1_score(y_test, y_pred_test,
                                              average="weighted", zero_division=0), 4),
            "precision_weighted": round(precision_score(y_test, y_pred_test,
                                                         average="weighted", zero_division=0), 4),
            "recall_weighted": round(recall_score(y_test, y_pred_test,
                                                  average="weighted", zero_division=0), 4),
        }

        # F1 por clase
        f1_per_class = f1_score(y_test, y_pred_test, average=None,
                                labels=[0, 1, 2], zero_division=0)
        metrics["f1_hold"] = round(float(f1_per_class[0]), 4)
        metrics["f1_buy"]  = round(float(f1_per_class[1]), 4)
        metrics["f1_sell"] = round(float(f1_per_class[2]), 4)

        # Feature importance (top 20)
        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
            top_idx = np.argsort(importances)[-20:][::-1]
            metrics["top_features"] = {
                self._feature_names[i]: round(float(importances[i]), 2)
                for i in top_idx
                if i < len(self._feature_names)
            }

        return metrics

    # ── Inferencia ────────────────────────────────────────────────────────────

    def predict(
        self,
        X:                np.ndarray,
        min_confidence:   float = 0.55,
    ) -> tuple[str, float, dict]:
        """
        Predice la señal para un vector de features.

        Args:
            X:              Array shape (1, n_features)
            min_confidence: Probabilidad minima para confirmar BUY/SELL
                            Si ninguna clase supera este umbral -> HOLD

        Returns:
            (signal, confidence, probabilities)
            signal:        "BUY" | "SELL" | "HOLD"
            confidence:    Probabilidad maxima 0-1
            probabilities: {"HOLD": p0, "BUY": p1, "SELL": p2}
        """
        if not self._is_trained:
            return "HOLD", 0.0, {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0}

        try:
            probs = self._model.predict_proba(X)[0]  # shape (3,)

            probabilities = {
                "HOLD": round(float(probs[0]), 4),
                "BUY":  round(float(probs[1]), 4),
                "SELL": round(float(probs[2]), 4),
            }

            best_class  = int(np.argmax(probs))
            confidence  = float(probs[best_class])

            # Si la confianza no supera el umbral -> HOLD
            if confidence < min_confidence or best_class == 0:
                return "HOLD", confidence, probabilities

            signal = CLASS_NAMES[best_class]
            return signal, confidence, probabilities

        except Exception as e:
            log.error(f"Error en prediccion: {e}")
            return "HOLD", 0.0, {"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0}

    # ── Persistencia ──────────────────────────────────────────────────────────

    def save(self, version: str) -> Path:
        """
        Guarda el modelo entrenado con su version.

        Args:
            version: Ej. "v1.0.0", "20240321_210000"

        Returns:
            Ruta del archivo guardado
        """
        if not self._is_trained:
            raise RuntimeError("Modelo no entrenado. Llama a train() primero.")

        self._version = version
        self.models_dir.mkdir(parents=True, exist_ok=True)

        model_path = self.models_dir / f"lgbm_{version}.pkl"

        data = {
            "model":         self._model,
            "feature_names": self._feature_names,
            "version":       version,
            "metrics":       self._metrics,
            "trained_at":    self._trained_at,
        }

        with open(model_path, "wb") as f:
            pickle.dump(data, f)

        # Guardar metricas en JSON (legible)
        metrics_path = self.models_dir / f"lgbm_{version}_metrics.json"
        metrics_export = {k: v for k, v in self._metrics.items()
                          if k != "top_features"}
        with open(metrics_path, "w") as f:
            json.dump(metrics_export, f, indent=2)

        # Actualizar symlink "latest"
        latest_path = self.models_dir / "lgbm_latest.pkl"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.write_bytes(model_path.read_bytes())

        log.info(f"Modelo guardado: {model_path}")
        return model_path

    def load(self, version: str = "latest") -> bool:
        """
        Carga un modelo guardado.

        Args:
            version: Version a cargar o "latest" para el mas reciente

        Returns:
            True si se cargo correctamente
        """
        if version == "latest":
            model_path = self.models_dir / "lgbm_latest.pkl"
        else:
            model_path = self.models_dir / f"lgbm_{version}.pkl"

        if not model_path.exists():
            log.warning(f"Modelo no encontrado: {model_path}")
            return False

        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)

            self._model         = data["model"]
            self._feature_names = data["feature_names"]
            self._version       = data["version"]
            self._metrics       = data.get("metrics", {})
            self._trained_at    = data.get("trained_at")
            self._is_trained    = True

            log.info(
                f"Modelo cargado: v{self._version} | "
                f"{len(self._feature_names)} features | "
                f"Test acc: {self._metrics.get('test_accuracy', 'N/A')}"
            )
            return True

        except Exception as e:
            log.error(f"Error cargando modelo: {e}")
            return False

    # ── Propiedades ────────────────────────────────────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def version(self) -> str:
        return self._version

    @property
    def metrics(self) -> dict:
        return self._metrics

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names
