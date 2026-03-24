"""
=============================================================
ai/model_trainer.py
Orquestador del entrenamiento del modelo de IA.

Coordina:
  - Recoleccion de datos de multiples simbolos
  - Feature engineering
  - Entrenamiento del modelo
  - Evaluacion y logging en PostgreSQL
  - Guardado versionado del modelo
=============================================================
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ai.feature_engineer import FeatureEngineer
from ai.ml_model import TradingMLModel
from database.connection import DatabaseManager
from database.models import ModelTrainingRun
from config.settings import Settings
from utils.logger import get_logger

log = get_logger(__name__)


class ModelTrainer:
    """
    Entrena el modelo de IA usando datos de multiples simbolos y timeframes.

    Consolida datos de todos los top simbolos en un solo dataset
    para maximizar el numero de muestras de entrenamiento.

    Uso:
        trainer = ModelTrainer(settings, db, data_pipeline)
        result  = await trainer.train(symbols=["BTCUSDT", "ETHUSDT"])
    """

    def __init__(
        self,
        settings: Settings,
        db:       DatabaseManager,
        models_dir: Optional[Path] = None,
    ):
        self.settings   = settings
        self.db         = db
        self.models_dir = models_dir or settings.get_models_path()

        self.fe    = FeatureEngineer(self.models_dir)
        self.model = TradingMLModel(self.models_dir)

        # Intentar cargar modelo existente
        self.model.load("latest")

    # ── Entrenamiento principal ───────────────────────────────────────────────

    async def train(
        self,
        symbols:     list[str],
        timeframe:   str = "15m",
        candles_per_symbol: int = 1000,
        use_optuna:  bool = True,
        n_trials:    int = 40,
    ) -> dict:
        """
        Entrena el modelo con datos de todos los simbolos indicados.

        Args:
            symbols:            Lista de pares a usar para entrenamiento
            timeframe:          Intervalo temporal
            candles_per_symbol: Velas a cargar por simbolo
            use_optuna:         Optimizar hiperparametros con Optuna
            n_trials:           Trials de Optuna

        Returns:
            Dict con metricas del entrenamiento
        """
        log.info(
            f"Iniciando entrenamiento | "
            f"{len(symbols)} simbolos | {timeframe} | "
            f"Optuna: {use_optuna}"
        )

        start_time = datetime.now(timezone.utc)

        # 1. Recolectar y combinar datos de todos los simbolos
        combined_df = await self._collect_training_data(
            symbols, timeframe, candles_per_symbol
        )

        if combined_df is None or len(combined_df) < 200:
            raise ValueError(
                f"Datos insuficientes para entrenar: "
                f"{len(combined_df) if combined_df is not None else 0} filas"
            )

        log.info(f"Dataset combinado: {len(combined_df)} filas de {len(symbols)} simbolos")

        # 2. Preparar features y target
        try:
            X_train, y_train, X_test, y_test, feature_names = (
                self.fe.prepare_training_data(combined_df)
            )
        except ValueError as e:
            raise ValueError(f"Error en feature engineering: {e}")

        # Estadisticas del dataset
        dist = self.fe.get_class_distribution(y_train)
        log.info(f"Distribucion de clases (train): {dist}")

        # 3. Entrenar modelo
        metrics = self.model.train(
            X_train, y_train,
            X_test,  y_test,
            feature_names=feature_names,
            use_optuna=use_optuna,
            n_trials=n_trials,
        )

        # 4. Guardar modelo y scaler
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.model.save(version)
        self.fe.save_scaler()

        # 5. Registrar en base de datos
        duration = int((datetime.now(timezone.utc) - start_time).total_seconds())
        await self._log_training_run(
            version=version,
            metrics=metrics,
            n_train=len(X_train),
            n_test=len(X_test),
            duration=duration,
        )

        log.info(
            f"Entrenamiento completado v{version} | "
            f"Acc: {metrics['test_accuracy']:.3f} | "
            f"F1: {metrics['f1_weighted']:.3f} | "
            f"{duration}s"
        )

        return {
            "version":    version,
            "metrics":    metrics,
            "n_train":    len(X_train),
            "n_test":     len(X_test),
            "n_features": len(feature_names),
            "symbols":    symbols,
            "duration_s": duration,
        }

    # ── Recoleccion de datos ──────────────────────────────────────────────────

    async def _collect_training_data(
        self,
        symbols:   list[str],
        timeframe: str,
        limit:     int,
    ) -> Optional[pd.DataFrame]:
        """
        Carga datos de multiples simbolos y los combina en un DataFrame.
        Agrega una columna 'symbol' para poder filtrar despues si se necesita.
        """
        from analysis.indicators import TechnicalIndicators
        from data.historic_collector import HistoricCollector

        ti        = TechnicalIndicators()
        collector = HistoricCollector(None, self.db)  # Sin cliente para solo leer BD

        dfs = []
        for symbol in symbols:
            try:
                df = await collector.load_candles_df(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit,
                )

                if df.empty or len(df) < 100:
                    log.debug(f"Pocos datos para {symbol}: {len(df)} velas")
                    continue

                # Calcular indicadores tecnicos
                df = ti.add_all(df)

                if df.empty:
                    continue

                df["symbol"] = symbol
                dfs.append(df)
                log.debug(f"  {symbol}: {len(df)} velas con indicadores")

            except Exception as e:
                log.warning(f"Error cargando {symbol}: {e}")

        if not dfs:
            return None

        # Combinar todos los DataFrames
        combined = pd.concat(dfs, ignore_index=True)
        combined.dropna(inplace=True)

        log.info(
            f"Datos recolectados: {len(combined)} filas "
            f"de {len(dfs)}/{len(symbols)} simbolos"
        )
        return combined

    # ── Logging en BD ─────────────────────────────────────────────────────────

    async def _log_training_run(
        self,
        version:  str,
        metrics:  dict,
        n_train:  int,
        n_test:   int,
        duration: int,
    ) -> None:
        """Guarda el registro del entrenamiento en PostgreSQL."""
        try:
            from decimal import Decimal

            async with self.db.async_session() as session:
                run = ModelTrainingRun(
                    model_name="lightgbm_classifier",
                    version=version,
                    training_samples=n_train,
                    test_samples=n_test,
                    train_accuracy=Decimal(str(metrics.get("train_accuracy", 0))),
                    test_accuracy=Decimal(str(metrics.get("test_accuracy", 0))),
                    precision=Decimal(str(metrics.get("precision_weighted", 0))),
                    recall=Decimal(str(metrics.get("recall_weighted", 0))),
                    f1_score=Decimal(str(metrics.get("f1_weighted", 0))),
                    hyperparams=metrics.get("top_features", {}),
                    model_path=str(self.models_dir / f"lgbm_{version}.pkl"),
                    is_deployed=True,
                    duration_seconds=duration,
                )
                session.add(run)

            log.info(f"Entrenamiento registrado en BD: v{version}")

        except Exception as e:
            log.warning(f"Error registrando entrenamiento en BD: {e}")

    # ── Propiedades ────────────────────────────────────────────────────────────

    @property
    def is_model_ready(self) -> bool:
        return self.model.is_trained

    @property
    def model_version(self) -> str:
        return self.model.version
