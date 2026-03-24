"""
=============================================================
ai/auto_retrain.py
Scheduler de reentrenamiento automatico del modelo.

Reentrena el modelo periodicamente usando datos recientes
y lo despliega automaticamente si mejora las metricas.

Politica de despliegue:
  - Solo despliega si F1 nuevo >= F1 actual * 0.98
    (permite hasta 2% degradacion para aceptar nueva version)
  - Guarda el modelo anterior como backup
  - Notifica via log cuando hay un nuevo modelo en produccion
=============================================================
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional

from config.settings import Settings
from database.connection import DatabaseManager
from utils.logger import get_logger

log = get_logger(__name__)


class AutoRetrainer:
    """
    Gestiona el ciclo de vida del reentrenamiento automatico.

    Se ejecuta en background como una tarea asyncio.
    El intervalo se configura con RETRAIN_INTERVAL_HOURS en .env

    Uso:
        retrainer = AutoRetrainer(settings, db, trainer, predictor)
        task = asyncio.create_task(retrainer.run_forever())
        # Para forzar un reentrenamiento inmediato:
        await retrainer.retrain_now(symbols)
    """

    def __init__(
        self,
        settings:  Settings,
        db:        DatabaseManager,
        trainer,   # ModelTrainer
        predictor, # RealTimePredictor
    ):
        self.settings  = settings
        self.db        = db
        self.trainer   = trainer
        self.predictor = predictor

        self._interval_hours = settings.retrain_interval_hours
        self._last_retrain:  Optional[datetime] = None
        self._retrain_count  = 0
        self._running        = False

    async def run_forever(self, symbols_fn) -> None:
        """
        Loop de reentrenamiento automatico.

        Args:
            symbols_fn: Callable que retorna la lista actual de top simbolos
        """
        self._running = True
        interval_secs = self._interval_hours * 3600

        log.info(
            f"Auto-retrainer iniciado | "
            f"Intervalo: cada {self._interval_hours}h"
        )

        # Esperar el primer intervalo antes de entrenar
        await asyncio.sleep(interval_secs)

        while self._running:
            try:
                symbols = symbols_fn()
                if symbols:
                    await self.retrain_now(symbols)
                else:
                    log.warning("No hay simbolos disponibles para reentrenar")

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error en auto-retrain: {e}", exc_info=True)

            await asyncio.sleep(interval_secs)

    async def retrain_now(
        self,
        symbols:    list[str],
        timeframe:  str = "15m",
        use_optuna: bool = False,  # False en auto-retrain para ser mas rapido
    ) -> Optional[dict]:
        """
        Ejecuta un ciclo completo de reentrenamiento ahora.

        Args:
            symbols:    Simbolos a usar para el entrenamiento
            timeframe:  Timeframe principal
            use_optuna: Si True, optimiza hiperparametros (mas lento)

        Returns:
            Dict con metricas del nuevo modelo, o None si fallo
        """
        log.info(
            f"Iniciando reentrenamiento | "
            f"{len(symbols)} simbolos | "
            f"Reentrenamiento #{self._retrain_count + 1}"
        )

        # Guardar F1 del modelo actual para comparar
        current_f1 = self.predictor.model.metrics.get("f1_weighted", 0.0)

        try:
            result = await self.trainer.train(
                symbols=symbols,
                timeframe=timeframe,
                use_optuna=use_optuna,
                n_trials=30,
            )

            new_f1     = result["metrics"].get("f1_weighted", 0.0)
            new_version = result["version"]

            # Politica de despliegue
            deploy_threshold = current_f1 * 0.98  # Acepta hasta 2% degradacion

            if new_f1 >= deploy_threshold or current_f1 == 0.0:
                # Desplegar nuevo modelo
                loaded = await self._deploy_new_model(new_version)

                if loaded:
                    self._retrain_count += 1
                    self._last_retrain = datetime.now(timezone.utc)

                    improvement = new_f1 - current_f1
                    log.info(
                        f"Nuevo modelo desplegado v{new_version} | "
                        f"F1: {current_f1:.3f} -> {new_f1:.3f} "
                        f"({'+'if improvement>=0 else ''}{improvement:.3f})"
                    )
                    return result

            else:
                log.warning(
                    f"Nuevo modelo NO desplegado (F1 muy bajo): "
                    f"nuevo={new_f1:.3f} < umbral={deploy_threshold:.3f}"
                )
                return result

        except Exception as e:
            log.error(f"Error durante reentrenamiento: {e}", exc_info=True)
            return None

    async def _deploy_new_model(self, version: str) -> bool:
        """Carga el nuevo modelo en el predictor en vivo."""
        try:
            # Re-inicializar el predictor con el nuevo modelo
            loaded = await self.predictor.initialize()
            return loaded
        except Exception as e:
            log.error(f"Error desplegando modelo v{version}: {e}")
            return False

    def stop(self) -> None:
        self._running = False

    @property
    def last_retrain(self) -> Optional[datetime]:
        return self._last_retrain

    @property
    def retrain_count(self) -> int:
        return self._retrain_count

    def get_status(self) -> dict:
        return {
            "running":       self._running,
            "retrain_count": self._retrain_count,
            "last_retrain":  self._last_retrain.isoformat() if self._last_retrain else None,
            "interval_hours": self._interval_hours,
            "model_version": self.predictor.model_version,
        }
