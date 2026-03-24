"""
=============================================================
ai/predictor.py
Motor de prediccion en tiempo real.

Se ejecuta en cada vela cerrada y genera señales de trading
con nivel de confianza para el motor de estrategia (Fase 4).

Caracteristicas:
  - Inferencia en <10ms por simbolo
  - Filtro de confianza minima configurable
  - Multi-timeframe confirmation (primario + secundario)
  - Guardado de señales en PostgreSQL para analisis posterior
  - Calculo de OBI en tiempo real desde el order book
=============================================================
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import numpy as np

from ai.feature_engineer import FeatureEngineer
from ai.ml_model import TradingMLModel
from analysis.indicators import TechnicalIndicators
from database.connection import DatabaseManager
from database.models import Signal, Symbol
from config.settings import Settings
from utils.logger import get_logger

log = get_logger(__name__)


class SignalResult:
    """Resultado de una prediccion para un simbolo."""

    __slots__ = (
        "symbol", "signal", "confidence", "probabilities",
        "price", "indicators_snapshot", "timeframe", "timestamp",
        "model_version",
    )

    def __init__(
        self,
        symbol:       str,
        signal:       str,
        confidence:   float,
        probabilities: dict,
        price:        float,
        timeframe:    str,
        indicators:   dict,
        model_version: str = "",
    ):
        self.symbol              = symbol
        self.signal              = signal
        self.confidence          = confidence
        self.probabilities       = probabilities
        self.price               = price
        self.timeframe           = timeframe
        self.indicators_snapshot = indicators
        self.timestamp           = datetime.now(timezone.utc)
        self.model_version       = model_version

    @property
    def is_actionable(self) -> bool:
        """True si la señal es BUY o SELL (no HOLD)."""
        return self.signal in ("BUY", "SELL")

    def __repr__(self) -> str:
        return (
            f"<Signal {self.signal} {self.symbol} "
            f"conf={self.confidence:.2%} @ {self.price}>"
        )


class RealTimePredictor:
    """
    Genera señales de trading en tiempo real usando el modelo entrenado.

    Se instancia una sola vez y se reutiliza en el loop principal.
    Mantiene un cache de DataFrames recientes por simbolo para
    no releer la BD en cada vela.

    Uso:
        predictor = RealTimePredictor(settings, db, data_pipeline)
        await predictor.initialize()

        # En el loop de trading (cada vela cerrada):
        result = await predictor.predict(symbol="BTCUSDT", timeframe="15m")
        if result.is_actionable:
            print(f"Señal: {result.signal} con {result.confidence:.1%} confianza")
    """

    def __init__(
        self,
        settings:     Settings,
        db:           DatabaseManager,
        models_dir:   Optional[object] = None,
        min_confidence: float = 0.58,
    ):
        self.settings       = settings
        self.db             = db
        self.models_dir     = models_dir or settings.get_models_path()
        self.min_confidence = min_confidence

        self.fe    = FeatureEngineer(self.models_dir)
        self.model = TradingMLModel(self.models_dir)
        self.ti    = TechnicalIndicators()

        # Cache de simbolo_id -> int
        self._symbol_id_cache: dict[str, int] = {}

        self._initialized = False

    async def initialize(self) -> bool:
        """
        Carga el modelo y scaler desde disco.

        Returns:
            True si el modelo esta listo para prediccion
        """
        model_loaded = self.model.load("latest")

        if not model_loaded:
            log.warning(
                "No se encontro modelo entrenado. "
                "Ejecuta el entrenamiento primero con ModelTrainer.train()"
            )
            self._initialized = False
            return False

        if not self.fe.is_ready:
            log.warning("Scaler no cargado")
            self._initialized = False
            return False

        self._initialized = True
        log.info(
            f"Predictor listo | Modelo v{self.model.version} | "
            f"{self.fe.n_features} features"
        )
        return True

    # ── Prediccion principal ──────────────────────────────────────────────────

    async def predict(
        self,
        symbol:     str,
        timeframe:  str,
        df=None,    # pd.DataFrame opcional (si ya se cargo)
        obi:        float = 0.0,
    ) -> SignalResult:
        """
        Genera una señal de trading para el simbolo/timeframe.

        Args:
            symbol:    Par de trading (ej. "BTCUSDT")
            timeframe: Intervalo (ej. "15m")
            df:        DataFrame con indicadores (opcional, se carga si None)
            obi:       Order Book Imbalance en tiempo real (-1 a 1)

        Returns:
            SignalResult con la señal, confianza y metadatos
        """
        if not self._initialized:
            return self._hold_signal(symbol, timeframe, 0.0)

        try:
            # Cargar DataFrame si no se provee
            if df is None:
                df = await self._load_df(symbol, timeframe)

            if df is None or df.empty:
                log.debug(f"Sin datos para {symbol}/{timeframe}")
                return self._hold_signal(symbol, timeframe, 0.0)

            # Preparar features para la ultima vela
            X = self.fe.prepare_inference(df)
            if X is None:
                return self._hold_signal(symbol, timeframe, 0.0)

            # Prediccion del modelo
            signal, confidence, probs = self.model.predict(
                X, min_confidence=self.min_confidence
            )

            # Ajuste por OBI: si el OBI contradice la señal, reducir confianza
            if signal == "BUY"  and obi < -0.3:
                confidence *= 0.85
            elif signal == "SELL" and obi > 0.3:
                confidence *= 0.85

            # Snapshot de indicadores clave para logging
            indicators = self._extract_indicators_snapshot(df)
            price = float(df["close"].iloc[-1]) if not df.empty else 0.0

            result = SignalResult(
                symbol=symbol,
                signal=signal,
                confidence=confidence,
                probabilities=probs,
                price=price,
                timeframe=timeframe,
                indicators=indicators,
                model_version=self.model.version,
            )

            if result.is_actionable:
                log.info(
                    f"Señal generada: {signal} {symbol}/{timeframe} | "
                    f"Confianza: {confidence:.2%} | "
                    f"Precio: {price}"
                )
                # Guardar señal en BD de forma no bloqueante
                asyncio.create_task(self._save_signal(result))

            return result

        except Exception as e:
            log.error(f"Error en prediccion {symbol}: {e}")
            return self._hold_signal(symbol, timeframe, 0.0)

    async def predict_batch(
        self,
        symbols:   list[str],
        timeframe: str,
    ) -> list[SignalResult]:
        """
        Predice señales para multiples simbolos en paralelo.

        Returns:
            Lista de SignalResult (solo los actionable, ordenados por confianza)
        """
        tasks = [
            self.predict(symbol=s, timeframe=timeframe)
            for s in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filtrar errores y HOLD, ordenar por confianza
        actionable = [
            r for r in results
            if isinstance(r, SignalResult) and r.is_actionable
        ]
        actionable.sort(key=lambda x: x.confidence, reverse=True)

        if actionable:
            log.info(
                f"Señales actionables: {len(actionable)}/{len(symbols)} | "
                f"Top: {actionable[0].signal} {actionable[0].symbol} "
                f"({actionable[0].confidence:.2%})"
            )

        return actionable

    # ── Utilidades internas ───────────────────────────────────────────────────

    async def _load_df(self, symbol: str, timeframe: str):
        """Carga DataFrame con indicadores desde la BD."""
        try:
            from data.historic_collector import HistoricCollector
            collector = HistoricCollector(None, self.db)
            df = await collector.load_candles_df(symbol, timeframe, limit=300)

            if df.empty:
                return None

            df = self.ti.add_all(df)
            return df if not df.empty else None

        except Exception as e:
            log.error(f"Error cargando DF {symbol}/{timeframe}: {e}")
            return None

    def _extract_indicators_snapshot(self, df) -> dict:
        """Extrae los indicadores mas relevantes de la ultima vela."""
        if df is None or df.empty:
            return {}

        last = df.iloc[-1]
        snapshot = {}

        for col in ["rsi_14", "macd", "macd_hist", "adx",
                    "bb_position", "atr_pct", "vol_ratio",
                    "zscore_20", "obv_signal"]:
            if col in df.columns:
                val = last.get(col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    snapshot[col] = round(float(val), 4)

        return snapshot

    def _hold_signal(
        self, symbol: str, timeframe: str, price: float
    ) -> SignalResult:
        """Retorna una señal HOLD por defecto."""
        return SignalResult(
            symbol=symbol,
            signal="HOLD",
            confidence=0.0,
            probabilities={"HOLD": 1.0, "BUY": 0.0, "SELL": 0.0},
            price=price,
            timeframe=timeframe,
            indicators={},
            model_version=self.model.version,
        )

    async def _save_signal(self, result: SignalResult) -> None:
        """Guarda la señal en PostgreSQL de forma asincrona."""
        try:
            from sqlalchemy import select as sa_select

            async with self.db.async_session() as session:
                # Obtener symbol_id
                if result.symbol not in self._symbol_id_cache:
                    stmt = sa_select(Symbol.id).where(
                        Symbol.symbol == result.symbol
                    )
                    res  = await session.execute(stmt)
                    sym_id = res.scalar_one_or_none()
                    if sym_id is None:
                        return
                    self._symbol_id_cache[result.symbol] = sym_id

                sym_id = self._symbol_id_cache[result.symbol]

                signal_class_map = {"HOLD": "hold", "BUY": "buy", "SELL": "sell"}

                signal_record = Signal(
                    symbol_id=sym_id,
                    signal_type=signal_class_map[result.signal],
                    confidence=Decimal(str(round(result.confidence, 4))),
                    price_at_signal=Decimal(str(result.price)),
                    timeframe=result.timeframe,
                    indicators=result.indicators_snapshot,
                    model_version=result.model_version,
                )
                session.add(signal_record)

        except Exception as e:
            log.debug(f"Error guardando señal: {e}")

    # ── Propiedades ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._initialized

    @property
    def model_version(self) -> str:
        return self.model.version
