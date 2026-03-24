"""
=============================================================
ai/feature_engineer.py
Pipeline de feature engineering para el modelo de ML.

Transforma el DataFrame de indicadores tecnicos en features
listos para entrenar y hacer inferencia con LightGBM.

Responsabilidades:
  - Seleccion de features relevantes
  - Normalizacion con StandardScaler (persistido en disco)
  - Creacion de lag features temporales
  - Etiquetado de targets (BUY / SELL / HOLD)
  - Split train/test con purging para evitar data leakage
=============================================================
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

from utils.logger import get_logger

log = get_logger(__name__)


# ── Definicion de features ─────────────────────────────────────────────────────

# Features base calculados por TechnicalIndicators
BASE_FEATURES = [
    # Tendencia
    "dist_ema9", "dist_ema21", "dist_ema50", "price_vs_ema200",
    "macd", "macd_signal", "macd_hist",
    "adx", "ema_cross_9_21", "ema_cross_21_50",
    # Momentum
    "rsi_7", "rsi_14", "rsi_21",
    "stoch_k", "stoch_d",
    "williams_r", "cci",
    "roc_10", "roc_20",
    # Volatilidad
    "atr_pct", "bb_width", "bb_position",
    "volatility_10", "volatility_20",
    "hl_range_pct",
    # Volumen
    "obv_signal", "price_vs_vwap",
    "vol_ratio", "vol_ratio_5", "cmf",
    # Custom
    "zscore_20", "return_1", "return_2", "return_3", "return_5",
    "vol_trend", "streak",
]

# Lags a calcular para cada feature base de precio/momentum
LAG_FEATURES_BASE = ["return_1", "rsi_14", "macd_hist", "vol_ratio", "bb_position"]
LAG_PERIODS = [1, 2, 3, 5]

# Clases de la señal
SIGNAL_CLASSES = {0: "HOLD", 1: "BUY", 2: "SELL"}
CLASS_NAMES    = ["HOLD", "BUY", "SELL"]


class FeatureEngineer:
    """
    Transforma DataFrames OHLCV+TA en matrices de features para ML.

    El scaler se entrena una vez y se persiste en disco para que
    la inferencia en tiempo real use exactamente los mismos parametros.

    Uso:
        fe = FeatureEngineer(models_dir)

        # Entrenamiento
        X_train, y_train, X_test, y_test = fe.prepare_training_data(df)
        fe.save_scaler()

        # Inferencia en tiempo real (una vela)
        features = fe.prepare_inference(df_last_100_candles)
        # features tiene shape (1, n_features), listo para model.predict()
    """

    def __init__(self, models_dir: Path):
        self.models_dir   = models_dir
        self.scaler_path  = models_dir / "scaler.pkl"
        self._scaler: Optional[RobustScaler] = None
        self._feature_names: list[str] = []
        self._is_fitted    = False

        # Intentar cargar scaler existente
        self._load_scaler()

    # ── Preparacion para entrenamiento ────────────────────────────────────────

    def prepare_training_data(
        self,
        df:            pd.DataFrame,
        test_size:     float = 0.2,
        min_confidence: float = 0.003,  # 0.3% minimo para BUY/SELL
        purge_periods:  int = 5,        # Eliminar N periodos entre train/test
    ) -> tuple:
        """
        Prepara train/test splits listos para LightGBM.

        Args:
            df:             DataFrame con indicadores calculados
            test_size:      Fraccion para test (respeta orden temporal)
            min_confidence: Retorno minimo para clasificar como BUY/SELL
            purge_periods:  Periodos a eliminar entre train y test (anti-leakage)

        Returns:
            (X_train, y_train, X_test, y_test, feature_names)
        """
        if df is None or len(df) < 100:
            raise ValueError(f"DataFrame insuficiente: {len(df) if df is not None else 0} filas (minimo 100)")

        df = df.copy()

        # 1. Calcular target
        df = self._add_target(df, min_confidence)

        # 2. Agregar lag features
        df = self._add_lag_features(df)

        # 3. Eliminar filas con NaN
        df.dropna(inplace=True)

        if len(df) < 60:
            raise ValueError(f"Muy pocas filas tras limpiar NaN: {len(df)}")

        # 4. Seleccionar features finales
        feature_cols = self._get_feature_columns(df)
        self._feature_names = feature_cols

        X = df[feature_cols].values
        y = df["target_signal"].values

        # 5. Split temporal (NO aleatorio para series de tiempo)
        split_idx  = int(len(X) * (1 - test_size))
        purge_end  = min(split_idx + purge_periods, len(X))

        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test  = X[purge_end:]
        y_test  = y[purge_end:]

        # 6. Ajustar y aplicar scaler
        self._scaler = RobustScaler()
        X_train = self._scaler.fit_transform(X_train)
        X_test  = self._scaler.transform(X_test)
        self._is_fitted = True

        log.info(
            f"Features preparados: {len(feature_cols)} features | "
            f"Train: {len(X_train)} | Test: {len(X_test)} | "
            f"Clases: BUY={np.sum(y_train==1)} "
            f"SELL={np.sum(y_train==2)} "
            f"HOLD={np.sum(y_train==0)}"
        )

        return X_train, y_train, X_test, y_test, feature_cols

    def prepare_inference(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Prepara el vector de features de la ULTIMA vela para inferencia.

        Args:
            df: DataFrame de las ultimas N velas (minimo 50) con indicadores

        Returns:
            Array shape (1, n_features) normalizado, o None si hay error
        """
        if not self._is_fitted:
            log.warning("Scaler no ajustado. Carga un modelo entrenado primero.")
            return None

        if df is None or len(df) < 50:
            log.warning(f"DataFrame insuficiente para inferencia: {len(df) if df is not None else 0}")
            return None

        try:
            df = df.copy()
            df = self._add_lag_features(df)
            df.dropna(inplace=True)

            if df.empty:
                return None

            # Tomar la ultima fila (vela mas reciente)
            last_row = df[self._feature_names].iloc[[-1]].values
            return self._scaler.transform(last_row)

        except KeyError as e:
            log.error(f"Feature faltante en DataFrame de inferencia: {e}")
            return None
        except Exception as e:
            log.error(f"Error preparando inferencia: {e}")
            return None

    # ── Construccion del target ────────────────────────────────────────────────

    def _add_target(
        self,
        df:             pd.DataFrame,
        min_confidence: float = 0.003,
    ) -> pd.DataFrame:
        """
        Crea la columna target_signal con 3 clases:
          0 = HOLD  (movimiento insignificante)
          1 = BUY   (sube mas de min_confidence en N periodos futuros)
          2 = SELL  (baja mas de min_confidence en N periodos futuros)

        Usa retorno maximo en ventana futura para reducir falsos negativos.
        """
        # Retorno maximo en las proximas 3 velas (forward looking)
        future_high  = df["high"].rolling(3).max().shift(-3)
        future_low   = df["low"].rolling(3).min().shift(-3)

        max_up   = (future_high - df["close"]) / df["close"]
        max_down = (df["close"] - future_low)  / df["close"]

        conditions = [
            max_up   >= min_confidence,   # BUY
            max_down >= min_confidence,   # SELL
        ]
        choices = [1, 2]

        df["target_signal"] = np.select(conditions, choices, default=0)

        # Eliminar las ultimas 3 filas (no tienen target valido)
        df = df.iloc[:-3]

        return df

    # ── Lag features ──────────────────────────────────────────────────────────

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega versiones lagged de los features mas importantes.
        Captura la dinamica temporal que LightGBM necesita.
        """
        for col in LAG_FEATURES_BASE:
            if col not in df.columns:
                continue
            for lag in LAG_PERIODS:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

        # Rolling stats rapidos
        if "return_1" in df.columns:
            df["return_rolling_mean5"]  = df["return_1"].rolling(5).mean()
            df["return_rolling_std5"]   = df["return_1"].rolling(5).std()
            df["return_rolling_mean10"] = df["return_1"].rolling(10).mean()

        if "vol_ratio" in df.columns:
            df["vol_spike"] = (df["vol_ratio"] > 2.0).astype(int)

        return df

    # ── Seleccion de columnas ─────────────────────────────────────────────────

    def _get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Retorna la lista final de columnas a usar como features.
        Solo incluye columnas que existan en el DataFrame.
        """
        # Base features
        candidates = list(BASE_FEATURES)

        # Lag features generados
        for col in LAG_FEATURES_BASE:
            for lag in LAG_PERIODS:
                candidates.append(f"{col}_lag{lag}")

        # Rolling stats
        candidates += [
            "return_rolling_mean5",
            "return_rolling_std5",
            "return_rolling_mean10",
            "vol_spike",
        ]

        # Filtrar solo los que existen en el DataFrame
        available = [c for c in candidates if c in df.columns]

        # Verificar que no hayan columnas con demasiados NaN
        valid = []
        for col in available:
            nan_pct = df[col].isna().mean()
            if nan_pct < 0.05:  # Maximo 5% NaN
                valid.append(col)
            else:
                log.debug(f"Feature descartado por NaN alto ({nan_pct:.1%}): {col}")

        log.debug(f"Features seleccionados: {len(valid)} de {len(candidates)} candidatos")
        return valid

    # ── Persistencia del scaler ───────────────────────────────────────────────

    def save_scaler(self) -> None:
        """Guarda el scaler y nombres de features en disco."""
        if not self._is_fitted:
            log.warning("Scaler no ajustado, nada que guardar")
            return

        data = {
            "scaler":        self._scaler,
            "feature_names": self._feature_names,
        }
        self.models_dir.mkdir(parents=True, exist_ok=True)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(data, f)

        log.info(f"Scaler guardado: {self.scaler_path}")

    def _load_scaler(self) -> None:
        """Carga el scaler desde disco si existe."""
        if not self.scaler_path.exists():
            return

        try:
            with open(self.scaler_path, "rb") as f:
                data = pickle.load(f)

            self._scaler       = data["scaler"]
            self._feature_names = data["feature_names"]
            self._is_fitted    = True
            log.info(
                f"Scaler cargado: {len(self._feature_names)} features"
            )
        except Exception as e:
            log.warning(f"No se pudo cargar el scaler: {e}")

    # ── Propiedades ────────────────────────────────────────────────────────────

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    @property
    def n_features(self) -> int:
        return len(self._feature_names)

    @property
    def is_ready(self) -> bool:
        return self._is_fitted

    def get_class_distribution(self, y: np.ndarray) -> dict:
        """Retorna la distribucion de clases del target."""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        return {
            SIGNAL_CLASSES[int(c)]: {
                "count": int(n),
                "pct":   round(int(n) / total * 100, 1),
            }
            for c, n in zip(unique, counts)
        }
