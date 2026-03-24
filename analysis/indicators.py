"""
=============================================================
analysis/indicators.py
Motor de indicadores tecnicos usando pandas-ta.

Calcula todos los indicadores necesarios para el modelo de IA:
  - Tendencia:  EMA, SMA, MACD, ADX, Ichimoku
  - Momentum:   RSI, Stochastic, Williams %R, CCI, MFI
  - Volatilidad: ATR, Bollinger Bands, Keltner Channels
  - Volumen:    OBV, VWAP, CMF, Volume Ratio
  - Custom:     Order Book Imbalance, Hurst Exponent, Z-score
=============================================================
"""

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

from utils.logger import get_logger, log_execution

log = get_logger(__name__)


class TechnicalIndicators:
    """
    Calcula indicadores tecnicos sobre un DataFrame OHLCV.

    Todos los metodos reciben y retornan un DataFrame de pandas,
    agregando columnas nuevas sin modificar las originales.

    Uso:
        ti = TechnicalIndicators()
        df = ti.add_all(df)   # Agrega TODOS los indicadores
        # o selectivos:
        df = ti.add_trend(df)
        df = ti.add_momentum(df)
    """

    def __init__(self):
        if not HAS_PANDAS_TA:
            log.warning(
                "pandas-ta no instalado. Instala con: pip install pandas-ta"
            )

    @log_execution
    def add_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega todos los indicadores al DataFrame.
        Este es el metodo principal para preparar features de ML.

        Args:
            df: DataFrame con columnas open, high, low, close, volume

        Returns:
            DataFrame con todas las columnas de indicadores agregadas
        """
        if df is None or len(df) < 50:
            log.warning("DataFrame muy pequeno para calcular indicadores (min 50)")
            return df

        df = df.copy()

        df = self.add_trend(df)
        df = self.add_momentum(df)
        df = self.add_volatility(df)
        df = self.add_volume(df)
        df = self.add_custom(df)

        # Eliminar filas con NaN al inicio (periodo de calentamiento de indicadores)
        df.dropna(inplace=True)

        log.debug(
            f"Indicadores calculados: {len(df.columns)} columnas, "
            f"{len(df)} filas validas"
        )
        return df

    # ── Indicadores de tendencia ───────────────────────────────────────────────

    def add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """EMA, SMA, MACD, ADX, PSAR."""

        # --- EMAs multi-periodo
        for period in [9, 21, 50, 200]:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

        # --- SMA
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()

        # --- MACD
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"]        = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]

        # --- ADX (Average Directional Index)
        if HAS_PANDAS_TA:
            adx = ta.adx(df["high"], df["low"], df["close"], length=14)
            if adx is not None and not adx.empty:
                df["adx"]    = adx.get("ADX_14", np.nan)
                df["dmi_pos"] = adx.get("DMP_14", np.nan)
                df["dmi_neg"] = adx.get("DMN_14", np.nan)
        else:
            df["adx"] = self._calc_adx_manual(df, 14)

        # --- Cruce de EMAs (senales de tendencia)
        df["ema_cross_9_21"]  = np.where(df["ema_9"]  > df["ema_21"],  1, -1)
        df["ema_cross_21_50"] = np.where(df["ema_21"] > df["ema_50"],  1, -1)
        df["price_vs_ema200"] = (df["close"] - df["ema_200"]) / df["ema_200"]

        # --- Distancia precio vs EMAs (normalizada)
        df["dist_ema9"]  = (df["close"] - df["ema_9"])  / df["ema_9"]
        df["dist_ema21"] = (df["close"] - df["ema_21"]) / df["ema_21"]
        df["dist_ema50"] = (df["close"] - df["ema_50"]) / df["ema_50"]

        return df

    # ── Indicadores de momentum ────────────────────────────────────────────────

    def add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI, Stochastic, Williams %R, CCI, MFI."""

        # --- RSI multi-periodo
        for period in [7, 14, 21]:
            df[f"rsi_{period}"] = self._calc_rsi(df["close"], period)

        # --- Stochastic
        if HAS_PANDAS_TA:
            stoch = ta.stoch(
                df["high"], df["low"], df["close"],
                k=14, d=3, smooth_k=3
            )
            if stoch is not None and not stoch.empty:
                df["stoch_k"] = stoch.get("STOCHk_14_3_3", np.nan)
                df["stoch_d"] = stoch.get("STOCHd_14_3_3", np.nan)
        else:
            low_14  = df["low"].rolling(14).min()
            high_14 = df["high"].rolling(14).max()
            df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14 + 1e-10)
            df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # --- Williams %R
        high_14 = df["high"].rolling(14).max()
        low_14  = df["low"].rolling(14).min()
        df["williams_r"] = -100 * (high_14 - df["close"]) / (high_14 - low_14 + 1e-10)

        # --- CCI (Commodity Channel Index)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad    = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        df["cci"] = (typical_price - sma_tp) / (0.015 * mad + 1e-10)

        # --- Rate of Change
        df["roc_10"] = df["close"].pct_change(10) * 100
        df["roc_20"] = df["close"].pct_change(20) * 100

        # --- Momentum puro
        df["momentum_10"] = df["close"] - df["close"].shift(10)

        return df

    # ── Indicadores de volatilidad ─────────────────────────────────────────────

    def add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR, Bollinger Bands, volatilidad historica."""

        # --- ATR (Average True Range)
        high_low   = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close  = (df["low"]  - df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        df["atr_14"] = true_range.ewm(alpha=1/14, adjust=False).mean()
        df["atr_21"] = true_range.ewm(alpha=1/21, adjust=False).mean()

        # ATR normalizado como % del precio
        df["atr_pct"] = df["atr_14"] / df["close"] * 100

        # --- Bollinger Bands (20, 2)
        sma20      = df["close"].rolling(20).mean()
        std20      = df["close"].rolling(20).std()
        df["bb_upper"]    = sma20 + 2 * std20
        df["bb_lower"]    = sma20 - 2 * std20
        df["bb_mid"]      = sma20
        df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / sma20
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"] + 1e-10
        )

        # --- Volatilidad historica (desviacion estandar de retornos)
        returns = df["close"].pct_change()
        df["volatility_10"] = returns.rolling(10).std() * np.sqrt(10)
        df["volatility_20"] = returns.rolling(20).std() * np.sqrt(20)

        # --- High-Low range
        df["hl_range_pct"] = (df["high"] - df["low"]) / df["close"] * 100

        return df

    # ── Indicadores de volumen ─────────────────────────────────────────────────

    def add_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """OBV, VWAP, relaciones de volumen."""

        # --- OBV (On-Balance Volume)
        obv = [0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["obv"] = obv

        # OBV normalizado (diferencia relativa)
        df["obv_ema"] = pd.Series(obv, index=df.index).ewm(span=20).mean()
        df["obv_signal"] = (df["obv"] - df["obv_ema"]) / (df["obv_ema"].abs() + 1e-10)

        # --- VWAP aproximado (precio ponderado por volumen, ventana deslizante)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap_num = (typical_price * df["volume"]).rolling(20).sum()
        vwap_den = df["volume"].rolling(20).sum()
        df["vwap"]         = vwap_num / (vwap_den + 1e-10)
        df["price_vs_vwap"] = (df["close"] - df["vwap"]) / df["vwap"]

        # --- Volume ratio (volumen actual vs promedio)
        df["vol_sma20"]   = df["volume"].rolling(20).mean()
        df["vol_ratio"]   = df["volume"] / (df["vol_sma20"] + 1e-10)
        df["vol_ratio_5"] = df["volume"].rolling(5).mean() / (df["vol_sma20"] + 1e-10)

        # --- CMF (Chaikin Money Flow)
        mfv = (
            ((df["close"] - df["low"]) - (df["high"] - df["close"]))
            / (df["high"] - df["low"] + 1e-10)
        ) * df["volume"]
        df["cmf"] = mfv.rolling(20).sum() / (df["volume"].rolling(20).sum() + 1e-10)

        return df

    # ── Indicadores custom ─────────────────────────────────────────────────────

    def add_custom(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicadores propietarios para el modelo de IA."""

        # --- Z-score del precio respecto a su media movil de 20 periodos
        sma20 = df["close"].rolling(20).mean()
        std20 = df["close"].rolling(20).std()
        df["zscore_20"] = (df["close"] - sma20) / (std20 + 1e-10)

        # --- Retornos a diferentes horizontes (features para ML)
        for lag in [1, 2, 3, 5, 10]:
            df[f"return_{lag}"] = df["close"].pct_change(lag)

        # --- Retorno futuro (TARGET para entrenamiento ML)
        # Retorno del proximo periodo (shifted -1)
        df["future_return_1"]  = df["close"].pct_change(1).shift(-1)
        df["future_return_3"]  = df["close"].pct_change(3).shift(-3)

        # Target binario: 1 si el precio sube mas de 0.3% en 1 vela
        df["target_up"] = (df["future_return_1"] > 0.003).astype(int)

        # --- Tendencia de volumen
        df["vol_trend"] = (
            df["volume"].rolling(5).mean() / df["volume"].rolling(20).mean() - 1
        )

        # --- Presion compradora/vendedora (si tenemos taker volumes)
        if "quote_volume" in df.columns:
            df["buy_pressure"] = df.get(
                "taker_buy_quote_vol", df["quote_volume"] * 0.5
            ) / (df["quote_volume"] + 1e-10)

        # --- Numero de velas consecutivas en la misma direccion
        direction = np.sign(df["close"].diff())
        consecutive = []
        count = 0
        for d in direction:
            if d > 0:
                count = max(count + 1, 1)
            elif d < 0:
                count = min(count - 1, -1)
            else:
                count = 0
            consecutive.append(count)
        df["streak"] = consecutive

        return df

    # ── Order Book Imbalance ───────────────────────────────────────────────────

    def calc_order_book_imbalance(
        self,
        bids: list[list[float]],
        asks: list[list[float]],
        levels: int = 10,
    ) -> float:
        """
        Calcula el desequilibrio del libro de ordenes.
        OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume)

        Valor positivo = presion compradora
        Valor negativo = presion vendedora

        Args:
            bids:   Lista de [precio, cantidad] para bids
            asks:   Lista de [precio, cantidad] para asks
            levels: Niveles del libro a considerar

        Returns:
            OBI entre -1 y 1
        """
        bid_vol = sum(float(b[1]) for b in bids[:levels])
        ask_vol = sum(float(a[1]) for a in asks[:levels])
        total   = bid_vol + ask_vol

        if total == 0:
            return 0.0

        return (bid_vol - ask_vol) / total

    # ── Metodos auxiliares ─────────────────────────────────────────────────────

    def _calc_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calcula RSI manualmente (para cuando no esta pandas-ta)."""
        delta = series.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _calc_adx_manual(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcula ADX manualmente."""
        high = df["high"]
        low  = df["low"]
        close = df["close"]

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)

        dm_pos = high.diff()
        dm_neg = -low.diff()
        dm_pos = dm_pos.where((dm_pos > dm_neg) & (dm_pos > 0), 0)
        dm_neg = dm_neg.where((dm_neg > dm_pos) & (dm_neg > 0), 0)

        atr   = tr.ewm(alpha=1/period, adjust=False).mean()
        di_pos = 100 * dm_pos.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10)
        di_neg = 100 * dm_neg.ewm(alpha=1/period, adjust=False).mean() / (atr + 1e-10)
        dx     = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg + 1e-10)
        adx    = dx.ewm(alpha=1/period, adjust=False).mean()
        return adx

    def get_feature_names(self) -> list[str]:
        """
        Retorna la lista de columnas que se usan como features en el modelo ML.
        Excluye columnas raw (OHLCV) y targets.
        """
        return [
            # Tendencia
            "ema_9", "ema_21", "ema_50",
            "dist_ema9", "dist_ema21", "dist_ema50",
            "macd", "macd_signal", "macd_hist",
            "adx", "ema_cross_9_21", "ema_cross_21_50", "price_vs_ema200",
            # Momentum
            "rsi_7", "rsi_14", "rsi_21",
            "stoch_k", "stoch_d",
            "williams_r", "cci",
            "roc_10", "roc_20", "momentum_10",
            # Volatilidad
            "atr_14", "atr_pct",
            "bb_width", "bb_position",
            "volatility_10", "volatility_20",
            "hl_range_pct",
            # Volumen
            "obv_signal", "price_vs_vwap",
            "vol_ratio", "vol_ratio_5", "cmf",
            # Custom / retornos
            "zscore_20",
            "return_1", "return_2", "return_3", "return_5", "return_10",
            "vol_trend", "streak",
        ]
