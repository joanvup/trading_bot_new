"""
=============================================================
risk/sl_tp_calculator.py
Calculador dinamico de Stop Loss y Take Profit.

Metodos implementados:
  - ATR-based SL/TP (principal) — se ajusta a la volatilidad actual
  - Swing High/Low SL — usa maximos/minimos recientes
  - Trailing Stop — sigue el precio en direccion favorable
  - Risk:Reward enforcement — garantiza R:R minimo de 1.5

El SL y TP se recalculan en cada vela cerrada para
mantenerlos siempre optimos segun las condiciones actuales.
=============================================================
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SLTPLevels:
    """Niveles calculados de SL y TP."""
    stop_loss:   float
    take_profit: float
    risk_reward: float      # TP distance / SL distance
    sl_distance: float      # Distancia SL en USDT
    tp_distance: float      # Distancia TP en USDT
    sl_pct:      float      # SL como % del precio
    tp_pct:      float      # TP como % del precio
    method:      str        # Metodo usado ("atr", "swing", "fixed")
    trailing_activated: bool = False

    @property
    def is_valid(self) -> bool:
        return self.risk_reward >= 1.0 and self.sl_distance > 0


class SLTPCalculator:
    """
    Calcula niveles optimos de Stop Loss y Take Profit.

    El metodo ATR es el principal: el SL se coloca a
    N * ATR del precio de entrada, donde N se ajusta
    segun la volatilidad relativa del activo.

    Uso:
        calc = SLTPCalculator()

        # Para un LONG en BTCUSDT a $50,000
        levels = calc.calculate(
            direction="long",
            entry_price=50000,
            df=df_with_indicators,   # Necesita columna atr_14
            min_rr=1.5,
        )
        print(f"SL: {levels.stop_loss} | TP: {levels.take_profit}")
        print(f"R:R: {levels.risk_reward:.2f}")
    """

    def __init__(
        self,
        atr_sl_multiplier:  float = 1.5,   # SL a 1.5x ATR del precio
        atr_tp_multiplier:  float = 3.0,   # TP a 3.0x ATR del precio (R:R=2)
        min_risk_reward:    float = 1.5,   # R:R minimo aceptable
        min_sl_pct:         float = 0.003, # SL minimo 0.3% (evita ruido)
        max_sl_pct:         float = 0.05,  # SL maximo 5%
    ):
        self.atr_sl_mult  = atr_sl_multiplier
        self.atr_tp_mult  = atr_tp_multiplier
        self.min_rr       = min_risk_reward
        self.min_sl_pct   = min_sl_pct
        self.max_sl_pct   = max_sl_pct

    # ── Calculo principal ──────────────────────────────────────────────────────

    def calculate(
        self,
        direction:   str,          # "long" o "short"
        entry_price: float,
        df:          pd.DataFrame, # DataFrame con columnas atr_14, high, low
        min_rr:      Optional[float] = None,
    ) -> SLTPLevels:
        """
        Calcula SL y TP usando el metodo ATR con fallback a swing.

        Args:
            direction:   "long" (BUY) o "short" (SELL)
            entry_price: Precio de entrada
            df:          DataFrame con al menos las ultimas 20 velas + atr_14
            min_rr:      R:R minimo (sobreescribe el default)

        Returns:
            SLTPLevels con todos los niveles calculados
        """
        min_rr = min_rr or self.min_rr

        # Intentar ATR-based (metodo principal)
        if "atr_14" in df.columns and not df["atr_14"].isna().all():
            levels = self._atr_based(direction, entry_price, df, min_rr)
            if levels.is_valid:
                return levels

        # Fallback: swing high/low
        levels = self._swing_based(direction, entry_price, df, min_rr)
        if levels.is_valid:
            return levels

        # Fallback final: porcentaje fijo
        return self._fixed_pct(direction, entry_price, sl_pct=0.015, min_rr=min_rr)

    # ── Metodo ATR ────────────────────────────────────────────────────────────

    def _atr_based(
        self,
        direction:   str,
        entry_price: float,
        df:          pd.DataFrame,
        min_rr:      float,
    ) -> SLTPLevels:
        """
        SL = entry +/- N * ATR
        TP se ajusta para garantizar R:R minimo.

        N se adapta a la volatilidad relativa:
          - ATR% < 1%:  N = 2.0  (poco volatil, SL mas amplio)
          - ATR% 1-3%:  N = 1.5  (normal)
          - ATR% > 3%:  N = 1.0  (muy volatil, SL mas ajustado)
        """
        atr        = float(df["atr_14"].iloc[-1])
        atr_pct    = atr / entry_price

        # Ajustar multiplicador segun volatilidad relativa
        if atr_pct < 0.01:
            n_sl = 2.0
        elif atr_pct < 0.03:
            n_sl = self.atr_sl_mult
        else:
            n_sl = 1.0

        sl_distance = n_sl * atr
        sl_distance = max(sl_distance, entry_price * self.min_sl_pct)
        sl_distance = min(sl_distance, entry_price * self.max_sl_pct)

        # TP para garantizar R:R minimo
        tp_distance = max(sl_distance * min_rr, sl_distance * self.atr_tp_mult / self.atr_sl_mult)

        if direction == "long":
            stop_loss   = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss   = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        rr = tp_distance / sl_distance if sl_distance > 0 else 0

        return SLTPLevels(
            stop_loss=round(stop_loss, 8),
            take_profit=round(take_profit, 8),
            risk_reward=round(rr, 2),
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            sl_pct=sl_distance / entry_price,
            tp_pct=tp_distance / entry_price,
            method="atr",
        )

    # ── Metodo Swing ──────────────────────────────────────────────────────────

    def _swing_based(
        self,
        direction:   str,
        entry_price: float,
        df:          pd.DataFrame,
        min_rr:      float,
        lookback:    int = 14,
    ) -> SLTPLevels:
        """
        SL justo debajo del swing low reciente (long) o
        encima del swing high reciente (short).
        """
        recent = df.tail(lookback)

        if direction == "long":
            swing_sl    = float(recent["low"].min())
            sl_distance = max(entry_price - swing_sl, entry_price * self.min_sl_pct)
            sl_distance = min(sl_distance, entry_price * self.max_sl_pct)
            stop_loss   = entry_price - sl_distance
            tp_distance = sl_distance * min_rr
            take_profit = entry_price + tp_distance
        else:
            swing_sl    = float(recent["high"].max())
            sl_distance = max(swing_sl - entry_price, entry_price * self.min_sl_pct)
            sl_distance = min(sl_distance, entry_price * self.max_sl_pct)
            stop_loss   = entry_price + sl_distance
            tp_distance = sl_distance * min_rr
            take_profit = entry_price - tp_distance

        rr = tp_distance / sl_distance if sl_distance > 0 else 0

        return SLTPLevels(
            stop_loss=round(stop_loss, 8),
            take_profit=round(take_profit, 8),
            risk_reward=round(rr, 2),
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            sl_pct=sl_distance / entry_price,
            tp_pct=tp_distance / entry_price,
            method="swing",
        )

    # ── Metodo fijo ───────────────────────────────────────────────────────────

    def _fixed_pct(
        self,
        direction:   str,
        entry_price: float,
        sl_pct:      float = 0.015,
        min_rr:      float = 1.5,
    ) -> SLTPLevels:
        """Fallback: SL/TP como porcentaje fijo del precio."""
        sl_distance = entry_price * sl_pct
        tp_distance = sl_distance * min_rr

        if direction == "long":
            stop_loss   = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss   = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        return SLTPLevels(
            stop_loss=round(stop_loss, 8),
            take_profit=round(take_profit, 8),
            risk_reward=min_rr,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            sl_pct=sl_pct,
            tp_pct=sl_pct * min_rr,
            method="fixed",
        )

    # ── Trailing Stop ─────────────────────────────────────────────────────────

    def update_trailing_stop(
        self,
        direction:     str,
        current_price: float,
        current_sl:    float,
        entry_price:   float,
        atr:           float,
        trail_atr_mult: float = 1.2,
    ) -> float:
        """
        Actualiza el trailing stop para una posicion abierta.
        Solo mueve el SL en la direccion favorable (nunca hacia atras).

        Args:
            direction:      "long" o "short"
            current_price:  Precio actual
            current_sl:     SL actual
            entry_price:    Precio de entrada original
            atr:            ATR actual
            trail_atr_mult: Multiplicador ATR para el trailing

        Returns:
            Nuevo precio de SL (puede ser igual al actual si no mejora)
        """
        trail_distance = atr * trail_atr_mult

        if direction == "long":
            new_sl = current_price - trail_distance
            # Solo avanzar el SL, nunca retroceder
            new_sl = max(new_sl, current_sl)
        else:
            new_sl = current_price + trail_distance
            new_sl = min(new_sl, current_sl)

        if new_sl != current_sl:
            log.debug(
                f"Trailing stop actualizado: "
                f"{current_sl:.4f} -> {new_sl:.4f} "
                f"(precio: {current_price:.4f})"
            )

        return round(new_sl, 8)
