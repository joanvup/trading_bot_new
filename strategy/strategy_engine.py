"""
=============================================================
strategy/strategy_engine.py
Motor de estrategia — convierte señales IA en decisiones de trading.

Combina:
  - Señal del modelo ML (BUY/SELL/HOLD + confianza)
  - Filtros de riesgo (RiskManager)
  - Calculo de SL/TP (SLTPCalculator)
  - Sizing de posicion (Kelly)
  - Confirmaciones adicionales (tendencia, sesion, volumen)

Produce un TradeDecision listo para enviar al ejecutor (Fase 5).
=============================================================
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import pandas as pd

from ai.predictor import SignalResult
from risk.risk_manager import RiskManager, PositionSize
from risk.sl_tp_calculator import SLTPCalculator, SLTPLevels
from config.settings import Settings
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TradeDecision:
    """
    Decision de trading completa lista para ejecutar.
    Si action == "SKIP", no se ejecuta nada.
    """
    action:     str       # "ENTER_LONG" | "ENTER_SHORT" | "SKIP"
    symbol:     str
    direction:  str       # "long" | "short" | ""

    # Precios
    entry_price:   float = 0.0
    stop_loss:     float = 0.0
    take_profit:   float = 0.0

    # Sizing
    quantity:      Decimal = Decimal("0")
    leverage:      int     = 1
    margin_used:   float   = 0.0
    risk_amount:   float   = 0.0
    risk_pct:      float   = 0.0

    # Metadata
    signal_confidence: float = 0.0
    skip_reason:       str   = ""
    filters_passed:    list  = field(default_factory=list)
    filters_failed:    list  = field(default_factory=list)
    risk_reward:       float = 0.0
    timestamp:         datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_actionable(self) -> bool:
        return self.action in ("ENTER_LONG", "ENTER_SHORT")

    def __repr__(self) -> str:
        if not self.is_actionable:
            return f"<TradeDecision SKIP: {self.skip_reason[:40]}>"
        return (
            f"<TradeDecision {self.action} {self.symbol} "
            f"qty={self.quantity} lev={self.leverage}x "
            f"SL={self.stop_loss:.4f} TP={self.take_profit:.4f} "
            f"conf={self.signal_confidence:.2%}>"
        )


class StrategyEngine:
    """
    Motor de estrategia principal.

    Toma una señal del predictor IA y decide si operar,
    con que tamaño, SL y TP. Aplica multiples filtros de
    confirmacion para reducir falsos positivos.

    Uso:
        engine = StrategyEngine(settings, risk_manager, sl_tp_calc)

        decision = engine.evaluate(
            signal=signal_result,      # De RealTimePredictor
            df=df_with_indicators,     # DataFrame 15m con TA
            df_htf=df_1h,             # DataFrame 1h para confirmacion
            current_price=50000,
            symbol_info={"step_size": Decimal("0.001")},
        )

        if decision.is_actionable:
            # Enviar a ejecutor...
    """

    # Umbrales de filtros — los que no vienen de settings son fijos por diseño
    MIN_ADX_FOR_TREND    = 20.0
    MIN_VOLUME_RATIO     = 0.8
    MAX_BB_POSITION_LONG = 0.85
    MIN_BB_POSITION_LONG = 0.15
    MAX_RSI_LONG         = 75.0
    MIN_RSI_SHORT        = 25.0
    MAX_FUNDING_RATE_AGAINST = 0.0005

    def __init__(
        self,
        settings:   Settings,
        risk_mgr:   RiskManager,
        sl_tp_calc: SLTPCalculator,
        leverage:   int = 3,
        client=None,
    ):
        self.settings       = settings
        self.risk_mgr       = risk_mgr
        self.sl_tp_calc     = sl_tp_calc
        self.leverage       = min(leverage, settings.max_leverage)
        self.client         = client
        self.min_confidence = settings.min_signal_confidence

    # ── Evaluacion principal ──────────────────────────────────────────────────

    def evaluate(
        self,
        signal:        SignalResult,
        df:            pd.DataFrame,     # Timeframe primario (15m)
        df_htf:        Optional[pd.DataFrame] = None,  # Timeframe alto (1h)
        current_price: Optional[float] = None,
        symbol_info:   Optional[dict]  = None,
    ) -> TradeDecision:
        """
        Evalua si ejecutar la señal del modelo IA.

        Args:
            signal:        Resultado del predictor IA
            df:            DataFrame con indicadores del TF primario
            df_htf:        DataFrame del TF superior (confirmacion)
            current_price: Precio actual (usa el close del df si None)
            symbol_info:   Info del par (step_size, min_qty, etc.)

        Returns:
            TradeDecision con action="ENTER_LONG/SHORT" o "SKIP"
        """
        filters_passed = []
        filters_failed = []

        # ── Filtro 0: Señal basica ─────────────────────────────────────────────
        if not signal.is_actionable:
            return self._skip(signal.symbol, "Señal es HOLD")

        if signal.confidence < self.min_confidence:
            return self._skip(
                signal.symbol,
                f"Confianza insuficiente: {signal.confidence:.2%} < {self.min_confidence:.2%}"
            )
        filters_passed.append("confianza_ia")

        direction   = "long" if signal.signal == "BUY" else "short"
        entry_price = current_price or float(df["close"].iloc[-1])

        # ── Filtro 1: Direccion del ranking (AssetSelector) ────────────────────
        # Si el AssetSelector identificó que el activo tiene tendencia contraria
        # a la señal del modelo, rechazar — el mercado ya tiene dirección clara
        if symbol_info and "ranking_direction" in symbol_info:
            ranking_dir = symbol_info["ranking_direction"]  # "long", "short", "neutral"
            if ranking_dir != "neutral" and ranking_dir != direction:
                return self._skip(
                    signal.symbol,
                    f"Direccion del ranking ({ranking_dir}) opuesta a la señal ({direction})"
                )
            filters_passed.append(f"ranking_dir_ok({ranking_dir})")
        # Rechazar si el funding rate va en contra de la direccion del trade
        if self.client and not self.client.is_dry_run:
            try:
                fr_data = self.client.get_funding_rate(signal.symbol)
                rate    = fr_data["funding_rate"]
                next_ts = fr_data["next_funding_time"]

                # Determinar si el rate va en contra
                # LONG paga si rate > 0, SHORT paga si rate < 0
                rate_against = rate if direction == "long" else -rate

                if rate_against > self.MAX_FUNDING_RATE_AGAINST:
                    import datetime as _dt
                    next_dt = _dt.datetime.fromtimestamp(next_ts / 1000).strftime("%H:%M")
                    return self._skip(
                        signal.symbol,
                        f"Funding rate desfavorable para {direction}: "
                        f"{rate*100:.4f}%/8h (max {self.MAX_FUNDING_RATE_AGAINST*100:.4f}%) "
                        f"proximo cobro: {next_dt}"
                    )
                filters_passed.append(f"funding_ok({rate*100:.4f}%)")
            except Exception as e:
                log.debug(f"No se pudo verificar funding rate {signal.symbol}: {e}")
                filters_passed.append("funding_skip")
        if df_htf is not None and not df_htf.empty:
            htf_ok = self._check_htf_alignment(direction, df_htf)
            if htf_ok:
                filters_passed.append("htf_alignment")
            else:
                filters_failed.append("htf_alignment")
                # HTF es recomendacion, no bloqueo duro (solo reduce confianza)
                log.debug(f"HTF no alineado para {signal.symbol} {direction}")

        # ── Filtro 2: RSI no extremo ───────────────────────────────────────────
        if "rsi_14" in df.columns:
            rsi = float(df["rsi_14"].iloc[-1])
            rsi_ok = self._check_rsi(direction, rsi)
            if rsi_ok:
                filters_passed.append("rsi_ok")
            else:
                filters_failed.append("rsi_extremo")
                return self._skip(
                    signal.symbol,
                    f"RSI extremo para {direction}: {rsi:.1f}"
                )

        # ── Filtro 3: Volumen suficiente ───────────────────────────────────────
        if "vol_ratio" in df.columns:
            vol_ratio = float(df["vol_ratio"].iloc[-1])
            if vol_ratio >= self.MIN_VOLUME_RATIO:
                filters_passed.append("volumen_ok")
            else:
                filters_failed.append("volumen_bajo")
                return self._skip(
                    signal.symbol,
                    f"Volumen bajo: ratio={vol_ratio:.2f} < {self.MIN_VOLUME_RATIO}"
                )

        # ── Filtro 4: Bollinger Bands position ────────────────────────────────
        if "bb_position" in df.columns:
            bb_pos = float(df["bb_position"].iloc[-1])
            bb_ok  = self._check_bb_position(direction, bb_pos)
            if bb_ok:
                filters_passed.append("bb_position_ok")
            else:
                filters_failed.append("bb_position_adverso")
                return self._skip(
                    signal.symbol,
                    f"BB position adversa para {direction}: {bb_pos:.2f}"
                )

        # ── Filtro 5: Risk manager check ───────────────────────────────────────
        if self.risk_mgr.state.is_paused:
            return self._skip(
                signal.symbol,
                f"RiskManager pausado: {self.risk_mgr.state.pause_reason}"
            )
        filters_passed.append("risk_gate")

        # ── Calcular SL/TP ────────────────────────────────────────────────────
        levels = self.sl_tp_calc.calculate(
            direction=direction,
            entry_price=entry_price,
            df=df,
            min_rr=1.5,
        )

        if not levels.is_valid:
            return self._skip(
                signal.symbol,
                f"SL/TP invalidos: R:R={levels.risk_reward:.2f}"
            )
        filters_passed.append("sl_tp_valid")

        # ── Calcular tamaño de posicion ────────────────────────────────────────
        step_size = None
        if symbol_info and "step_size" in symbol_info:
            step_size = symbol_info["step_size"]

        size = self.risk_mgr.calculate_position_size(
            signal_confidence=signal.confidence,
            entry_price=entry_price,
            stop_loss_price=levels.stop_loss,
            leverage=self.leverage,
            symbol_step_size=step_size,
        )

        if not size.approved:
            return self._skip(signal.symbol, f"Sizing rechazado: {size.reject_reason}")
        filters_passed.append("position_sizing")

        # ── Decision final ────────────────────────────────────────────────────
        action = "ENTER_LONG" if direction == "long" else "ENTER_SHORT"

        decision = TradeDecision(
            action=action,
            symbol=signal.symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=levels.stop_loss,
            take_profit=levels.take_profit,
            quantity=size.quantity,
            leverage=self.leverage,
            margin_used=size.margin_used,
            risk_amount=size.risk_amount,
            risk_pct=size.risk_pct,
            signal_confidence=signal.confidence,
            filters_passed=filters_passed,
            filters_failed=filters_failed,
            risk_reward=levels.risk_reward,
        )

        log.info(
            f"Decision: {action} {signal.symbol} | "
            f"qty={size.quantity} | lev={self.leverage}x | "
            f"SL={levels.stop_loss:.4f} | TP={levels.take_profit:.4f} | "
            f"R:R={levels.risk_reward:.2f} | "
            f"Riesgo=${size.risk_amount:.2f} ({size.risk_pct*100:.2f}%) | "
            f"Filtros: {len(filters_passed)} OK / {len(filters_failed)} WARN"
        )

        return decision

    # ── Filtros individuales ──────────────────────────────────────────────────

    def _check_htf_alignment(self, direction: str, df_htf: pd.DataFrame) -> bool:
        """Confirma que el TF superior apoya la direccion del trade."""
        if df_htf.empty:
            return True  # Sin datos = no bloquear

        last = df_htf.iloc[-1]

        if direction == "long":
            # Para long: EMA9 > EMA21 en HTF y precio sobre EMA50
            ema9_gt_ema21 = last.get("ema_9", 0) > last.get("ema_21", 0)
            return bool(ema9_gt_ema21)
        else:
            # Para short: EMA9 < EMA21 en HTF
            ema9_lt_ema21 = last.get("ema_9", float("inf")) < last.get("ema_21", float("inf"))
            return bool(ema9_lt_ema21)

    def _check_rsi(self, direction: str, rsi: float) -> bool:
        """Rechaza entradas en condiciones extremas de RSI."""
        if direction == "long" and rsi > self.MAX_RSI_LONG:
            return False
        if direction == "short" and rsi < self.MIN_RSI_SHORT:
            return False
        return True

    def _check_bb_position(self, direction: str, bb_pos: float) -> bool:
        """
        Verifica que la posicion en Bollinger Bands sea favorable.
        bb_pos: 0=en banda baja, 1=en banda alta
        """
        if direction == "long" and bb_pos > self.MAX_BB_POSITION_LONG:
            return False  # Precio en parte alta de BB, no comprar
        if direction == "short" and bb_pos < (1.0 - self.MAX_BB_POSITION_LONG):
            return False  # Precio en parte baja de BB, no vender
        return True

    # ── Utilidades ─────────────────────────────────────────────────────────────

    def _skip(self, symbol: str, reason: str) -> TradeDecision:
        log.debug(f"SKIP {symbol}: {reason}")
        return TradeDecision(
            action="SKIP",
            symbol=symbol,
            direction="",
            skip_reason=reason,
        )
