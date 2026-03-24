"""
=============================================================
risk/risk_manager.py
Gestor de riesgo avanzado.

Implementa:
  - Kelly Criterion fraccionado para sizing optimo
  - Control dinamico de drawdown con pausa automatica
  - Proteccion anti-ruina (nunca arriesgar mas de X% por trade)
  - Reduccion gradual de capital en rachas perdedoras
  - Portfolio heat (riesgo total abierto en todo momento)
=============================================================
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from config.settings import Settings
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RiskState:
    """Estado actual del riesgo del portfolio."""
    total_balance:      float = 0.0
    available_balance:  float = 0.0
    peak_balance:       float = 0.0
    current_drawdown:   float = 0.0   # 0.0 - 1.0
    open_positions:     int   = 0
    portfolio_heat:     float = 0.0   # % capital en riesgo ahora mismo
    is_paused:          bool  = False
    pause_reason:       str   = ""
    consecutive_losses: int   = 0
    win_streak:         int   = 0
    total_trades_today: int   = 0
    daily_pnl:          float = 0.0

    # Historial reciente de trades (True=win, False=loss)
    recent_results: list = field(default_factory=list)

    def update_peak(self) -> None:
        if self.total_balance > self.peak_balance:
            self.peak_balance = self.total_balance

    def calc_drawdown(self) -> float:
        if self.peak_balance <= 0:
            return 0.0
        self.current_drawdown = max(
            0.0,
            (self.peak_balance - self.total_balance) / self.peak_balance
        )
        return self.current_drawdown

    def record_trade_result(self, won: bool, pnl: float) -> None:
        self.recent_results.append(won)
        if len(self.recent_results) > 20:
            self.recent_results.pop(0)
        self.daily_pnl += pnl
        self.total_trades_today += 1
        if won:
            self.consecutive_losses = 0
            self.win_streak += 1
        else:
            self.consecutive_losses += 1
            self.win_streak = 0


@dataclass
class PositionSize:
    """Resultado del calculo de tamaño de posicion."""
    quantity:       Decimal   # Cantidad del activo base
    notional_usdt:  float     # Valor en USDT sin apalancamiento
    margin_used:    float     # Margen real usado (notional / leverage)
    risk_amount:    float     # USDT en riesgo (distancia al SL)
    leverage:       int
    risk_pct:       float     # % del capital total en riesgo
    kelly_fraction: float     # Fraccion Kelly calculada
    approved:       bool = True
    reject_reason:  str  = ""


class RiskManager:
    """
    Gestor centralizado de riesgo del portfolio.

    Decide cuanto capital arriesgar en cada trade usando
    Kelly Criterion fraccionado, ajustado por drawdown actual,
    confianza del modelo IA y racha de resultados recientes.

    Uso:
        rm = RiskManager(settings)
        rm.update_balance(balance=1000, available=850)

        # Calcular tamaño de posicion
        size = rm.calculate_position_size(
            signal_confidence=0.72,
            entry_price=50000,
            stop_loss_price=49500,
            leverage=3,
        )
        if size.approved:
            print(f"Cantidad: {size.quantity} BTC")
            print(f"Riesgo:   ${size.risk_amount:.2f}")
    """

    # Fraccion de Kelly a usar (Kelly completo es muy agresivo)
    KELLY_FRACTION = 0.25   # 1/4 Kelly = conservador pero optimo a largo plazo

    # Limites de riesgo absolutos
    MAX_RISK_PER_TRADE_ABS = 0.05    # Nunca mas del 5% por trade (tope duro)
    MAX_PORTFOLIO_HEAT     = 0.12    # Maximo 12% del capital en riesgo simultaneo
    MAX_DAILY_LOSS_PCT     = 0.08    # Pausa si perdemos mas del 8% en un dia

    def __init__(self, settings: Settings):
        self.settings = settings
        self.state    = RiskState()

        # Configuracion desde settings
        self.max_risk_per_trade = settings.max_risk_per_trade
        self.max_drawdown       = settings.max_drawdown
        self.max_positions      = settings.max_open_positions
        self.max_leverage       = settings.max_leverage

        log.info(
            f"RiskManager iniciado | "
            f"Riesgo/trade: {self.max_risk_per_trade*100:.1f}% | "
            f"Drawdown max: {self.max_drawdown*100:.1f}% | "
            f"Max posiciones: {self.max_positions}"
        )

    # ── Actualizacion de estado ────────────────────────────────────────────────

    def update_balance(
        self,
        total_balance:     float,
        available_balance: float,
        unrealized_pnl:    float = 0.0,
    ) -> None:
        """Actualiza el estado del balance y calcula drawdown."""
        self.state.total_balance     = total_balance
        self.state.available_balance = available_balance
        self.state.update_peak()
        self.state.calc_drawdown()

        # Verificar si hay que pausar
        self._check_pause_conditions()

        log.debug(
            f"Balance: ${total_balance:,.2f} | "
            f"DD: {self.state.current_drawdown*100:.2f}% | "
            f"Paused: {self.state.is_paused}"
        )

    def update_open_positions(self, count: int, heat: float = 0.0) -> None:
        """Actualiza numero de posiciones abiertas y heat."""
        self.state.open_positions = count
        self.state.portfolio_heat = heat

    def record_trade_closed(self, won: bool, pnl: float) -> None:
        """Registra el resultado de un trade cerrado."""
        self.state.record_trade_result(won, pnl)
        log.info(
            f"Trade cerrado | {'GANADO' if won else 'PERDIDO'} | "
            f"PnL: ${pnl:+.2f} | "
            f"Racha perdedora: {self.state.consecutive_losses}"
        )

    # ── Calculo de tamaño de posicion ─────────────────────────────────────────

    def calculate_position_size(
        self,
        signal_confidence: float,
        entry_price:       float,
        stop_loss_price:   float,
        leverage:          int,
        symbol_step_size:  Optional[Decimal] = None,
    ) -> PositionSize:
        """
        Calcula el tamaño optimo de la posicion con Kelly fraccionado.

        Algoritmo:
        1. Kelly fraction = f* = (edge * odds - (1-edge)) / odds
           donde edge = win_rate historica, odds = avg_win/avg_loss
        2. Ajustar Kelly por confianza IA, drawdown actual y racha
        3. Calcular USDT en riesgo = balance * kelly_ajustado
        4. Calcular cantidad = risk_USDT / distancia_SL

        Args:
            signal_confidence: Confianza del modelo IA (0-1)
            entry_price:       Precio de entrada
            stop_loss_price:   Precio del stop loss
            leverage:          Apalancamiento a usar
            symbol_step_size:  Precision minima del par (ej. 0.001 BTC)

        Returns:
            PositionSize con todos los calculos
        """
        # ── Validaciones previas ───────────────────────────────────────────────

        if self.state.is_paused:
            return self._reject(f"Trading pausado: {self.state.pause_reason}")

        if self.state.open_positions >= self.max_positions:
            return self._reject(
                f"Max posiciones alcanzado: {self.state.open_positions}/{self.max_positions}"
            )

        if self.state.portfolio_heat >= self.MAX_PORTFOLIO_HEAT:
            return self._reject(
                f"Portfolio heat muy alto: {self.state.portfolio_heat*100:.1f}%"
            )

        balance = self.state.available_balance
        if balance <= 0:
            return self._reject("Balance disponible = 0")

        # ── Kelly Criterion ───────────────────────────────────────────────────

        kelly_raw = self._calc_kelly_fraction(signal_confidence)

        # Ajustes dinamicos al Kelly
        kelly_adjusted = kelly_raw

        # 1. Reducir por drawdown actual (mas DD = menos tamaño)
        if self.state.current_drawdown > 0.05:
            dd_factor = 1.0 - (self.state.current_drawdown / self.max_drawdown) * 0.5
            kelly_adjusted *= max(dd_factor, 0.3)

        # 2. Reducir por racha de perdidas
        if self.state.consecutive_losses >= 3:
            loss_factor = max(0.4, 1.0 - self.state.consecutive_losses * 0.1)
            kelly_adjusted *= loss_factor

        # 3. Aumentar ligeramente en rachas ganadoras (maximo 1.2x)
        if self.state.win_streak >= 3:
            kelly_adjusted *= min(1.2, 1.0 + self.state.win_streak * 0.05)

        # 4. Aplicar limite absoluto anti-ruina
        kelly_adjusted = min(kelly_adjusted, self.MAX_RISK_PER_TRADE_ABS)
        kelly_adjusted = min(kelly_adjusted, self.max_risk_per_trade)

        # ── Calculo de cantidades ──────────────────────────────────────────────

        # Distancia al SL como % del precio
        sl_distance = abs(entry_price - stop_loss_price)
        sl_pct      = sl_distance / entry_price

        if sl_pct <= 0.0001:
            return self._reject("Stop loss demasiado cerca del precio de entrada")

        # USDT en riesgo = balance * fraccion Kelly ajustada
        risk_usdt = balance * kelly_adjusted

        # Notional (sin apalancamiento) = risk_usdt / sl_pct
        # Con apalancamiento: notional = risk_usdt / sl_pct * leverage
        # Pero el margen usado = notional / leverage
        notional_usdt = risk_usdt / sl_pct
        margin_used   = notional_usdt / leverage

        # Si el margen supera el disponible, reducir proporcionalmente
        if margin_used > balance * 0.9:
            factor        = (balance * 0.9) / margin_used
            margin_used   *= factor
            notional_usdt *= factor
            risk_usdt     *= factor

        # Cantidad del activo base
        quantity = Decimal(str(notional_usdt / entry_price))

        # Aplicar step_size si se provee
        if symbol_step_size and symbol_step_size > 0:
            quantity = (quantity // symbol_step_size) * symbol_step_size

        if quantity <= 0:
            return self._reject("Cantidad calculada <= 0")

        result = PositionSize(
            quantity=quantity,
            notional_usdt=notional_usdt,
            margin_used=margin_used,
            risk_amount=risk_usdt,
            leverage=leverage,
            risk_pct=kelly_adjusted,
            kelly_fraction=kelly_raw,
            approved=True,
        )

        log.info(
            f"Position size: qty={quantity} | "
            f"notional=${notional_usdt:.2f} | "
            f"margin=${margin_used:.2f} | "
            f"riesgo=${risk_usdt:.2f} ({kelly_adjusted*100:.2f}%) | "
            f"kelly_raw={kelly_raw:.4f}"
        )

        return result

    def _calc_kelly_fraction(self, confidence: float) -> float:
        """
        Calcula la fraccion Kelly basada en el historial de trades
        y la confianza actual del modelo IA.

        f* = (W * R - L) / R
        donde W = win_rate, L = loss_rate, R = win/loss ratio promedio
        """
        # Si tenemos historial, usar win_rate real
        recent = self.state.recent_results
        if len(recent) >= 10:
            win_rate  = sum(recent) / len(recent)
            loss_rate = 1.0 - win_rate
        else:
            # Sin historial, usar estimacion conservadora basada en confianza
            win_rate  = confidence * 0.7   # Descuento del 30% por incertidumbre
            loss_rate = 1.0 - win_rate

        # Ratio win/loss tipico en crypto futuros con buena estrategia
        avg_rr = 1.5  # Risk:Reward minimo

        # Kelly formula
        kelly = (win_rate * avg_rr - loss_rate) / avg_rr

        # Kelly fraccionado (1/4)
        kelly_fractional = max(0.0, kelly * self.KELLY_FRACTION)

        # Ajustar por confianza del modelo (mayor confianza = mas cerca del Kelly)
        kelly_confidence_adjusted = kelly_fractional * (0.5 + confidence * 0.5)

        # Aplicar limites
        return min(kelly_confidence_adjusted, self.max_risk_per_trade)

    # ── Pausas automaticas ─────────────────────────────────────────────────────

    def _check_pause_conditions(self) -> None:
        """Verifica y aplica condiciones de pausa automatica."""

        # 1. Drawdown maximo alcanzado
        if self.state.current_drawdown >= self.max_drawdown:
            self._pause(
                f"Drawdown maximo alcanzado: "
                f"{self.state.current_drawdown*100:.1f}% >= {self.max_drawdown*100:.1f}%"
            )
            return

        # 2. Perdida diaria excesiva
        daily_loss_pct = abs(self.state.daily_pnl) / max(self.state.total_balance, 1)
        if self.state.daily_pnl < 0 and daily_loss_pct >= self.MAX_DAILY_LOSS_PCT:
            self._pause(
                f"Perdida diaria excesiva: "
                f"{daily_loss_pct*100:.1f}% >= {self.MAX_DAILY_LOSS_PCT*100:.1f}%"
            )
            return

        # 3. Racha perdedora extrema
        if self.state.consecutive_losses >= 6:
            self._pause(
                f"Racha perdedora extrema: {self.state.consecutive_losses} seguidas"
            )
            return

        # Reanudar si la condicion ya no aplica
        if self.state.is_paused and self.state.current_drawdown < self.max_drawdown * 0.85:
            self._resume()

    def _pause(self, reason: str) -> None:
        if not self.state.is_paused:
            self.state.is_paused    = True
            self.state.pause_reason = reason
            log.warning(f"TRADING PAUSADO: {reason}")

    def _resume(self) -> None:
        if self.state.is_paused:
            log.info(f"Trading reanudado | Drawdown: {self.state.current_drawdown*100:.1f}%")
            self.state.is_paused    = False
            self.state.pause_reason = ""

    def reset_daily_stats(self) -> None:
        """Resetear estadisticas diarias (llamar al inicio de cada dia)."""
        self.state.daily_pnl        = 0.0
        self.state.total_trades_today = 0
        if self.state.consecutive_losses < 6:
            self._resume()
        log.info("Estadisticas diarias reseteadas")

    # ── Utilidades ─────────────────────────────────────────────────────────────

    def _reject(self, reason: str) -> PositionSize:
        log.warning(f"Posicion rechazada: {reason}")
        return PositionSize(
            quantity=Decimal("0"),
            notional_usdt=0.0,
            margin_used=0.0,
            risk_amount=0.0,
            leverage=1,
            risk_pct=0.0,
            kelly_fraction=0.0,
            approved=False,
            reject_reason=reason,
        )

    def get_status(self) -> dict:
        return {
            "balance":            round(self.state.total_balance, 2),
            "available":          round(self.state.available_balance, 2),
            "peak_balance":       round(self.state.peak_balance, 2),
            "drawdown_pct":       round(self.state.current_drawdown * 100, 2),
            "is_paused":          self.state.is_paused,
            "pause_reason":       self.state.pause_reason,
            "open_positions":     self.state.open_positions,
            "portfolio_heat_pct": round(self.state.portfolio_heat * 100, 2),
            "consecutive_losses": self.state.consecutive_losses,
            "win_streak":         self.state.win_streak,
            "daily_pnl":          round(self.state.daily_pnl, 2),
            "trades_today":       self.state.total_trades_today,
        }
