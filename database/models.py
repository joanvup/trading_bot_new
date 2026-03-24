"""
=============================================================
database/models.py
Modelos de base de datos con SQLAlchemy 2.0.
Define todas las tablas necesarias para el bot de trading.
=============================================================
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from sqlalchemy import (
    BigInteger, Boolean, DateTime, ForeignKey, Index,
    Integer, Numeric, String, Text, UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ── Base declarativa ──────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """Clase base para todos los modelos."""
    pass


# ── Enumeraciones para columnas ────────────────────────────────────────────────

class TradeStatus(str, Enum):
    PENDING   = "pending"
    OPEN      = "open"
    CLOSED    = "closed"
    CANCELLED = "cancelled"
    ERROR     = "error"


class TradeDirection(str, Enum):
    LONG  = "long"
    SHORT = "short"


class SignalType(str, Enum):
    BUY   = "buy"
    SELL  = "sell"
    HOLD  = "hold"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT  = "limit"
    STOP   = "stop"


# ── Tabla: symbols (criptomonedas monitoreadas) ───────────────────────────────

class Symbol(Base):
    """Registro de criptomonedas monitoreadas por el bot."""
    __tablename__ = "symbols"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    base_asset: Mapped[str] = mapped_column(String(30), nullable=False)
    quote_asset: Mapped[str] = mapped_column(String(10), nullable=False, default="USDT")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    min_qty: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    step_size: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    tick_size: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    max_leverage: Mapped[int] = mapped_column(Integer, default=20)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relaciones
    trades: Mapped[list["Trade"]] = relationship("Trade", back_populates="symbol_ref")
    signals: Mapped[list["Signal"]] = relationship("Signal", back_populates="symbol_ref")
    candles: Mapped[list["Candle"]] = relationship("Candle", back_populates="symbol_ref")

    def __repr__(self) -> str:
        return f"<Symbol {self.symbol}>"


# ── Tabla: candles (datos OHLCV históricos) ───────────────────────────────────

class Candle(Base):
    """Datos OHLCV almacenados localmente para análisis y entrenamiento."""
    __tablename__ = "candles"
    __table_args__ = (
        UniqueConstraint("symbol_id", "timeframe", "open_time", name="uq_candle"),
        Index("ix_candle_symbol_tf_time", "symbol_id", "timeframe", "open_time"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False
    )
    timeframe: Mapped[str] = mapped_column(String(5), nullable=False)
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    close_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    quote_volume: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    trades_count: Mapped[int] = mapped_column(Integer, default=0)
    taker_buy_base_vol: Mapped[Decimal] = mapped_column(Numeric(30, 8), default=0)
    taker_buy_quote_vol: Mapped[Decimal] = mapped_column(Numeric(30, 8), default=0)
    is_closed: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relación
    symbol_ref: Mapped["Symbol"] = relationship("Symbol", back_populates="candles")

    def __repr__(self) -> str:
        return f"<Candle {self.symbol_id} {self.timeframe} {self.open_time}>"


# ── Tabla: signals (señales generadas por IA) ─────────────────────────────────

class Signal(Base):
    """Señales de trading generadas por el motor de IA."""
    __tablename__ = "signals"
    __table_args__ = (
        Index("ix_signal_symbol_time", "symbol_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False
    )
    signal_type: Mapped[str] = mapped_column(String(10), nullable=False)  # buy/sell/hold
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)  # 0.0 - 1.0
    price_at_signal: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    predicted_return: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    timeframe: Mapped[str] = mapped_column(String(5), nullable=False)

    # Indicadores técnicos en el momento de la señal (JSON)
    indicators: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)

    # Resultado real (se rellena después)
    actual_return: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    was_profitable: Mapped[Optional[bool]] = mapped_column(Boolean)

    model_version: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relaciones
    symbol_ref: Mapped["Symbol"] = relationship("Symbol", back_populates="signals")
    trade: Mapped[Optional["Trade"]] = relationship("Trade", back_populates="signal")

    def __repr__(self) -> str:
        return f"<Signal {self.signal_type} {self.confidence:.2%} @ {self.price_at_signal}>"


# ── Tabla: trades (operaciones ejecutadas) ────────────────────────────────────

class Trade(Base):
    """Registro completo de cada operación de trading."""
    __tablename__ = "trades"
    __table_args__ = (
        Index("ix_trade_symbol_status", "symbol_id", "status"),
        Index("ix_trade_opened_at", "opened_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("symbols.id", ondelete="CASCADE"), nullable=False
    )
    signal_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("signals.id", ondelete="SET NULL"), nullable=True
    )

    # Identificadores externos de Binance
    binance_order_id: Mapped[Optional[str]] = mapped_column(String(50))
    binance_client_order_id: Mapped[Optional[str]] = mapped_column(String(50))

    # Dirección y estado
    direction: Mapped[str] = mapped_column(String(10), nullable=False)  # long/short
    status: Mapped[str] = mapped_column(String(15), nullable=False, default="pending")
    order_type: Mapped[str] = mapped_column(String(10), nullable=False, default="market")

    # Precios
    entry_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    exit_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    stop_loss: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    take_profit: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    liquidation_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))

    # Tamaño y capital
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    leverage: Mapped[int] = mapped_column(Integer, default=1)
    notional_value: Mapped[Decimal] = mapped_column(Numeric(20, 4), nullable=False)
    margin_used: Mapped[Decimal] = mapped_column(Numeric(20, 4), nullable=False)

    # Resultado
    pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 4))
    pnl_pct: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    fees_paid: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=0)
    net_pnl: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 4))

    # Timestamps
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer)

    # Motivo de cierre
    close_reason: Mapped[Optional[str]] = mapped_column(String(50))  # sl_hit, tp_hit, manual, etc.

    # Datos extras en JSON
    trade_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)

    is_dry_run: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relaciones
    symbol_ref: Mapped["Symbol"] = relationship("Symbol", back_populates="trades")
    signal: Mapped[Optional["Signal"]] = relationship("Signal", back_populates="trade")

    def __repr__(self) -> str:
        return (
            f"<Trade {self.direction.upper()} {self.symbol_id} "
            f"@ {self.entry_price} [{self.status}]>"
        )


# ── Tabla: portfolio_snapshots (estado del capital en el tiempo) ───────────────

class PortfolioSnapshot(Base):
    """Instantáneas periódicas del estado del portafolio para métricas."""
    __tablename__ = "portfolio_snapshots"
    __table_args__ = (
        Index("ix_snapshot_timestamp", "timestamp"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    total_balance: Mapped[Decimal] = mapped_column(Numeric(20, 4), nullable=False)
    available_balance: Mapped[Decimal] = mapped_column(Numeric(20, 4), nullable=False)
    unrealized_pnl: Mapped[Decimal] = mapped_column(Numeric(20, 4), default=0)
    realized_pnl_day: Mapped[Decimal] = mapped_column(Numeric(20, 4), default=0)
    open_positions_count: Mapped[int] = mapped_column(Integer, default=0)
    peak_balance: Mapped[Decimal] = mapped_column(Numeric(20, 4), nullable=False)
    current_drawdown: Mapped[Decimal] = mapped_column(Numeric(10, 6), default=0)
    total_trades_day: Mapped[int] = mapped_column(Integer, default=0)
    win_rate_day: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))

    def __repr__(self) -> str:
        return f"<Snapshot {self.timestamp} balance={self.total_balance}>"


# ── Tabla: model_training_runs (historial de entrenamientos) ──────────────────

class ModelTrainingRun(Base):
    """Registro de cada entrenamiento del modelo de IA."""
    __tablename__ = "model_training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    training_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    test_samples: Mapped[int] = mapped_column(Integer, nullable=False)

    # Métricas del modelo
    train_accuracy: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))
    test_accuracy: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))
    precision: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))
    recall: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))
    f1_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))
    sharpe_ratio: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))

    # Hiperparámetros usados
    hyperparams: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)
    feature_importance: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)

    model_path: Mapped[Optional[str]] = mapped_column(String(255))
    is_deployed: Mapped[bool] = mapped_column(Boolean, default=False)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer)

    def __repr__(self) -> str:
        return f"<ModelTrainingRun {self.model_name} v{self.version}>"
