"""
=============================================================
config/settings.py
Sistema de configuración centralizado con validación Pydantic.
Carga variables desde .env y las valida automáticamente.
=============================================================
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Directorio raíz del proyecto ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent


# ── Enumeraciones ─────────────────────────────────────────────────────────────

class TradingMode(str, Enum):
    TESTNET = "testnet"
    DRY_RUN = "dry_run"
    LIVE = "live"


class Timeframe(str, Enum):
    M1  = "1m"
    M3  = "3m"
    M5  = "5m"
    M15 = "15m"
    H1  = "1h"
    H4  = "4h"
    D1  = "1d"


# ── Configuración principal ────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    Todas las configuraciones del bot.
    Se cargan automáticamente desde el archivo .env en el directorio raíz.
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Modo de operación ──────────────────────────────────────────────────────
    trading_mode: TradingMode = TradingMode.DRY_RUN

    # ── Binance API ────────────────────────────────────────────────────────────
    binance_testnet_api_key: str = ""
    binance_testnet_api_secret: str = ""
    binance_live_api_key: str = ""
    binance_live_api_secret: str = ""

    # ── Base de datos ──────────────────────────────────────────────────────────
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "trading_bot"
    db_user: str = "postgres"
    db_password: str = ""

    # ── Capital y riesgo ───────────────────────────────────────────────────────
    initial_capital: float = Field(default=1000.0, gt=0)
    max_risk_per_trade: float = Field(default=0.02, gt=0, le=0.1)
    max_drawdown: float = Field(default=0.15, gt=0, lt=1.0)
    max_open_positions: int = Field(default=3, ge=1, le=10)
    max_leverage: int = Field(default=5, ge=1, le=20)

    # ── Selección de activos ───────────────────────────────────────────────────
    top_symbols_count: int = Field(default=20, ge=5, le=100)
    primary_timeframe: Timeframe = Timeframe.M15
    secondary_timeframe: Timeframe = Timeframe.H1

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = "logs/trading_bot.log"
    log_rotation: str = "10 MB"
    log_retention: str = "30 days"

    # ── API REST ───────────────────────────────────────────────────────────────
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_secret_key: str = "change-this-in-production"

    # ── Alertas ────────────────────────────────────────────────────────────────
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    # ── Modelos IA ─────────────────────────────────────────────────────────────
    models_dir: str = "models/saved"
    retrain_interval_hours: int = Field(default=24, ge=1)

    # ── Estrategia ────────────────────────────────────────────────────────────
    # Confianza minima del modelo IA para considerar una señal accionable
    # 0.57 = 57% — por debajo de esto la señal se descarta como HOLD
    min_signal_confidence: float = Field(default=0.57, ge=0.5, le=0.95)
    sl_atr_multiplier:    float = Field(default=1.5,  gt=0)  # SL a N*ATR del precio
    tp_atr_multiplier:    float = Field(default=3.0,  gt=0)  # TP a N*ATR del precio
    min_risk_reward:      float = Field(default=1.5,  gt=0)  # R:R minimo
    min_sl_pct:           float = Field(default=0.003, gt=0) # SL minimo 0.3%
    max_sl_pct:           float = Field(default=0.05,  gt=0) # SL maximo 5%
    trail_atr_multiplier: float = Field(default=1.2,  gt=0)  # Trailing antes del BE

    # Trailing DESPUES del breakeven — mas agresivo para no devolver lo ganado
    # 0.5 = persigue a 0.5x ATR del precio (muy ajustado)
    # 0.8 = persigue a 0.8x ATR (moderado, recomendado)
    # 1.2 = mismo que antes del BE (suave)
    trail_atr_after_be: float = Field(default=0.8, gt=0)

    # Activar trailing cuando el precio se haya movido N*ATR a favor (0 = siempre activo)
    trail_activation_atr: float = Field(default=0.5, ge=0)   # 0.5x ATR para activar TS

    # Mover SL a breakeven cuando el precio se haya movido N*ATR a favor (0 = desactivado)
    breakeven_atr: float = Field(default=1.0, ge=0)          # 1.0x ATR para breakeven

    # ── Propiedades calculadas ─────────────────────────────────────────────────

    @property
    def database_url(self) -> str:
        """URL de conexión a PostgreSQL (síncrona)."""
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def async_database_url(self) -> str:
        """URL de conexión a PostgreSQL (asíncrona con asyncpg)."""
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def is_testnet(self) -> bool:
        return self.trading_mode == TradingMode.TESTNET

    @property
    def is_dry_run(self) -> bool:
        return self.trading_mode == TradingMode.DRY_RUN

    @property
    def is_live(self) -> bool:
        return self.trading_mode == TradingMode.LIVE

    @property
    def active_api_key(self) -> str:
        """Retorna la API key correcta según el modo activo."""
        if self.is_testnet:
            return self.binance_testnet_api_key
        return self.binance_live_api_key

    @property
    def active_api_secret(self) -> str:
        """Retorna el API secret correcto según el modo activo."""
        if self.is_testnet:
            return self.binance_testnet_api_secret
        return self.binance_live_api_secret

    # ── Validaciones ───────────────────────────────────────────────────────────

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in valid:
            raise ValueError(f"Log level debe ser uno de: {valid}")
        return v

    @model_validator(mode="after")
    def validate_api_keys_for_mode(self) -> "Settings":
        """Valida que las API keys existan según el modo seleccionado."""
        if self.trading_mode == TradingMode.TESTNET:
            if not self.binance_testnet_api_key or not self.binance_testnet_api_secret:
                raise ValueError(
                    "Se requieren BINANCE_TESTNET_API_KEY y "
                    "BINANCE_TESTNET_API_SECRET para el modo testnet"
                )
        elif self.trading_mode == TradingMode.LIVE:
            if not self.binance_live_api_key or not self.binance_live_api_secret:
                raise ValueError(
                    "Se requieren BINANCE_LIVE_API_KEY y "
                    "BINANCE_LIVE_API_SECRET para el modo live"
                )
        return self

    def get_models_path(self) -> Path:
        """Retorna la ruta absoluta del directorio de modelos."""
        path = BASE_DIR / self.models_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_logs_path(self) -> Path:
        """Retorna la ruta absoluta del directorio de logs."""
        path = BASE_DIR / Path(self.log_file).parent
        path.mkdir(parents=True, exist_ok=True)
        return path


# ── Singleton con caché ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Retorna la instancia única de configuración.
    Usa lru_cache para que solo se cargue una vez.

    Uso:
        from config.settings import get_settings
        settings = get_settings()
    """
    return Settings()
