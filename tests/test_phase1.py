"""
=============================================================
tests/test_phase1.py
Tests básicos para verificar que la Fase 1 funciona.

Uso:
    pytest tests/test_phase1.py -v
    python tests/test_phase1.py  (ejecución directa)

=============================================================
"""

import sys
from pathlib import Path
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_settings_load():
    """Verifica que la configuración carga sin errores."""
    from config.settings import get_settings, TradingMode
    get_settings.cache_clear()

    import os
    os.environ.setdefault("TRADING_MODE", "dry_run")
    os.environ.setdefault("INITIAL_CAPITAL", "1000")
    os.environ.setdefault("DB_HOST", "localhost")
    os.environ.setdefault("DB_NAME", "trading_bot")
    os.environ.setdefault("DB_USER", "postgres")
    os.environ.setdefault("DB_PASSWORD", "test")

    settings = get_settings()
    assert settings is not None
    assert settings.initial_capital > 0
    assert settings.max_risk_per_trade > 0
    assert settings.max_risk_per_trade <= 0.1
    print("✓ Configuración cargada correctamente")


def test_dry_run_client():
    """Verifica que el cliente funciona en modo dry_run."""
    import os
    os.environ["TRADING_MODE"] = "dry_run"

    from config.settings import get_settings, TradingMode
    get_settings.cache_clear()
    settings = get_settings()

    from core.binance_client import BinanceFuturesClient
    client = BinanceFuturesClient(settings)
    client.initialize()

    assert client.is_ready
    balance = client.get_account_balance()
    assert float(balance["availableBalance"]) == settings.initial_capital

    # Test orden simulada
    from decimal import Decimal
    order = client.place_market_order(
        symbol="BTCUSDT",
        side="BUY",
        quantity=Decimal("0.001"),
    )
    assert order is not None
    assert "orderId" in order
    assert order["orderId"].startswith("DRY_")
    print("✓ Cliente Binance (dry_run) funcionando correctamente")


def test_database_models():
    """Verifica que los modelos de BD se pueden importar."""
    from database.models import (
        Symbol, Candle, Signal, Trade,
        PortfolioSnapshot, ModelTrainingRun,
        TradeStatus, TradeDirection, SignalType,
    )

    # Verificar enumeraciones
    assert TradeStatus.OPEN == "open"
    assert TradeDirection.LONG == "long"
    assert SignalType.BUY == "buy"

    print("✓ Modelos de base de datos importados correctamente")


def test_logger():
    """Verifica que el sistema de logging funciona."""
    from utils.logger import get_logger, setup_logging

    setup_logging(log_level="DEBUG", enable_console=False)
    log = get_logger("test")
    log.info("Test de logging")
    log.debug("Debug message")
    log.warning("Warning message")
    print("✓ Sistema de logging funcionando")


if __name__ == "__main__":
    """Ejecución directa de los tests."""
    print("\n🔍 Ejecutando tests de Fase 1...\n")

    tests = [
        ("Configuración", test_settings_load),
        ("Cliente Binance (dry_run)", test_dry_run_client),
        ("Modelos de base de datos", test_database_models),
        ("Sistema de logging", test_logger),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Tests: {passed} pasados, {failed} fallados")

    if failed == 0:
        print("✅ Todos los tests de Fase 1 pasaron")
    else:
        print("❌ Hay tests fallando - revisa los errores")
        sys.exit(1)
