"""
=============================================================
tests/test_phase5.py
Tests para el motor de ejecucion — Fase 5.

Uso:
    python tests/test_phase5.py
=============================================================
"""

import sys
import os
import asyncio
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Forzar dry_run para estos tests — evita llamadas reales a Binance
os.environ["TRADING_MODE"] = "dry_run"


def _make_settings():
    from config.settings import get_settings
    get_settings.cache_clear()
    return get_settings()


async def test_order_executor_open_close():
    """Verifica apertura y cierre de posicion en dry_run."""
    from config.settings import get_settings
    from core.binance_client import BinanceFuturesClient
    from database.connection import init_database
    from execution.order_executor import OrderExecutor
    from strategy.strategy_engine import TradeDecision

    get_settings.cache_clear()
    settings = _make_settings()
    client   = BinanceFuturesClient(settings)
    client.initialize()
    db = init_database(settings.database_url, settings.async_database_url)

    executor = OrderExecutor(settings, client, db)

    # Crear decision de trading simulada
    decision = TradeDecision(
        action="ENTER_LONG",
        symbol="BTCUSDT",
        direction="long",
        entry_price=50000.0,
        stop_loss=49000.0,
        take_profit=52500.0,
        quantity=Decimal("0.001"),
        leverage=3,
        margin_used=16.67,
        risk_amount=1.0,
        risk_pct=0.001,
        signal_confidence=0.72,
        risk_reward=2.5,
    )

    # Abrir posicion
    pos = await executor.open_position(decision)
    assert pos is not None, "Posicion no se abrio"
    assert pos.symbol == "BTCUSDT"
    assert pos.direction == "long"
    assert executor.open_count == 1

    # Verificar que no se puede abrir otra en el mismo simbolo
    pos2 = await executor.open_position(decision)
    assert pos2 is None, "No debe abrir posicion duplicada"
    assert executor.open_count == 1

    print(f"  Posicion abierta: {pos}")
    print(f"  Trade ID: {pos.trade_id}")
    print(f"  SL order: {pos.sl_order_id}")

    # PnL no realizado
    pnl_up   = pos.calc_unrealized_pnl(51000)
    pnl_down = pos.calc_unrealized_pnl(-49500)
    assert pnl_up > 0,   f"PnL subida debe ser positivo: {pnl_up}"
    assert pnl_down < 0, f"PnL bajada debe ser negativo: {pnl_down}"

    # Cerrar posicion
    closed = await executor.close_position(pos, "tp_hit", 52500.0)
    assert closed, "Posicion no se cerro"
    assert executor.open_count == 0

    print(f"  PnL subida (+1000): ${pnl_up:.2f}")
    print(f"  PnL bajada (-500): ${pnl_down:.2f}")
    print("  OK: OrderExecutor open/close")


async def test_monitor_positions():
    """Verifica deteccion de SL/TP hit en monitor."""
    from config.settings import get_settings
    from core.binance_client import BinanceFuturesClient
    from database.connection import init_database
    from execution.order_executor import OrderExecutor, OpenPosition
    from strategy.strategy_engine import TradeDecision

    get_settings.cache_clear()
    settings = _make_settings()
    client   = BinanceFuturesClient(settings)
    client.initialize()
    db = init_database(settings.database_url, settings.async_database_url)

    executor = OrderExecutor(settings, client, db)

    # Abrir posicion long simulada
    decision = TradeDecision(
        action="ENTER_LONG", symbol="ETHUSDT", direction="long",
        entry_price=3000.0, stop_loss=2940.0, take_profit=3150.0,
        quantity=Decimal("0.01"), leverage=3,
        margin_used=10.0, risk_amount=0.6, risk_pct=0.0006,
        signal_confidence=0.68, risk_reward=2.5,
    )
    pos = await executor.open_position(decision)
    assert pos is not None

    # Precio cae al SL -> debe detectar sl_hit
    to_close = await executor.monitor_positions({"ETHUSDT": 2930.0})
    assert len(to_close) == 1
    assert to_close[0][1] == "sl_hit"
    print(f"  SL hit detectado correctamente a precio 2930")

    # Limpiar
    await executor.close_position(pos, "sl_hit", 2930.0)

    # Abrir otra posicion short
    decision2 = TradeDecision(
        action="ENTER_SHORT", symbol="SOLUSDT", direction="short",
        entry_price=150.0, stop_loss=156.0, take_profit=138.0,
        quantity=Decimal("0.1"), leverage=3,
        margin_used=5.0, risk_amount=0.4, risk_pct=0.0004,
        signal_confidence=0.65, risk_reward=2.0,
    )
    pos2 = await executor.open_position(decision2)

    # Precio baja al TP del short -> tp_hit
    to_close2 = await executor.monitor_positions({"SOLUSDT": 137.0})
    assert len(to_close2) == 1
    assert to_close2[0][1] == "tp_hit"
    print(f"  TP hit detectado correctamente (short a 137)")
    await executor.close_position(pos2, "tp_hit", 137.0)

    print("  OK: Monitor posiciones")


async def test_portfolio_manager_sync():
    """Verifica sincronizacion del portfolio."""
    from config.settings import get_settings
    from core.binance_client import BinanceFuturesClient
    from database.connection import init_database
    from execution.order_executor import OrderExecutor
    from execution.portfolio_manager import PortfolioManager
    from risk.risk_manager import RiskManager

    get_settings.cache_clear()
    settings = _make_settings()
    client   = BinanceFuturesClient(settings)
    client.initialize()
    db       = init_database(settings.database_url, settings.async_database_url)

    risk_mgr  = RiskManager(settings)
    executor  = OrderExecutor(settings, client, db)
    portfolio = PortfolioManager(settings, client, db, executor, risk_mgr)

    state = await portfolio.sync()

    assert "total_balance" in state
    assert "drawdown_pct"  in state
    assert state["total_balance"] > 0

    print(f"  Balance: ${state['total_balance']:.2f}")
    print(f"  DD: {state['drawdown_pct']:.2f}%")
    print(f"  Posiciones: {state['open_positions']}")
    print("  OK: PortfolioManager sync")


def test_open_position_in_memory():
    """Verifica calculos de OpenPosition."""
    from execution.order_executor import OpenPosition

    pos = OpenPosition(
        trade_id=1,
        symbol="BTCUSDT",
        direction="long",
        entry_price=50000.0,
        quantity=Decimal("0.002"),
        leverage=5,
        stop_loss=49000.0,
        take_profit=53000.0,
    )

    # PnL con leverage 5x
    pnl = pos.calc_unrealized_pnl(51000)
    expected = (51000 - 50000) * 0.002 * 5  # = 10
    assert abs(pnl - expected) < 0.01, f"PnL incorrecto: {pnl} != {expected}"

    pnl_pct = pos.calc_pnl_pct(51000)
    assert pnl_pct > 0

    pos.update_price_extremes(52000)
    pos.update_price_extremes(49500)
    assert pos.highest_price == 52000
    assert pos.lowest_price  == 49500

    print(f"  PnL a 51000: ${pnl:.2f} ({pnl_pct:.2%})")
    print(f"  Max: {pos.highest_price} | Min: {pos.lowest_price}")
    print("  OK: OpenPosition calculos")


if __name__ == "__main__":
    print("\n Ejecutando tests de Fase 5...\n")

    sync_tests = [
        ("OpenPosition calculos", test_open_position_in_memory),
    ]

    async_tests = [
        ("OrderExecutor open/close",   test_order_executor_open_close),
        ("Monitor posiciones SL/TP",   test_monitor_positions),
        ("PortfolioManager sync",      test_portfolio_manager_sync),
    ]

    passed = failed = 0

    for name, fn in sync_tests:
        print(f"[{name}]")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FALLO: {e}")
            import traceback; traceback.print_exc()
            failed += 1
        print()

    for name, fn in async_tests:
        print(f"[{name}]")
        try:
            asyncio.run(fn())
            passed += 1
        except Exception as e:
            print(f"  FALLO: {e}")
            import traceback; traceback.print_exc()
            failed += 1
        print()

    print("=" * 45)
    print(f"Tests: {passed} pasados, {failed} fallados")
    print("Fase 5 lista" if failed == 0 else "Hay errores")
    sys.exit(0 if failed == 0 else 1)
