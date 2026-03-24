"""
=============================================================
tests/test_phase4.py
Tests para estrategia y gestion de riesgo — Fase 4.

Uso:
    python tests/test_phase4.py
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
os.environ.setdefault("TRADING_MODE", "dry_run")


def _make_settings():
    from config.settings import get_settings
    get_settings.cache_clear()
    return get_settings()


def _make_df(n=200, trend="up"):
    """DataFrame sintetico con indicadores para tests."""
    import numpy as np
    import pandas as pd
    from analysis.indicators import TechnicalIndicators

    np.random.seed(42)
    if trend == "up":
        price = 50000 + np.cumsum(np.abs(np.random.randn(n)) * 100)
    elif trend == "down":
        price = 55000 - np.cumsum(np.abs(np.random.randn(n)) * 100)
    else:
        price = 50000 + np.cumsum(np.random.randn(n) * 80)

    price = np.maximum(price, 100)
    df = pd.DataFrame({
        "open":         price * (1 - np.random.rand(n) * 0.001),
        "high":         price * (1 + np.random.rand(n) * 0.004),
        "low":          price * (1 - np.random.rand(n) * 0.004),
        "close":        price,
        "volume":       np.random.rand(n) * 1000 + 300,
        "quote_volume": np.random.rand(n) * 50_000_000,
    })
    return TechnicalIndicators().add_all(df)


def test_risk_manager_sizing():
    """Verifica el calculo de tamaño de posicion con Kelly."""
    from risk.risk_manager import RiskManager
    settings = _make_settings()
    rm = RiskManager(settings)
    rm.update_balance(total_balance=1000, available_balance=1000)

    # Trade normal: BTC a 50000, SL a 49500 (1% abajo), leverage 3x
    size = rm.calculate_position_size(
        signal_confidence=0.70,
        entry_price=50000.0,
        stop_loss_price=49500.0,
        leverage=3,
    )

    assert size.approved, f"Deberia aprobarse: {size.reject_reason}"
    assert size.quantity > 0
    assert size.risk_amount > 0
    assert size.risk_pct <= settings.max_risk_per_trade + 0.001
    assert size.margin_used < 1000

    print(f"  Cantidad: {size.quantity} BTC")
    print(f"  Margen:   ${size.margin_used:.2f}")
    print(f"  Riesgo:   ${size.risk_amount:.2f} ({size.risk_pct*100:.2f}%)")
    print(f"  Kelly raw: {size.kelly_fraction:.4f}")
    print("  OK: RiskManager sizing")


def test_risk_manager_drawdown_pause():
    """Verifica la pausa automatica por drawdown."""
    from risk.risk_manager import RiskManager
    settings = _make_settings()
    rm = RiskManager(settings)

    # Simular drawdown elevado
    rm.update_balance(total_balance=1000, available_balance=1000)  # Peak = 1000
    rm.update_balance(total_balance=850, available_balance=850)    # DD = 15%

    # Debe estar pausado si max_drawdown=0.15
    if settings.max_drawdown <= 0.15:
        assert rm.state.is_paused, "Deberia estar pausado con 15% DD"
        size = rm.calculate_position_size(0.7, 50000, 49500, 3)
        assert not size.approved, "No deberia aprobar si esta pausado"

    print(f"  DD actual: {rm.state.current_drawdown*100:.1f}%")
    print(f"  Pausado: {rm.state.is_paused}")
    print("  OK: RiskManager pausa por drawdown")


def test_sl_tp_atr():
    """Verifica calculo de SL/TP basado en ATR."""
    from risk.sl_tp_calculator import SLTPCalculator
    df = _make_df(200)
    calc = SLTPCalculator()

    entry = float(df["close"].iloc[-1])

    # LONG
    levels_long = calc.calculate("long", entry, df)
    assert levels_long.is_valid
    assert levels_long.stop_loss < entry
    assert levels_long.take_profit > entry
    assert levels_long.risk_reward >= 1.4
    assert levels_long.method == "atr"

    # SHORT
    levels_short = calc.calculate("short", entry, df)
    assert levels_short.is_valid
    assert levels_short.stop_loss > entry
    assert levels_short.take_profit < entry
    assert levels_short.risk_reward >= 1.4

    print(f"  LONG  SL={levels_long.stop_loss:.2f} TP={levels_long.take_profit:.2f} R:R={levels_long.risk_reward:.2f}")
    print(f"  SHORT SL={levels_short.stop_loss:.2f} TP={levels_short.take_profit:.2f} R:R={levels_short.risk_reward:.2f}")
    print(f"  Metodo: {levels_long.method} | ATR%: {levels_long.sl_pct*100:.2f}%")
    print("  OK: SLTPCalculator ATR")


def test_trailing_stop():
    """Verifica el trailing stop."""
    from risk.sl_tp_calculator import SLTPCalculator
    calc = SLTPCalculator()

    # LONG: precio sube, el SL debe subir
    sl1 = calc.update_trailing_stop("long", 51000, 49500, 50000, atr=200)
    sl2 = calc.update_trailing_stop("long", 52000, sl1, 50000, atr=200)
    assert sl2 > sl1, f"SL debe subir en long: {sl1} -> {sl2}"

    # LONG: precio baja, el SL NO debe bajar
    sl3 = calc.update_trailing_stop("long", 50500, sl2, 50000, atr=200)
    assert sl3 == sl2, f"SL no debe bajar: {sl2} -> {sl3}"

    print(f"  Trailing LONG: SL {49500} -> {sl1:.0f} -> {sl2:.0f} (no baja: {sl3:.0f})")
    print("  OK: Trailing stop")


def test_strategy_engine_full():
    """Verifica el motor de estrategia end-to-end."""
    from ai.predictor import SignalResult
    from risk.risk_manager import RiskManager
    from risk.sl_tp_calculator import SLTPCalculator
    from strategy.strategy_engine import StrategyEngine

    settings = _make_settings()
    rm   = RiskManager(settings)
    calc = SLTPCalculator()
    eng  = StrategyEngine(settings, rm, calc, leverage=3)

    rm.update_balance(total_balance=1000, available_balance=1000)
    df = _make_df(200, trend="up")

    # Test: señal BUY con alta confianza -> debe aprobar
    signal_buy = SignalResult(
        symbol="BTCUSDT",
        signal="BUY",
        confidence=0.72,
        probabilities={"HOLD": 0.1, "BUY": 0.72, "SELL": 0.18},
        price=float(df["close"].iloc[-1]),
        timeframe="15m",
        indicators={},
    )
    decision = eng.evaluate(signal_buy, df)
    print(f"  BUY decision: {decision.action} | filtros OK: {decision.filters_passed}")
    assert decision.action in ("ENTER_LONG", "SKIP")  # Puede pasar o fallar filtros

    # Test: señal HOLD -> siempre SKIP
    signal_hold = SignalResult(
        symbol="ETHUSDT", signal="HOLD", confidence=0.45,
        probabilities={"HOLD": 0.55, "BUY": 0.3, "SELL": 0.15},
        price=3000.0, timeframe="15m", indicators={},
    )
    decision_hold = eng.evaluate(signal_hold, df)
    assert decision_hold.action == "SKIP"
    assert not decision_hold.is_actionable

    # Test: confianza baja -> SKIP
    signal_low_conf = SignalResult(
        symbol="SOLUSDT", signal="BUY", confidence=0.45,
        probabilities={"HOLD": 0.3, "BUY": 0.45, "SELL": 0.25},
        price=150.0, timeframe="15m", indicators={},
    )
    decision_low = eng.evaluate(signal_low_conf, df)
    assert decision_low.action == "SKIP"
    assert "Confianza" in decision_low.skip_reason

    print(f"  HOLD -> SKIP: OK")
    print(f"  Low conf -> SKIP: OK")
    print("  OK: StrategyEngine completo")


def test_kelly_conservative():
    """Verifica que Kelly nunca supera el maximo configurado."""
    from risk.risk_manager import RiskManager
    settings = _make_settings()
    rm = RiskManager(settings)
    rm.update_balance(1000, 1000)

    for confidence in [0.55, 0.65, 0.75, 0.85, 0.95]:
        size = rm.calculate_position_size(confidence, 50000, 49000, 3)
        if size.approved:
            assert size.risk_pct <= settings.max_risk_per_trade + 0.0001, \
                f"Kelly supera max_risk_per_trade a conf={confidence}: {size.risk_pct}"

    print(f"  Max riesgo configurado: {settings.max_risk_per_trade*100:.1f}%")
    print("  OK: Kelly dentro de limites")


if __name__ == "__main__":
    print("\n Ejecutando tests de Fase 4...\n")

    tests = [
        ("Risk Manager sizing",      test_risk_manager_sizing),
        ("Pausa por drawdown",        test_risk_manager_drawdown_pause),
        ("SL/TP ATR-based",          test_sl_tp_atr),
        ("Trailing stop",             test_trailing_stop),
        ("Strategy Engine completo",  test_strategy_engine_full),
        ("Kelly dentro de limites",   test_kelly_conservative),
    ]

    passed = failed = 0
    for name, fn in tests:
        print(f"[{name}]")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FALLO: {e}")
            import traceback; traceback.print_exc()
            failed += 1
        print()

    print("=" * 45)
    print(f"Tests: {passed} pasados, {failed} fallados")
    print("Fase 4 lista" if failed == 0 else "Hay errores")
    sys.exit(0 if failed == 0 else 1)
