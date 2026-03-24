"""
=============================================================
tests/test_phase2.py
Tests para verificar que la Fase 2 funciona correctamente.

Uso:
    python tests/test_phase2.py
    pytest tests/test_phase2.py -v
=============================================================
"""

import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Cargar .env real antes de cualquier import del proyecto
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Solo aplicar defaults para lo que NO este en .env
os.environ.setdefault("TRADING_MODE", "dry_run")


def test_indicators_basic():
    """Verifica calculo de indicadores tecnicos."""
    import numpy as np
    import pandas as pd
    from analysis.indicators import TechnicalIndicators

    # Crear DataFrame sintetico de 200 velas
    np.random.seed(42)
    n = 200
    price = 50000 + np.cumsum(np.random.randn(n) * 100)
    df = pd.DataFrame({
        "open":   price - np.random.rand(n) * 50,
        "high":   price + np.random.rand(n) * 100,
        "low":    price - np.random.rand(n) * 100,
        "close":  price,
        "volume": np.random.rand(n) * 1000 + 500,
        "quote_volume": np.random.rand(n) * 50_000_000,
    })

    ti = TechnicalIndicators()
    df_out = ti.add_all(df)

    # Verificar que se calcularon columnas clave
    expected_cols = [
        "ema_9", "ema_21", "rsi_14", "macd",
        "bb_upper", "atr_14", "obv", "vol_ratio",
    ]
    for col in expected_cols:
        assert col in df_out.columns, f"Falta columna: {col}"

    assert len(df_out) > 0, "DataFrame vacio tras calcular indicadores"
    assert not df_out["rsi_14"].isna().all(), "RSI no calculado"
    assert not df_out["macd"].isna().all(), "MACD no calculado"

    print(f"  Columnas calculadas: {len(df_out.columns)}")
    print(f"  Filas validas: {len(df_out)}")
    print(f"  RSI ultimo: {df_out['rsi_14'].iloc[-1]:.1f}")
    print(f"  MACD ultimo: {df_out['macd'].iloc[-1]:.4f}")
    print("  OK: Indicadores tecnicos")


def test_asset_selector_dry_run():
    """Verifica el selector de activos en modo dry_run."""
    from config.settings import get_settings
    from core.binance_client import BinanceFuturesClient
    from analysis.asset_selector import AssetSelector

    get_settings.cache_clear()
    settings = get_settings()
    client   = BinanceFuturesClient(settings)
    client.initialize()

    selector = AssetSelector(client=client, top_n=10)
    symbols  = selector._get_dry_run_symbols()

    assert len(symbols) == 10
    assert symbols[0].rank == 1
    assert symbols[0].score > symbols[-1].score  # Ordenado
    assert all(s.symbol.endswith("USDT") for s in symbols)
    assert all(s.volume_24h_usdt > 0 for s in symbols)

    print(f"  Top 3: {[s.symbol for s in symbols[:3]]}")
    print(f"  Score rango: {symbols[-1].score:.1f} - {symbols[0].score:.1f}")
    print("  OK: Selector de activos (dry_run)")


def test_stream_buffer():
    """Verifica el buffer de datos en tiempo real."""
    from decimal import Decimal
    from data.websocket_manager import StreamBuffer

    buf = StreamBuffer(maxlen=50)

    # Simular actualizaciones
    for i in range(10):
        buf.update_price("BTCUSDT", Decimal(str(50000 + i)), Decimal("1.5"))

    buf.update_book("BTCUSDT", Decimal("49999"), Decimal("50001"))
    buf.update_mark_price("ETHUSDT", Decimal("3000"))
    buf.update_kline("BTCUSDT", "15m", {"close": Decimal("50009"), "closed": True})

    assert buf.get_last_price("BTCUSDT") == 50009.0
    assert buf.get_mark_price("ETHUSDT") == Decimal("3000")
    assert buf.get_book("BTCUSDT")["spread"] == 2.0
    assert len(buf.get_recent_prices("BTCUSDT", 5)) == 5

    print("  OK: StreamBuffer funcionando")


def test_obi_calculation():
    """Verifica el calculo de Order Book Imbalance."""
    from analysis.indicators import TechnicalIndicators

    ti = TechnicalIndicators()

    # Mas bids que asks -> OBI positivo (presion compradora)
    bids = [[50000, 10], [49999, 8], [49998, 6]]
    asks = [[50001, 2],  [50002, 1], [50003, 1]]
    obi = ti.calc_order_book_imbalance(bids, asks)
    assert obi > 0, f"OBI deberia ser positivo: {obi}"

    # Mas asks que bids -> OBI negativo (presion vendedora)
    bids2 = [[50000, 1], [49999, 1]]
    asks2 = [[50001, 10], [50002, 8]]
    obi2 = ti.calc_order_book_imbalance(bids2, asks2)
    assert obi2 < 0, f"OBI deberia ser negativo: {obi2}"

    print(f"  OBI comprador: {obi:.3f} | OBI vendedor: {obi2:.3f}")
    print("  OK: Order Book Imbalance")


async def test_data_pipeline_dry_run():
    """Verifica el pipeline completo en modo dry_run."""
    from config.settings import get_settings
    from core.binance_client import BinanceFuturesClient
    from database.connection import init_database
    from data.data_pipeline import DataPipeline

    get_settings.cache_clear()
    settings = get_settings()
    client   = BinanceFuturesClient(settings)
    client.initialize()

    db = init_database(settings.database_url, settings.async_database_url)
    pipeline = DataPipeline(settings, client, db)

    await pipeline.start()

    top = pipeline.get_top_symbols()
    assert len(top) > 0
    assert top[0].rank == 1

    names = pipeline.get_top_symbol_names(5)
    assert len(names) == 5
    assert all(isinstance(n, str) for n in names)

    status = pipeline.get_status()
    assert "top_symbols" in status
    assert status["top_symbols"] > 0

    print(f"  Top simbolos: {names}")
    print(f"  Status: {status['top_symbols']} simbolos | running={status['running']}")
    print("  OK: DataPipeline (dry_run)")

    await pipeline.stop()


if __name__ == "__main__":
    print("\n Ejecutando tests de Fase 2...\n")

    sync_tests = [
        ("Indicadores tecnicos",         test_indicators_basic),
        ("Selector de activos dry_run",  test_asset_selector_dry_run),
        ("StreamBuffer",                 test_stream_buffer),
        ("Order Book Imbalance",         test_obi_calculation),
    ]

    async_tests = [
        ("DataPipeline dry_run",         test_data_pipeline_dry_run),
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
    print("Fase 2 lista" if failed == 0 else "Hay errores - revisa arriba")

    sys.exit(0 if failed == 0 else 1)
