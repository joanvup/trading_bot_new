"""
=============================================================
tests/test_phase3.py
Tests para el motor de IA - Fase 3.

Uso:
    python tests/test_phase3.py
=============================================================
"""

import sys
import os
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
os.environ.setdefault("TRADING_MODE", "dry_run")


def _make_synthetic_df(n: int = 300):
    """Crea un DataFrame sintetico de velas OHLCV con indicadores."""
    import numpy as np
    import pandas as pd
    from analysis.indicators import TechnicalIndicators

    np.random.seed(42)
    price = 50000 + np.cumsum(np.random.randn(n) * 150)
    price = np.maximum(price, 100)

    df = pd.DataFrame({
        "open":         price * (1 - np.random.rand(n) * 0.001),
        "high":         price * (1 + np.random.rand(n) * 0.003),
        "low":          price * (1 - np.random.rand(n) * 0.003),
        "close":        price,
        "volume":       np.random.rand(n) * 1000 + 200,
        "quote_volume": np.random.rand(n) * 50_000_000,
    })

    ti = TechnicalIndicators()
    return ti.add_all(df)


def test_feature_engineer():
    """Verifica el pipeline de feature engineering."""
    import tempfile
    import numpy as np
    from ai.feature_engineer import FeatureEngineer

    df = _make_synthetic_df(300)
    assert not df.empty, "DataFrame vacio"

    with tempfile.TemporaryDirectory() as tmp:
        fe = FeatureEngineer(Path(tmp))
        X_train, y_train, X_test, y_test, names = fe.prepare_training_data(df)

        assert X_train.shape[1] == len(names), "Dimensiones no coinciden"
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(names) >= 20, f"Pocos features: {len(names)}"

        # Verificar que el scaler funciona en inferencia
        fe.save_scaler()
        X_inf = fe.prepare_inference(df)
        assert X_inf is not None
        assert X_inf.shape == (1, len(names))

        dist = fe.get_class_distribution(y_train)
        assert "BUY" in dist
        assert "SELL" in dist
        assert "HOLD" in dist

        print(f"  Features: {len(names)}")
        print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
        print(f"  Distribucion: {dist}")
        print("  OK: FeatureEngineer")


def test_ml_model_train():
    """Verifica entrenamiento basico del modelo LightGBM."""
    import tempfile
    import numpy as np
    from ai.feature_engineer import FeatureEngineer
    from ai.ml_model import TradingMLModel

    try:
        import lightgbm
    except ImportError:
        print("  SKIP: LightGBM no instalado (pip install lightgbm)")
        return

    df = _make_synthetic_df(400)

    with tempfile.TemporaryDirectory() as tmp:
        models_dir = Path(tmp)
        fe    = FeatureEngineer(models_dir)
        model = TradingMLModel(models_dir)

        X_train, y_train, X_test, y_test, names = fe.prepare_training_data(df)

        # Entrenar SIN Optuna para que sea rapido en tests
        metrics = model.train(
            X_train, y_train,
            X_test,  y_test,
            feature_names=names,
            use_optuna=False,
        )

        assert metrics["test_accuracy"] > 0
        assert metrics["f1_weighted"] > 0
        assert model.is_trained

        # Test de prediccion
        X_inf = fe.prepare_inference(df)
        signal, confidence, probs = model.predict(X_inf, min_confidence=0.4)

        assert signal in ("BUY", "SELL", "HOLD")
        assert 0.0 <= confidence <= 1.0
        assert abs(sum(probs.values()) - 1.0) < 0.01

        # Test de guardado/carga
        fe.save_scaler()
        path = model.save("test_v1")
        assert path.exists()

        # Cargar en nuevo objeto
        model2 = TradingMLModel(models_dir)
        loaded = model2.load("test_v1")
        assert loaded
        assert model2.is_trained

        print(f"  Test accuracy: {metrics['test_accuracy']:.3f}")
        print(f"  F1 weighted:   {metrics['f1_weighted']:.3f}")
        print(f"  F1 BUY:        {metrics['f1_buy']:.3f}")
        print(f"  F1 SELL:       {metrics['f1_sell']:.3f}")
        print(f"  Señal test:    {signal} ({confidence:.2%})")
        print("  OK: TradingMLModel")


def test_signal_result():
    """Verifica la clase SignalResult."""
    from ai.predictor import SignalResult

    r = SignalResult(
        symbol="BTCUSDT",
        signal="BUY",
        confidence=0.72,
        probabilities={"HOLD": 0.1, "BUY": 0.72, "SELL": 0.18},
        price=50000.0,
        timeframe="15m",
        indicators={"rsi_14": 45.2, "macd": 120.5},
    )

    assert r.is_actionable == True
    assert r.symbol == "BTCUSDT"
    assert r.confidence == 0.72

    r_hold = SignalResult(
        symbol="ETHUSDT", signal="HOLD", confidence=0.45,
        probabilities={"HOLD": 0.55, "BUY": 0.3, "SELL": 0.15},
        price=3000.0, timeframe="15m", indicators={},
    )
    assert r_hold.is_actionable == False

    print("  OK: SignalResult")


async def test_predictor_no_model():
    """Verifica que el predictor devuelve HOLD cuando no hay modelo."""
    import tempfile
    from config.settings import get_settings
    from database.connection import init_database
    from ai.predictor import RealTimePredictor

    get_settings.cache_clear()
    settings = get_settings()
    db = init_database(settings.database_url, settings.async_database_url)

    with tempfile.TemporaryDirectory() as tmp:
        predictor = RealTimePredictor(
            settings=settings,
            db=db,
            models_dir=Path(tmp),
        )
        # Sin modelo entrenado -> should return HOLD
        initialized = await predictor.initialize()
        assert not initialized, "No deberia inicializarse sin modelo"

        df = _make_synthetic_df(200)
        result = await predictor.predict("BTCUSDT", "15m", df=df)

        assert result.signal == "HOLD"
        assert result.confidence == 0.0

    print("  OK: Predictor sin modelo -> HOLD")


if __name__ == "__main__":
    print("\n Ejecutando tests de Fase 3...\n")

    sync_tests = [
        ("Feature Engineering",  test_feature_engineer),
        ("LightGBM training",    test_ml_model_train),
        ("SignalResult",         test_signal_result),
    ]

    async_tests = [
        ("Predictor sin modelo", test_predictor_no_model),
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
    print("Fase 3 lista" if failed == 0 else "Hay errores - revisa arriba")
    sys.exit(0 if failed == 0 else 1)
