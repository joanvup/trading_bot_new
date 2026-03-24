"""
=============================================================
analysis/asset_selector.py
Selector automatico de activos con mayor proyeccion diaria.

Analiza todos los pares USDT-M perpetuos y calcula un score
compuesto basado en:
  - Volumen relativo (liquidez)
  - Momentum de precio (direccionalidad)
  - Volatilidad (oportunidad de movimiento)
  - Order Book Imbalance (presion compradora/vendedora)
  - Variacion 24h y tendencia reciente

El resultado es un ranking dinamico que se actualiza cada
ciclo de analisis (configurable).
=============================================================
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from core.binance_client import BinanceFuturesClient
from analysis.indicators import TechnicalIndicators
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SymbolScore:
    """Score y metricas de un simbolo candidato."""
    symbol:           str
    score:            float          # Score compuesto 0-100
    rank:             int = 0

    # Metricas individuales
    volume_score:     float = 0.0   # Volumen relativo
    momentum_score:   float = 0.0   # Momentum de precio
    volatility_score: float = 0.0   # Volatilidad (ATR%)
    obi_score:        float = 0.0   # Order Book Imbalance
    trend_score:      float = 0.0   # Calidad de tendencia (ADX)

    # Datos de mercado crudos
    price:            float = 0.0
    change_24h_pct:   float = 0.0
    volume_24h_usdt:  float = 0.0
    atr_pct:          float = 0.0
    rsi_14:           float = 50.0
    adx:              float = 0.0
    obi:              float = 0.0   # -1 a 1

    # Metadatos
    direction:        str   = "neutral"  # "long", "short", "neutral"
    updated_at:       float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "symbol":          self.symbol,
            "score":           round(self.score, 2),
            "rank":            self.rank,
            "price":           self.price,
            "change_24h_pct":  round(self.change_24h_pct, 3),
            "volume_24h_usdt": round(self.volume_24h_usdt / 1e6, 2),  # en millones
            "atr_pct":         round(self.atr_pct, 3),
            "rsi_14":          round(self.rsi_14, 1),
            "adx":             round(self.adx, 1),
            "obi":             round(self.obi, 3),
            "direction":       self.direction,
            "volume_score":    round(self.volume_score, 2),
            "momentum_score":  round(self.momentum_score, 2),
            "volatility_score":round(self.volatility_score, 2),
            "trend_score":     round(self.trend_score, 2),
        }


class AssetSelector:
    """
    Selecciona automaticamente los mejores activos del dia.

    Flujo:
        1. Obtiene todos los tickers 24h de Binance
        2. Aplica filtros de liquidez minima
        3. Calcula score compuesto para cada simbolo
        4. Retorna el Top N segun configuracion

    Uso:
        selector = AssetSelector(client, top_n=20)
        top_symbols = await selector.get_top_symbols()

        for s in top_symbols:
            print(f"{s.rank}. {s.symbol}: {s.score:.1f} pts | {s.direction}")
    """

    # Pesos del score compuesto (deben sumar 1.0)
    WEIGHTS = {
        "volume":     0.25,   # Liquidez y actividad
        "momentum":   0.30,   # Fuerza del movimiento de precio
        "volatility": 0.20,   # Rango de movimiento (ATR)
        "trend":      0.15,   # Calidad de la tendencia (ADX)
        "obi":        0.10,   # Presion del libro de ordenes
    }

    # Filtros de inclusion minima
    MIN_VOLUME_USDT      = 20_000_000    # Minimo 20M USDT/dia de volumen
    MIN_PRICE_USDT       = 0.001         # Precio minimo (excluye micro-caps)
    MAX_CHANGE_24H_PCT   = 25.0          # Excluir si subio/bajo mas de 25% (ya tarde)
    MIN_ATR_PCT          = 0.3           # Minimo 0.3% de ATR (algo de movimiento)

    def __init__(
        self,
        client: BinanceFuturesClient,
        indicators: Optional[TechnicalIndicators] = None,
        top_n: int = 20,
        excluded_symbols: Optional[list[str]] = None,
    ):
        self.client     = client
        self.ti         = indicators or TechnicalIndicators()
        self.top_n      = top_n
        self.excluded   = set(excluded_symbols or [
            # Excluir stablecoins y tokens con baja liquidez problematica
            "USDCUSDT", "BUSDUSDT", "TUSDUSDT", "USDTUSDT",
            "LUNAUSDT", "LUNCUSDT",  # Tokens con historial de colapso
        ])

        self._last_scores:    list[SymbolScore] = []
        self._last_update:    float = 0
        self._cache_ttl_secs: int = 300   # Cache valido 5 minutos

    # ── Metodo principal ───────────────────────────────────────────────────────

    async def get_top_symbols(
        self, force_refresh: bool = False
    ) -> list[SymbolScore]:
        """
        Retorna el ranking de los mejores simbolos del dia.
        Usa cache de 5 minutos para no sobrecargar la API.

        Args:
            force_refresh: Si True, ignora el cache y recalcula

        Returns:
            Lista de SymbolScore ordenada por score descendente
        """
        now = time.time()

        # Usar cache si es reciente
        if (
            not force_refresh
            and self._last_scores
            and (now - self._last_update) < self._cache_ttl_secs
        ):
            log.debug(
                f"Usando cache de activos "
                f"({int(now - self._last_update)}s de antiguedad)"
            )
            return self._last_scores

        log.info("Calculando ranking de activos...")

        try:
            scores = await self._calculate_all_scores()
            self._last_scores = scores
            self._last_update = now

            self._log_top_symbols(scores)
            return scores

        except Exception as e:
            log.error(f"Error calculando scores de activos: {e}")
            if self._last_scores:
                log.warning("Usando ultimo ranking conocido")
                return self._last_scores
            # Fallback a simbolos simulados solo si no hay conexion
            log.warning("Sin datos de mercado, usando simbolos simulados")
            return self._get_dry_run_symbols()

    def get_top_symbol_names(self, n: Optional[int] = None) -> list[str]:
        """
        Retorna solo los nombres de los top simbolos.
        Util para pasarlos a otras funciones.
        """
        top = self._last_scores[:n or self.top_n]
        return [s.symbol for s in top]

    # ── Calculo de scores ──────────────────────────────────────────────────────

    async def _calculate_all_scores(self) -> list[SymbolScore]:
        """
        Obtiene tickers 24h y calcula el score de cada simbolo.
        """
        # Obtener todos los tickers 24h de una sola llamada
        tickers = self.client.get_ticker_24h()

        if not tickers:
            return []

        candidates: list[SymbolScore] = []

        for ticker in tickers:
            symbol = ticker.get("symbol", "")

            # Filtros basicos rapidos
            if not symbol.endswith("USDT"):
                continue
            if symbol in self.excluded:
                continue

            try:
                price      = float(ticker.get("lastPrice", 0))
                vol_usdt   = float(ticker.get("quoteVolume", 0))
                change_pct = float(ticker.get("priceChangePercent", 0))
                high_24h   = float(ticker.get("highPrice", price))
                low_24h    = float(ticker.get("lowPrice", price))

                # Filtros de liquidez y precio
                if price < self.MIN_PRICE_USDT:
                    continue
                if vol_usdt < self.MIN_VOLUME_USDT:
                    continue
                if abs(change_pct) > self.MAX_CHANGE_24H_PCT:
                    continue

                # Metricas basicas del ticker
                range_pct = (high_24h - low_24h) / price * 100 if price > 0 else 0

                score = SymbolScore(
                    symbol=symbol,
                    score=0.0,
                    price=price,
                    change_24h_pct=change_pct,
                    volume_24h_usdt=vol_usdt,
                    atr_pct=range_pct,  # Proxy de ATR con rango 24h
                )

                candidates.append(score)

            except (ValueError, TypeError) as e:
                log.debug(f"Error parseando ticker {symbol}: {e}")
                continue

        log.debug(f"Candidatos tras filtros: {len(candidates)}")

        if not candidates:
            return []

        # Calcular scores componente por componente
        self._calc_volume_scores(candidates)
        self._calc_momentum_scores(candidates)
        self._calc_volatility_scores(candidates)

        # Enriquecer con Order Book (solo top 50 para no sobrecargar API)
        top_by_volume = sorted(
            candidates, key=lambda x: x.volume_24h_usdt, reverse=True
        )[:50]

        await self._enrich_with_order_book(top_by_volume)

        # Score final compuesto
        for s in candidates:
            s.score = (
                s.volume_score     * self.WEIGHTS["volume"]
                + s.momentum_score * self.WEIGHTS["momentum"]
                + s.volatility_score * self.WEIGHTS["volatility"]
                + s.trend_score    * self.WEIGHTS["trend"]
                + s.obi_score      * self.WEIGHTS["obi"]
            )

        # Ordenar y asignar rankings
        sorted_scores = sorted(candidates, key=lambda x: x.score, reverse=True)
        for i, s in enumerate(sorted_scores):
            s.rank = i + 1

        return sorted_scores[:self.top_n]

    def _calc_volume_scores(self, candidates: list[SymbolScore]) -> None:
        """Normaliza y asigna score de volumen (0-100)."""
        volumes = [c.volume_24h_usdt for c in candidates]
        if not volumes:
            return

        log_vols = np.log1p(volumes)
        min_v, max_v = log_vols.min(), log_vols.max()
        rng = max_v - min_v

        for i, c in enumerate(candidates):
            if rng > 0:
                c.volume_score = (log_vols[i] - min_v) / rng * 100
            else:
                c.volume_score = 50.0

    def _calc_momentum_scores(self, candidates: list[SymbolScore]) -> None:
        """
        Score de momentum basado en variacion 24h.
        Favorece tanto movimientos alcistas como bajistas (oportunidades en ambos).
        """
        changes = np.array([c.change_24h_pct for c in candidates])

        # Usar valor absoluto (tanto subidas como bajadas son oportunidades)
        abs_changes = np.abs(changes)
        min_c, max_c = abs_changes.min(), abs_changes.max()
        rng = max_c - min_c

        for i, c in enumerate(candidates):
            if rng > 0:
                c.momentum_score = (abs_changes[i] - min_c) / rng * 100
            else:
                c.momentum_score = 50.0

            # Asignar direccion sugerida
            if c.change_24h_pct > 1.0:
                c.direction = "long"
            elif c.change_24h_pct < -1.0:
                c.direction = "short"
            else:
                c.direction = "neutral"

    def _calc_volatility_scores(self, candidates: list[SymbolScore]) -> None:
        """
        Score de volatilidad basado en rango 24h.
        Premia activos con suficiente movimiento, pero no demasiado (penaliza extremos).
        Zona optima: 1.5% - 8% de rango diario.
        """
        for c in candidates:
            atr = c.atr_pct

            if atr < self.MIN_ATR_PCT:
                c.volatility_score = 0.0
                continue

            # Funcion de puntuacion: optimo entre 2% y 6% de rango
            if atr < 2.0:
                c.volatility_score = atr / 2.0 * 60   # Escala lineal baja
            elif atr <= 6.0:
                c.volatility_score = 60 + (atr - 2.0) / 4.0 * 40  # Zona optima
            elif atr <= 12.0:
                c.volatility_score = 100 - (atr - 6.0) / 6.0 * 40  # Penaliza exceso
            else:
                c.volatility_score = 20.0  # Muy volatil: riesgo alto

    async def _enrich_with_order_book(
        self, candidates: list[SymbolScore]
    ) -> None:
        """
        Agrega OBI (Order Book Imbalance) a los candidatos.
        Solo para el top por volumen (limitar llamadas API).
        """
        from analysis.indicators import TechnicalIndicators

        for c in candidates:
            try:
                book = self.client.get_order_book(c.symbol, limit=20)
                bids = book.get("bids", [])
                asks = book.get("asks", [])

                obi = self.ti.calc_order_book_imbalance(bids, asks)
                c.obi = obi
                # Normalizar OBI de [-1, 1] a [0, 100]
                c.obi_score = (obi + 1) / 2 * 100

                await asyncio.sleep(0.02)  # 20ms entre requests

            except Exception as e:
                log.debug(f"No se pudo obtener order book de {c.symbol}: {e}")
                c.obi_score = 50.0  # Neutral si no hay datos

    # ── Utilidades ─────────────────────────────────────────────────────────────

    def _get_dry_run_symbols(self) -> list[SymbolScore]:
        """Retorna simbolos de ejemplo para modo dry_run."""
        default_symbols = [
            ("BTCUSDT",  50000.0,  2.5,  1_500_000_000),
            ("ETHUSDT",  3000.0,   1.8,    800_000_000),
            ("BNBUSDT",  400.0,    3.1,    200_000_000),
            ("SOLUSDT",  150.0,    4.2,    300_000_000),
            ("XRPUSDT",  0.8,     -1.5,    250_000_000),
            ("ADAUSDT",  0.6,      2.0,    100_000_000),
            ("DOGEUSDT", 0.15,     5.3,    120_000_000),
            ("AVAXUSDT", 35.0,     3.8,     80_000_000),
            ("LINKUSDT", 18.0,     2.2,     70_000_000),
            ("LTCUSDT",  90.0,     1.9,     60_000_000),
        ]

        scores = []
        for i, (symbol, price, change, vol) in enumerate(default_symbols):
            s = SymbolScore(
                symbol=symbol,
                score=90.0 - i * 5,
                rank=i + 1,
                price=price,
                change_24h_pct=change,
                volume_24h_usdt=vol,
                volume_score=80.0 - i * 3,
                momentum_score=75.0 - i * 2,
                volatility_score=70.0,
                trend_score=65.0,
                obi_score=50.0,
                atr_pct=abs(change) * 0.8,
                rsi_14=55.0 + change * 2,
                direction="long" if change > 0 else "short",
            )
            scores.append(s)

        log.info(f"[DRY RUN] Top {len(scores)} simbolos simulados")
        return scores[:self.top_n]

    def _log_top_symbols(self, scores: list[SymbolScore]) -> None:
        """Loguea el ranking de forma legible."""
        log.info(f"{'='*60}")
        log.info(f"TOP {len(scores)} SIMBOLOS DEL DIA")
        log.info(f"{'='*60}")
        for s in scores[:10]:  # Solo top 10 en el log
            log.info(
                f"  #{s.rank:2d} {s.symbol:<12} "
                f"Score:{s.score:5.1f} | "
                f"Vol:{s.volume_24h_usdt/1e6:6.1f}M$ | "
                f"Chg:{s.change_24h_pct:+5.2f}% | "
                f"Dir:{s.direction}"
            )
        log.info(f"{'='*60}")

    @property
    def last_update_age_secs(self) -> float:
        """Segundos desde la ultima actualizacion del ranking."""
        if self._last_update == 0:
            return float("inf")
        return time.time() - self._last_update

    @property
    def is_cache_valid(self) -> bool:
        return self.last_update_age_secs < self._cache_ttl_secs
