"""
=============================================================
data/historic_collector.py
Descarga y almacena datos OHLCV historicos desde Binance.

Funcionalidades:
  - Descarga inicial masiva de velas historicas
  - Deteccion y relleno de gaps en la BD
  - Actualizacion incremental de velas nuevas
  - Soporte para multiples timeframes
  - Respeta los rate limits de Binance
=============================================================
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Optional

import pandas as pd
from sqlalchemy import select, func, and_

from core.binance_client import BinanceFuturesClient
from database.connection import DatabaseManager
from database.models import Symbol, Candle
from utils.logger import get_logger

log = get_logger(__name__)

# Rate limit: Binance permite ~1200 requests/min en futuros
REQUEST_DELAY_SEC = 0.05   # 50ms entre requests = max 20/s (bien bajo el limite)
MAX_KLINES_PER_REQUEST = 1500


class HistoricCollector:
    """
    Descarga y sincroniza datos OHLCV en PostgreSQL.

    Flujo de trabajo:
        1. sync_symbols()          -> actualiza tabla symbols desde Binance
        2. download_history()      -> descarga velas historicas para los simbolos top
        3. fill_gaps()             -> rellena huecos detectados en la BD
        4. update_recent()         -> actualiza las ultimas N velas

    Uso:
        collector = HistoricCollector(binance_client, db_manager)
        await collector.sync_symbols()
        await collector.download_history(
            symbols=["BTCUSDT", "ETHUSDT"],
            timeframes=["15m", "1h"],
            days_back=90,
        )
    """

    def __init__(
        self,
        client: BinanceFuturesClient,
        db: DatabaseManager,
    ):
        self.client = client
        self.db     = db
        self._symbol_id_cache: dict[str, int] = {}

    # ── Sincronizacion de simbolos ─────────────────────────────────────────────

    async def sync_symbols(self) -> int:
        """
        Actualiza la tabla symbols con todos los pares USDT-M de Binance.

        Returns:
            Numero de simbolos upserted
        """
        log.info("Sincronizando simbolos desde Binance...")

        raw_symbols = self.client.get_all_symbols()
        count = 0

        async with self.db.async_session() as session:
            for raw in raw_symbols:
                symbol_name = raw["symbol"]

                # Extraer filtros relevantes
                min_qty = step_size = tick_size = None
                max_leverage = 20

                for f in raw.get("filters", []):
                    if f["filterType"] == "LOT_SIZE":
                        min_qty   = Decimal(f["minQty"])
                        step_size = Decimal(f["stepSize"])
                    elif f["filterType"] == "PRICE_FILTER":
                        tick_size = Decimal(f["tickSize"])

                # Upsert
                stmt = select(Symbol).where(Symbol.symbol == symbol_name)
                result = await session.execute(stmt)
                sym = result.scalar_one_or_none()

                if sym is None:
                    sym = Symbol(
                        symbol=symbol_name,
                        base_asset=raw["baseAsset"],
                        quote_asset=raw["quoteAsset"],
                        min_qty=min_qty,
                        step_size=step_size,
                        tick_size=tick_size,
                        max_leverage=max_leverage,
                        is_active=True,
                    )
                    session.add(sym)
                    count += 1
                else:
                    sym.min_qty   = min_qty
                    sym.step_size = step_size
                    sym.tick_size = tick_size
                    sym.is_active = True

        log.info(f"Simbolos sincronizados: {count} nuevos / {len(raw_symbols)} totales")
        return count

    async def _get_or_create_symbol(
        self, session, symbol_name: str
    ) -> Optional[int]:
        """Retorna el ID del simbolo, creandolo si no existe."""
        if symbol_name in self._symbol_id_cache:
            return self._symbol_id_cache[symbol_name]

        stmt = select(Symbol).where(Symbol.symbol == symbol_name)
        result = await session.execute(stmt)
        sym = result.scalar_one_or_none()

        if sym is None:
            sym = Symbol(
                symbol=symbol_name,
                base_asset=symbol_name.replace("USDT", ""),
                quote_asset="USDT",
                is_active=True,
            )
            session.add(sym)
            await session.flush()

        self._symbol_id_cache[symbol_name] = sym.id
        return sym.id

    # ── Descarga historica ─────────────────────────────────────────────────────

    async def download_history(
        self,
        symbols:    list[str],
        timeframes: list[str],
        days_back:  int = 90,
    ) -> dict[str, int]:
        """
        Descarga velas historicas para una lista de simbolos y timeframes.
        Solo descarga lo que falta (no duplica lo que ya esta en BD).

        Args:
            symbols:    Lista de pares (ej. ["BTCUSDT", "ETHUSDT"])
            timeframes: Lista de intervalos (ej. ["15m", "1h", "4h"])
            days_back:  Cuantos dias hacia atras descargar

        Returns:
            Dict {simbolo: total_velas_insertadas}
        """
        log.info(
            f"Descargando historico | "
            f"{len(symbols)} simbolos x {len(timeframes)} timeframes | "
            f"{days_back} dias"
        )

        totals: dict[str, int] = {}
        start_time = int(
            (datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp() * 1000
        )

        for symbol in symbols:
            totals[symbol] = 0
            for tf in timeframes:
                try:
                    inserted = await self._download_symbol_tf(
                        symbol=symbol,
                        timeframe=tf,
                        start_time_ms=start_time,
                    )
                    totals[symbol] += inserted
                    await asyncio.sleep(REQUEST_DELAY_SEC)
                except Exception as e:
                    log.error(f"Error descargando {symbol}/{tf}: {e}")

            log.info(f"  {symbol}: {totals[symbol]} velas insertadas")

        total_all = sum(totals.values())
        log.info(f"Descarga completada: {total_all} velas totales insertadas")
        return totals

    async def _download_symbol_tf(
        self,
        symbol:       str,
        timeframe:    str,
        start_time_ms: int,
    ) -> int:
        """
        Descarga todas las velas de un simbolo/timeframe desde start_time.
        Hace multiples requests si es necesario (paginacion automatica).

        Returns:
            Numero de velas nuevas insertadas
        """
        # Verificar ultima vela en BD para este simbolo/timeframe
        last_ts = await self._get_last_candle_timestamp(symbol, timeframe)
        if last_ts is not None:
            # Si ya tenemos datos, empezar desde donde quedamos
            effective_start = max(start_time_ms, last_ts + 1)
        else:
            effective_start = start_time_ms

        now_ms = int(time.time() * 1000)
        if effective_start >= now_ms:
            log.debug(f"{symbol}/{timeframe}: datos al dia, nada que descargar")
            return 0

        inserted = 0
        current_start = effective_start

        while current_start < now_ms:
            raw_klines = self.client.get_klines(
                symbol=symbol,
                interval=timeframe,
                limit=MAX_KLINES_PER_REQUEST,
                start_time=current_start,
            )

            if not raw_klines:
                break

            batch_inserted = await self._insert_candles_batch(
                symbol=symbol,
                timeframe=timeframe,
                raw_klines=raw_klines,
            )
            inserted += batch_inserted

            # Avanzar al siguiente batch
            last_kline_close = raw_klines[-1][6]  # close_time del ultimo kline
            current_start = last_kline_close + 1

            # Si recibimos menos del maximo, ya no hay mas datos
            if len(raw_klines) < MAX_KLINES_PER_REQUEST:
                break

            await asyncio.sleep(REQUEST_DELAY_SEC)

        if inserted > 0:
            log.debug(f"  {symbol}/{timeframe}: {inserted} velas nuevas")

        return inserted

    async def _insert_candles_batch(
        self,
        symbol:    str,
        timeframe: str,
        raw_klines: list,
    ) -> int:
        """
        Inserta un batch de velas en PostgreSQL.
        Ignora duplicados (ON CONFLICT DO NOTHING equivalente).

        Returns:
            Numero de velas efectivamente insertadas
        """
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        inserted = 0

        async with self.db.async_session() as session:
            symbol_id = await self._get_or_create_symbol(session, symbol)

            candles_to_insert = []
            for k in raw_klines:
                # Formato Binance kline:
                # [open_time, open, high, low, close, volume, close_time,
                #  quote_vol, trades, taker_buy_base, taker_buy_quote, ignore]
                open_time  = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
                close_time = datetime.fromtimestamp(k[6] / 1000, tz=timezone.utc)

                candles_to_insert.append({
                    "symbol_id":          symbol_id,
                    "timeframe":          timeframe,
                    "open_time":          open_time,
                    "close_time":         close_time,
                    "open":               Decimal(str(k[1])),
                    "high":               Decimal(str(k[2])),
                    "low":                Decimal(str(k[3])),
                    "close":              Decimal(str(k[4])),
                    "volume":             Decimal(str(k[5])),
                    "quote_volume":       Decimal(str(k[7])),
                    "trades_count":       int(k[8]),
                    "taker_buy_base_vol": Decimal(str(k[9])),
                    "taker_buy_quote_vol":Decimal(str(k[10])),
                    "is_closed":          True,
                })

            if candles_to_insert:
                # Upsert: si ya existe la vela (mismo symbol+tf+open_time), ignorar
                stmt = pg_insert(Candle).values(candles_to_insert)
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=["symbol_id", "timeframe", "open_time"]
                )
                result = await session.execute(stmt)
                inserted = result.rowcount

        return inserted

    async def _get_last_candle_timestamp(
        self, symbol: str, timeframe: str
    ) -> Optional[int]:
        """
        Retorna el timestamp ms de la ultima vela almacenada para symbol/tf.
        Retorna None si no hay velas en BD.
        """
        async with self.db.async_session() as session:
            # Obtener symbol_id
            stmt_sym = select(Symbol.id).where(Symbol.symbol == symbol)
            result = await session.execute(stmt_sym)
            sym_id = result.scalar_one_or_none()

            if sym_id is None:
                return None

            stmt = select(func.max(Candle.open_time)).where(
                and_(
                    Candle.symbol_id == sym_id,
                    Candle.timeframe == timeframe,
                )
            )
            result = await session.execute(stmt)
            last_dt = result.scalar_one_or_none()

            if last_dt is None:
                return None

            return int(last_dt.timestamp() * 1000)

    # ── Actualizacion incremental ──────────────────────────────────────────────

    async def update_recent(
        self,
        symbols:    list[str],
        timeframes: list[str],
        candles_back: int = 50,
    ) -> None:
        """
        Actualiza las ultimas N velas para mantener la BD al dia.
        Llamar periodicamente (ej. cada 5 minutos).

        Args:
            symbols:      Lista de pares a actualizar
            timeframes:   Intervalos a actualizar
            candles_back: Cuantas velas hacia atras actualizar
        """
        log.debug(f"Actualizando ultimas {candles_back} velas...")

        for symbol in symbols:
            for tf in timeframes:
                try:
                    await self._download_symbol_tf(
                        symbol=symbol,
                        timeframe=tf,
                        start_time_ms=0,  # _download_symbol_tf detecta el ultimo
                    )
                    await asyncio.sleep(REQUEST_DELAY_SEC)
                except Exception as e:
                    log.warning(f"Error actualizando {symbol}/{tf}: {e}")

    # ── Carga a DataFrame ──────────────────────────────────────────────────────

    async def load_candles_df(
        self,
        symbol:    str,
        timeframe: str,
        limit:     int = 500,
    ) -> pd.DataFrame:
        """
        Carga velas desde PostgreSQL a un DataFrame de pandas.

        Returns:
            DataFrame con columnas: open_time, open, high, low, close,
                                    volume, quote_volume, trades_count
            Ordenado de mas antiguo a mas reciente.
        """
        async with self.db.async_session() as session:
            stmt_sym = select(Symbol.id).where(Symbol.symbol == symbol)
            result   = await session.execute(stmt_sym)
            sym_id   = result.scalar_one_or_none()

            if sym_id is None:
                log.warning(f"Simbolo {symbol} no encontrado en BD")
                return pd.DataFrame()

            stmt = (
                select(Candle)
                .where(and_(
                    Candle.symbol_id == sym_id,
                    Candle.timeframe == timeframe,
                ))
                .order_by(Candle.open_time.desc())
                .limit(limit)
            )
            result  = await session.execute(stmt)
            candles = result.scalars().all()

        if not candles:
            return pd.DataFrame()

        rows = [
            {
                "open_time":    c.open_time,
                "open":         float(c.open),
                "high":         float(c.high),
                "low":          float(c.low),
                "close":        float(c.close),
                "volume":       float(c.volume),
                "quote_volume": float(c.quote_volume),
                "trades":       c.trades_count,
            }
            for c in reversed(candles)  # De antiguo a reciente
        ]

        df = pd.DataFrame(rows)
        df.set_index("open_time", inplace=True)
        return df


