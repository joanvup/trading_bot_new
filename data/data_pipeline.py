"""
=============================================================
data/data_pipeline.py
Orquestador del pipeline de datos.

Coordina todos los modulos de datos en un loop unico:
  1. Arranque: sincroniza simbolos + descarga historico
  2. Loop de WebSocket: mantiene datos en tiempo real
  3. Loop de actualizacion: refresca velas periodicamente
  4. Loop de ranking: recalcula top simbolos cada 5min

Este es el componente que se instancia desde main.py
=============================================================
"""

import asyncio
from datetime import datetime, timezone

from core.binance_client import BinanceFuturesClient
from database.connection import DatabaseManager
from data.websocket_manager import BinanceWebSocketManager, stream_buffer
from data.historic_collector import HistoricCollector
from analysis.indicators import TechnicalIndicators
from analysis.asset_selector import AssetSelector, SymbolScore
from config.settings import Settings
from utils.logger import get_logger

log = get_logger(__name__)


class DataPipeline:
    """
    Orquestador principal de todos los datos del bot.

    Gestiona el ciclo de vida completo:
        - WebSocket para datos en tiempo real
        - Descarga y actualizacion de historico OHLCV
        - Calculo de indicadores tecnicos
        - Ranking dinamico de activos

    Uso (desde main.py):
        pipeline = DataPipeline(settings, client, db)
        await pipeline.start()              # Arranque inicial
        await pipeline.run_forever()        # Loop continuo
        top = pipeline.get_top_symbols()    # Obtener ranking actual
        df = await pipeline.get_df("BTCUSDT", "15m")  # Obtener datos con indicadores
    """

    def __init__(
        self,
        settings: Settings,
        client:   BinanceFuturesClient,
        db:       DatabaseManager,
    ):
        self.settings = settings
        self.client   = client
        self.db       = db

        # Componentes
        self.ti        = TechnicalIndicators()
        self.collector = HistoricCollector(client, db)
        self.selector  = AssetSelector(
            client=client,
            indicators=self.ti,
            top_n=settings.top_symbols_count,
        )
        self.ws_manager = BinanceWebSocketManager(
            # testnet=True solo si el modo es TESTNET
            # dry_run usa streams reales de Binance (datos públicos)
            testnet=settings.is_testnet
        )

        # Estado
        self._running       = False
        self._top_symbols:  list[SymbolScore] = []
        self._ws_task       = None
        self._update_task   = None
        self._ranking_task  = None

        # Timeframes a mantener actualizados
        self.timeframes = [
            settings.primary_timeframe.value,
            settings.secondary_timeframe.value,
        ]

    # ── Arranque ───────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Inicializa el pipeline completo.
        Debe llamarse una sola vez antes de run_forever().
        """
        log.info("Iniciando pipeline de datos...")

        # 1. Sincronizar metadatos de simbolos en BD
        log.info("Paso 1/4: Sincronizando simbolos...")
        await self.collector.sync_symbols()

        # 2. Obtener ranking inicial de activos
        log.info("Paso 2/4: Calculando ranking inicial de activos...")
        self._top_symbols = await self.selector.get_top_symbols()
        top_names = [s.symbol for s in self._top_symbols]
        log.info(f"Top simbolos: {top_names[:5]}... (+{len(top_names)-5} mas)")

        # 3. Descargar historico de los top simbolos
        # dry_run usa la API publica real de Binance — se descarga igual que testnet/live
        log.info("Paso 3/4: Descargando historico OHLCV...")
        await self.collector.download_history(
            symbols=top_names,
            timeframes=self.timeframes,
            days_back=90,
        )

        # 4. Suscribir WebSocket a los top simbolos
        # dry_run usa los streams publicos reales de Binance
        log.info("Paso 4/4: Configurando WebSocket...")
        self.ws_manager.subscribe(
            symbols=top_names,
            streams=[
                f"kline_{self.settings.primary_timeframe.value}",
                f"kline_{self.settings.secondary_timeframe.value}",
                "bookTicker",
                "markPrice",
            ],
            callback=self._on_ws_message,
        )

        log.info("Pipeline de datos listo")

    async def run_forever(self) -> None:
        """
        Loop principal del pipeline.
        Corre hasta que se llame a stop().
        """
        self._running = True
        tasks = []

        # WebSocket y actualizacion de velas activos en todos los modos
        # dry_run usa streams/API publicos reales de Binance
        self._ws_task = asyncio.create_task(
            self.ws_manager.start(),
            name="websocket",
        )
        tasks.append(self._ws_task)

        self._update_task = asyncio.create_task(
            self._update_candles_loop(),
            name="candle_updater",
        )
        tasks.append(self._update_task)

        # Task de actualizacion del ranking (siempre activo)
        self._ranking_task = asyncio.create_task(
            self._ranking_loop(),
            name="ranking_updater",
        )
        tasks.append(self._ranking_task)

        log.info(f"Pipeline corriendo con {len(tasks)} tasks activos")

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            log.info("Pipeline cancelado")
        except Exception as e:
            log.error(f"Error en pipeline: {e}", exc_info=True)
            raise

    # ── Loops internos ─────────────────────────────────────────────────────────

    async def _update_candles_loop(self) -> None:
        """Actualiza velas en BD cada 5 minutos."""
        while self._running:
            try:
                await asyncio.sleep(300)  # 5 minutos
                top_names = self.get_top_symbol_names()
                if top_names:
                    log.debug("Actualizando velas recientes en BD...")
                    await self.collector.update_recent(
                        symbols=top_names,
                        timeframes=self.timeframes,
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error actualizando velas: {e}")

    async def _ranking_loop(self) -> None:
        """Recalcula el ranking de activos cada 5 minutos."""
        while self._running:
            try:
                await asyncio.sleep(300)  # 5 minutos
                log.debug("Actualizando ranking de activos...")
                new_top = await self.selector.get_top_symbols(force_refresh=True)

                # Detectar cambios en el top
                old_names = set(self.get_top_symbol_names())
                new_names = set(s.symbol for s in new_top)
                added   = new_names - old_names
                removed = old_names - new_names

                if added or removed:
                    log.info(
                        f"Cambios en el ranking: "
                        f"+{list(added)} / -{list(removed)}"
                    )
                    # Suscribir nuevos simbolos al WebSocket
                    if added and not self.client.is_dry_run:
                        self.ws_manager.subscribe(
                            symbols=list(added),
                            streams=[
                                f"kline_{self.settings.primary_timeframe.value}",
                                "bookTicker",
                            ],
                        )
                        # Descargar historico de nuevos simbolos
                        await self.collector.download_history(
                            symbols=list(added),
                            timeframes=self.timeframes,
                            days_back=30,
                        )

                self._top_symbols = new_top

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error actualizando ranking: {e}")

    async def _on_ws_message(
        self, stream_type: str, symbol: str, data: dict
    ) -> None:
        """
        Callback para mensajes de WebSocket.
        Aqui se pueden encadenar acciones ante velas cerradas.
        """
        # Solo actuar sobre velas cerradas (no actualizaciones parciales)
        if stream_type.startswith("kline_"):
            k = data.get("k", {})
            if k.get("x", False):  # vela cerrada
                tf = k.get("i", "")
                log.debug(f"Vela cerrada: {symbol}/{tf}")
                # En Fase 3, aqui se disparara el analisis de ML

    # ── Acceso a datos ─────────────────────────────────────────────────────────

    def get_top_symbols(self) -> list[SymbolScore]:
        """Retorna el ranking actual de activos."""
        return self._top_symbols

    def get_top_symbol_names(self, n: int | None = None) -> list[str]:
        """Retorna los nombres del top N simbolos."""
        top = self._top_symbols[:n] if n else self._top_symbols
        return [s.symbol for s in top]

    async def get_df(
        self,
        symbol:    str,
        timeframe: str,
        limit:     int = 300,
        with_indicators: bool = True,
    ) -> "pd.DataFrame":
        """
        Retorna un DataFrame listo para analisis/ML con indicadores calculados.

        Args:
            symbol:          Par de trading
            timeframe:       Intervalo (ej "15m", "1h")
            limit:           Numero de velas a cargar
            with_indicators: Si True, calcula todos los indicadores TA

        Returns:
            DataFrame con OHLCV + indicadores, o DataFrame vacio si no hay datos
        """
        df = await self.collector.load_candles_df(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
        )

        if df.empty:
            log.warning(f"No hay datos en BD para {symbol}/{timeframe}")
            return df

        if with_indicators:
            df = self.ti.add_all(df)

        return df

    def get_live_price(self, symbol: str) -> float | None:
        """Retorna el ultimo precio del stream en tiempo real."""
        return stream_buffer.get_last_price(symbol)

    def get_live_book(self, symbol: str) -> dict | None:
        """Retorna el mejor bid/ask del stream en tiempo real."""
        return stream_buffer.get_book(symbol)

    # ── Control ────────────────────────────────────────────────────────────────

    async def stop(self) -> None:
        """Detiene el pipeline limpiamente."""
        self._running = False

        if self._ws_task:
            self._ws_task.cancel()
        if self._update_task:
            self._update_task.cancel()
        if self._ranking_task:
            self._ranking_task.cancel()

        await self.ws_manager.stop()
        log.info("Pipeline de datos detenido")

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> dict:
        """Retorna el estado actual del pipeline para el dashboard."""
        age = self.selector.last_update_age_secs
        return {
            "running":         self._running,
            "top_symbols":     len(self._top_symbols),
            "ws_streams":      self.ws_manager.stream_count,
            "ws_messages":     self.ws_manager.message_count,
            "ranking_age_sec": int(age) if age != float("inf") else -1,
            "top_5": [s.to_dict() for s in self._top_symbols[:5]],
        }
