"""
=============================================================
data/websocket_manager.py
Gestor de WebSocket para streams en tiempo real de Binance.

Maneja multiples streams simultaneos con:
  - Reconexion automatica con backoff exponencial
  - Callbacks por tipo de mensaje
  - Buffer en memoria para datos recientes
  - Soporte para: kline, bookTicker, aggTrade, markPrice
=============================================================
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from decimal import Decimal
from typing import Callable, Optional

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from utils.logger import get_logger

log = get_logger(__name__)

# URLs WebSocket de Binance Futuros
WS_BASE_TESTNET = "wss://stream.binancefuture.com/stream"
WS_BASE_LIVE    = "wss://fstream.binance.com/stream"


class StreamBuffer:
    """
    Buffer circular en memoria para los ultimos N ticks de cada simbolo.
    Permite acceso O(1) al precio/volumen mas reciente sin consultar la BD.
    """

    def __init__(self, maxlen: int = 200):
        self._prices:   dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self._volumes:  dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))
        self._klines:   dict[str, dict]  = {}   # ultima vela por simbolo+tf
        self._book:     dict[str, dict]  = {}   # mejor bid/ask por simbolo
        self._mark:     dict[str, Decimal] = {}  # mark price por simbolo

    def update_price(self, symbol: str, price: Decimal, volume: Decimal) -> None:
        self._prices[symbol].append((time.time(), float(price)))
        self._volumes[symbol].append(float(volume))

    def update_kline(self, symbol: str, timeframe: str, kline: dict) -> None:
        key = f"{symbol}_{timeframe}"
        self._klines[key] = kline

    def update_book(self, symbol: str, best_bid: Decimal, best_ask: Decimal) -> None:
        self._book[symbol] = {
            "bid": float(best_bid),
            "ask": float(best_ask),
            "spread": float(best_ask - best_bid),
            "mid":    float((best_bid + best_ask) / 2),
            "ts":     time.time(),
        }

    def update_mark_price(self, symbol: str, price: Decimal) -> None:
        self._mark[symbol] = price

    def get_last_price(self, symbol: str) -> Optional[float]:
        prices = self._prices.get(symbol)
        if prices:
            return prices[-1][1]
        return None

    def get_mark_price(self, symbol: str) -> Optional[Decimal]:
        return self._mark.get(symbol)

    def get_last_kline(self, symbol: str, timeframe: str) -> Optional[dict]:
        return self._klines.get(f"{symbol}_{timeframe}")

    def get_book(self, symbol: str) -> Optional[dict]:
        return self._book.get(symbol)

    def get_recent_prices(self, symbol: str, n: int = 20) -> list[float]:
        prices = self._prices.get(symbol, deque())
        return [p[1] for p in list(prices)[-n:]]


# Instancia global del buffer (compartida por todos los modulos)
stream_buffer = StreamBuffer()


class BinanceWebSocketManager:
    """
    Gestor de conexiones WebSocket a Binance Futuros.

    Soporta multiples streams combinados usando el endpoint /stream?streams=...
    con reconexion automatica y backoff exponencial.

    Uso:
        ws = BinanceWebSocketManager(testnet=True)

        # Suscribir a klines de 15m y book ticker
        await ws.subscribe(
            symbols=["BTCUSDT", "ETHUSDT"],
            streams=["kline_15m", "bookTicker"],
            callback=mi_callback,
        )

        await ws.start()
    """

    def __init__(self, testnet: bool = True):
        self.base_url = WS_BASE_TESTNET if testnet else WS_BASE_LIVE
        self._streams:   list[str] = []
        self._callbacks: dict[str, list[Callable]] = defaultdict(list)
        self._running    = False
        self._ws         = None
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        self._message_count = 0

    # ── Suscripciones ──────────────────────────────────────────────────────────

    def subscribe(
        self,
        symbols:   list[str],
        streams:   list[str],
        callback:  Optional[Callable] = None,
    ) -> None:
        """
        Registra streams para una lista de simbolos.

        Args:
            symbols:  Lista de pares (ej. ["BTCUSDT", "ETHUSDT"])
            streams:  Tipos de stream (ej. ["kline_15m", "bookTicker", "markPrice"])
            callback: Funcion que recibe (stream_type, symbol, data)
        """
        new_streams = []
        for symbol in symbols:
            sym = symbol.lower()
            for stream in streams:
                stream_name = f"{sym}@{stream}"
                if stream_name not in self._streams:
                    self._streams.append(stream_name)
                    new_streams.append(stream_name)

        if callback:
            for stream in streams:
                self._callbacks[stream].append(callback)

        log.info(
            f"Suscrito a {len(new_streams)} streams | "
            f"Total activos: {len(self._streams)}"
        )

    def add_callback(self, stream_type: str, callback: Callable) -> None:
        """Agrega un callback para un tipo especifico de stream."""
        self._callbacks[stream_type].append(callback)

    # ── Loop principal ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Inicia el loop de WebSocket con reconexion automatica."""
        self._running = True
        log.info(f"Iniciando WebSocket | {len(self._streams)} streams")

        while self._running:
            try:
                await self._connect_and_listen()
                self._reconnect_delay = 1.0  # Reset delay tras conexion exitosa
            except asyncio.CancelledError:
                log.info("WebSocket cancelado limpiamente")
                break
            except Exception as e:
                if not self._running:
                    break
                log.warning(
                    f"WebSocket desconectado: {e} | "
                    f"Reconectando en {self._reconnect_delay:.1f}s..."
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self._max_reconnect_delay
                )

    async def _connect_and_listen(self) -> None:
        """Establece la conexion y escucha mensajes."""
        if not self._streams:
            log.warning("No hay streams configurados")
            return

        # Construir URL con todos los streams combinados
        streams_param = "/".join(self._streams)
        url = f"{self.base_url}?streams={streams_param}"

        log.info(f"Conectando WebSocket: {url[:80]}...")

        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            self._ws = ws
            self._reconnect_delay = 1.0
            log.info(f"WebSocket conectado | {len(self._streams)} streams activos")

            async for raw_message in ws:
                if not self._running:
                    break
                await self._handle_message(raw_message)

    async def _handle_message(self, raw: str) -> None:
        """Parsea y despacha cada mensaje recibido."""
        try:
            msg = json.loads(raw)
            self._message_count += 1

            # Formato combinado: {"stream": "btcusdt@kline_15m", "data": {...}}
            stream_name = msg.get("stream", "")
            data        = msg.get("data", msg)

            # Extraer tipo y simbolo del nombre del stream
            # Ej: "btcusdt@kline_15m" -> symbol="BTCUSDT", type="kline_15m"
            if "@" in stream_name:
                parts = stream_name.split("@", 1)
                symbol      = parts[0].upper()
                stream_type = parts[1]
            else:
                symbol      = data.get("s", "").upper()
                stream_type = data.get("e", "unknown")

            # Actualizar buffer en memoria
            await self._update_buffer(symbol, stream_type, data)

            # Disparar callbacks registrados
            callbacks = self._callbacks.get(stream_type, [])
            for cb in callbacks:
                try:
                    if asyncio.iscoroutinefunction(cb):
                        await cb(stream_type, symbol, data)
                    else:
                        cb(stream_type, symbol, data)
                except Exception as e:
                    log.error(f"Error en callback {stream_type}: {e}")

            # Log periodico de actividad
            if self._message_count % 1000 == 0:
                log.debug(f"WebSocket: {self._message_count} mensajes procesados")

        except json.JSONDecodeError as e:
            log.warning(f"Mensaje WebSocket no es JSON valido: {e}")
        except Exception as e:
            log.error(f"Error procesando mensaje WebSocket: {e}")

    async def _update_buffer(
        self, symbol: str, stream_type: str, data: dict
    ) -> None:
        """Actualiza el StreamBuffer global segun el tipo de mensaje."""

        # ── Kline (vela OHLCV) ─────────────────────────────────────────────────
        if stream_type.startswith("kline_"):
            k = data.get("k", {})
            timeframe = k.get("i", stream_type.replace("kline_", ""))
            kline_data = {
                "open_time":  k.get("t"),
                "close_time": k.get("T"),
                "open":   Decimal(str(k.get("o", 0))),
                "high":   Decimal(str(k.get("h", 0))),
                "low":    Decimal(str(k.get("l", 0))),
                "close":  Decimal(str(k.get("c", 0))),
                "volume": Decimal(str(k.get("v", 0))),
                "trades": k.get("n", 0),
                "closed": k.get("x", False),
            }
            stream_buffer.update_kline(symbol, timeframe, kline_data)

            # Tambien actualizar precio con el close actual
            stream_buffer.update_price(
                symbol,
                kline_data["close"],
                kline_data["volume"],
            )

        # ── Book Ticker (mejor bid/ask) ────────────────────────────────────────
        elif stream_type == "bookTicker":
            best_bid = Decimal(str(data.get("b", 0)))
            best_ask = Decimal(str(data.get("a", 0)))
            if best_bid > 0 and best_ask > 0:
                stream_buffer.update_book(symbol, best_bid, best_ask)

        # ── Mark Price ────────────────────────────────────────────────────────
        elif stream_type == "markPrice":
            mark = Decimal(str(data.get("p", 0)))
            if mark > 0:
                stream_buffer.update_mark_price(symbol, mark)

        # ── Aggregate Trade ───────────────────────────────────────────────────
        elif stream_type == "aggTrade":
            price  = Decimal(str(data.get("p", 0)))
            volume = Decimal(str(data.get("q", 0)))
            if price > 0:
                stream_buffer.update_price(symbol, price, volume)

    # ── Control ───────────────────────────────────────────────────────────────

    async def stop(self) -> None:
        """Detiene el WebSocket limpiamente."""
        self._running = False
        if self._ws:
            await self._ws.close()
        log.info("WebSocket detenido")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stream_count(self) -> int:
        return len(self._streams)

    @property
    def message_count(self) -> int:
        return self._message_count
