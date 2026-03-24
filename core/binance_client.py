"""
=============================================================
core/binance_client.py
Cliente de Binance Futuros USDT-M con soporte para:
  - Testnet (testnet.binancefuture.com)
  - Dry Run (simulación local sin llamadas reales)
  - Live (producción real)

Proporciona métodos de alto nivel con manejo de errores,
reintentos automáticos y logging detallado.
=============================================================
"""

import asyncio
import time
from datetime import datetime
from decimal import Decimal
from typing import Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from binance.enums import (
    FUTURE_ORDER_TYPE_MARKET,
    FUTURE_ORDER_TYPE_LIMIT,
    SIDE_BUY,
    SIDE_SELL,
)

from config.settings import Settings, TradingMode
from utils.logger import get_logger

log = get_logger(__name__)


# ── URLs de Binance ────────────────────────────────────────────────────────────

TESTNET_BASE_URL  = "https://testnet.binancefuture.com"
TESTNET_WS_URL    = "wss://stream.binancefuture.com"
LIVE_FUTURES_URL  = "https://fapi.binance.com"


# ── Datos de cuenta simulada para dry_run ─────────────────────────────────────

class DryRunAccount:
    """Simula el estado de la cuenta en modo dry run."""

    def __init__(self, initial_balance: float = 1000.0):
        self.balance = Decimal(str(initial_balance))
        self.available_balance = Decimal(str(initial_balance))
        self.unrealized_pnl = Decimal("0")
        self.positions: dict = {}
        self.orders: list = []
        self._order_counter = 0

    def get_balance_info(self) -> dict:
        return {
            "totalWalletBalance": str(self.balance),
            "availableBalance": str(self.available_balance),
            "totalUnrealizedProfit": str(self.unrealized_pnl),
            "totalMarginBalance": str(self.balance + self.unrealized_pnl),
        }

    def create_mock_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        order_type: str = "MARKET",
    ) -> dict:
        self._order_counter += 1
        order_id = f"DRY_{self._order_counter}_{int(time.time())}"
        return {
            "orderId": order_id,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "origQty": str(quantity),
            "avgPrice": str(price),
            "status": "FILLED",
            "updateTime": int(time.time() * 1000),
        }


# ── Cliente principal ─────────────────────────────────────────────────────────

class BinanceFuturesClient:
    """
    Cliente de alto nivel para Binance Futuros USDT-M.

    Abstrae la complejidad de la API y unifica testnet/live/dry_run
    con la misma interfaz. Todo el código de trading usa este cliente.

    Uso:
        client = BinanceFuturesClient(settings)
        await client.initialize()

        # Obtener precio actual
        price = await client.get_mark_price("BTCUSDT")

        # Colocar orden de mercado
        order = await client.place_market_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.001
        )
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.mode = settings.trading_mode
        self._client: Optional[Client] = None
        self._dry_run_account: Optional[DryRunAccount] = None
        self._initialized = False

        log.info(f"BinanceFuturesClient creado en modo: {self.mode.value.upper()}")

    # ── Inicialización ─────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Comportamiento por modo:

        DRY_RUN:  Conecta a la API LIVE de Binance (sin autenticacion)
                  para obtener precios, simbolos y velas reales.
                  El balance viene de INITIAL_CAPITAL y las ordenes son simuladas.

        TESTNET:  Conecta a testnet.binancefuture.com con credenciales testnet.
                  Balance, precios y ordenes son del entorno de pruebas de Binance.

        LIVE:     Conecta a fapi.binance.com con credenciales reales.
                  Todo es real — usar con precaucion.
        """
        if self.mode == TradingMode.DRY_RUN:
            self._dry_run_account = DryRunAccount(
                initial_balance=self.settings.initial_capital
            )
            # Conectar a la API publica LIVE para datos de mercado reales
            # Las rutas publicas (klines, ticker, exchange_info) no requieren auth
            try:
                self._client = Client("", "")  # Sin credenciales para datos publicos
                self._client.futures_ping()
                log.info(
                    f"✓ Modo DRY RUN | "
                    f"Capital simulado: ${self.settings.initial_capital:,.2f} USDT | "
                    f"Datos de mercado: Binance LIVE (real)"
                )
            except Exception:
                # Si no hay red, el cliente queda None y los metodos de mercado
                # devolveran listas vacias con un warning
                self._client = None
                log.warning(
                    "DRY RUN sin conexion a internet — "
                    "datos de mercado no disponibles"
                )
            self._initialized = True
            return

        try:
            kwargs = {
                "api_key":         self.settings.active_api_key,
                "api_secret":      self.settings.active_api_secret,
                "requests_params": {"timeout": 30},
            }

            if self.mode == TradingMode.TESTNET:
                kwargs["testnet"] = True
                log.info("Conectando a Binance Futures TESTNET...")
            else:
                log.warning("⚠️  Conectando a Binance Futures LIVE (dinero real)")

            self._client = Client(**kwargs)
            self._client.futures_ping()
            server_time = self._client.futures_time()
            log.info(
                f"✓ Conexion Binance OK | "
                f"Server time: {datetime.fromtimestamp(server_time['serverTime'] / 1000)}"
            )
            self._sync_time()
            self._initialized = True

        except BinanceAPIException as e:
            log.error(f"✗ Error de API Binance: {e.status_code} - {e.message}")
            raise
        except Exception as e:
            log.error(f"✗ Error inicializando cliente Binance: {e}")
            raise

    def _sync_time(self) -> None:
        """Sincroniza el timestamp con el servidor para evitar errores."""
        try:
            server_time = self._client.futures_time()["serverTime"]
            local_time = int(time.time() * 1000)
            drift_ms = server_time - local_time
            if abs(drift_ms) > 1000:
                log.warning(f"Drift de tiempo con servidor Binance: {drift_ms}ms")
            else:
                log.debug(f"Drift de tiempo: {drift_ms}ms (OK)")
        except Exception as e:
            log.warning(f"No se pudo sincronizar tiempo: {e}")

    # ── Información de mercado ─────────────────────────────────────────────────

    def get_mark_price(self, symbol: str) -> Decimal:
        """Precio mark actual. En dry_run usa la API publica real."""
        if self._client is None:
            return Decimal("0")
        try:
            data  = self._client.futures_mark_price(symbol=symbol)
            price = Decimal(data["markPrice"])
            log.debug(f"Mark price {symbol}: {price}")
            return price
        except BinanceAPIException as e:
            log.error(f"Error obteniendo mark price {symbol}: {e}")
            raise

    def get_all_symbols(self) -> list[dict]:
        """
        Obtiene todos los pares USDT-M disponibles.
        En dry_run usa la API publica real de Binance (sin autenticacion).
        """
        if self._client is None:
            return []

        try:
            info = self._client.futures_exchange_info()
            symbols = [
                s for s in info["symbols"]
                if s["quoteAsset"] == "USDT"
                and s["status"] == "TRADING"
                and s["contractType"] == "PERPETUAL"
            ]
            log.info(f"✓ {len(symbols)} pares USDT-M perpetuos encontrados")
            return symbols
        except BinanceAPIException as e:
            log.error(f"Error obteniendo símbolos: {e}")
            raise

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> list:
        """
        Obtiene datos OHLCV historicos.
        En dry_run usa la API publica real de Binance.
        """
        if self._client is None:
            return []

        kwargs = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            kwargs["startTime"] = start_time
        if end_time:
            kwargs["endTime"] = end_time

        try:
            klines = self._client.futures_klines(**kwargs)
            log.debug(f"✓ {len(klines)} velas {interval} obtenidas para {symbol}")
            return klines
        except BinanceAPIException as e:
            log.error(f"Error obteniendo klines {symbol}/{interval}: {e}")
            raise

    def get_order_book(self, symbol: str, limit: int = 20) -> dict:
        """Libro de ordenes actual. En dry_run usa la API publica real."""
        if self._client is None:
            return {"bids": [], "asks": [], "lastUpdateId": 0}

        try:
            return self._client.futures_order_book(symbol=symbol, limit=limit)
        except BinanceAPIException as e:
            log.error(f"Error obteniendo order book {symbol}: {e}")
            raise

    def get_ticker_24h(self, symbol: Optional[str] = None) -> list[dict] | dict:
        """
        Estadisticas 24h. En dry_run usa la API publica real de Binance.
        Si symbol es None, retorna todos los tickers.
        """
        if self._client is None:
            return [] if symbol is None else {}

        try:
            if symbol:
                return self._client.futures_ticker(symbol=symbol)
            return self._client.futures_ticker()
        except BinanceAPIException as e:
            log.error(f"Error obteniendo ticker 24h: {e}")
            raise

    # ── Información de cuenta ──────────────────────────────────────────────────

    def get_account_balance(self) -> dict:
        """
        Obtiene el balance actual de la cuenta.

        Returns:
            Dict con totalWalletBalance, availableBalance, unrealizedProfit
        """
        if self.mode == TradingMode.DRY_RUN:
            return self._dry_run_account.get_balance_info()

        try:
            account = self._client.futures_account()
            usdt_asset = next(
                (a for a in account["assets"] if a["asset"] == "USDT"),
                None
            )
            if usdt_asset:
                return {
                    "totalWalletBalance": usdt_asset["walletBalance"],
                    "availableBalance": usdt_asset["availableBalance"],
                    "totalUnrealizedProfit": account["totalUnrealizedProfit"],
                    "totalMarginBalance": account["totalMarginBalance"],
                }
            return {}
        except BinanceAPIException as e:
            log.error(f"Error obteniendo balance: {e}")
            raise

    def get_funding_rate(self, symbol: str) -> dict:
        """
        Obtiene el funding rate actual y el proximo para un simbolo.

        Returns:
            Dict con:
              - funding_rate: tasa actual (ej. 0.0001 = 0.01%)
              - next_funding_time: timestamp del proximo cobro
              - mark_price: precio mark actual
        """
        if self._client is None:
            return {"funding_rate": 0.0, "next_funding_time": 0, "mark_price": 0.0}

        try:
            data = self._client.futures_mark_price(symbol=symbol)
            return {
                "funding_rate":      float(data.get("lastFundingRate", 0)),
                "next_funding_time": int(data.get("nextFundingTime", 0)),
                "mark_price":        float(data.get("markPrice", 0)),
            }
        except Exception as e:
            log.warning(f"Error obteniendo funding rate {symbol}: {e}")
            return {"funding_rate": 0.0, "next_funding_time": 0, "mark_price": 0.0}

    def get_funding_rates_batch(self, symbols: list[str]) -> dict[str, float]:
        """
        Obtiene el funding rate de multiples simbolos en una sola llamada.

        Returns:
            Dict {symbol: funding_rate}
        """
        if self._client is None:
            return {s: 0.0 for s in symbols}

        try:
            all_data = self._client.futures_mark_price()
            rates = {
                item["symbol"]: float(item.get("lastFundingRate", 0))
                for item in all_data
                if item["symbol"] in symbols
            }
            return rates
        except Exception as e:
            log.warning(f"Error obteniendo funding rates batch: {e}")
            return {s: 0.0 for s in symbols}

    def get_open_positions(self) -> list[dict]:
        """
        Retorna las posiciones abiertas activas (cantidad != 0).

        Returns:
            Lista de posiciones abiertas con info completa
        """
        if self.mode == TradingMode.DRY_RUN:
            return list(self._dry_run_account.positions.values())

        try:
            positions = self._client.futures_position_information()
            open_pos = [
                p for p in positions
                if float(p.get("positionAmt", 0)) != 0
            ]
            return open_pos
        except BinanceAPIException as e:
            log.error(f"Error obteniendo posiciones: {e}")
            raise

    # ── Gestión de órdenes ─────────────────────────────────────────────────────

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Establece el apalancamiento para un símbolo.

        Args:
            symbol:   Par de trading
            leverage: Apalancamiento (1-125)

        Returns:
            True si fue exitoso
        """
        if self.mode == TradingMode.DRY_RUN:
            log.info(f"[DRY RUN] Leverage {symbol} → {leverage}x")
            return True

        try:
            self._client.futures_change_leverage(symbol=symbol, leverage=leverage)
            log.info(f"✓ Leverage {symbol} configurado: {leverage}x")
            return True
        except BinanceAPIException as e:
            log.error(f"Error configurando leverage {symbol}: {e}")
            return False

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        reduce_only: bool = False,
    ) -> dict:
        """
        Coloca una orden de mercado en Futuros.

        Args:
            symbol:      Par de trading (ej. "BTCUSDT")
            side:        "BUY" o "SELL"
            quantity:    Cantidad del activo base
            reduce_only: Si True, solo cierra posición existente

        Returns:
            Dict con la respuesta de la orden de Binance
        """
        if self.mode == TradingMode.DRY_RUN:
            # Obtener el precio real del mercado desde el StreamBuffer
            from data.websocket_manager import stream_buffer
            live_price = stream_buffer.get_last_price(symbol)

            # Si no hay precio en el buffer todavia, obtenerlo via API publica
            if not live_price or live_price <= 0:
                try:
                    ticker = self._client.futures_mark_price(symbol=symbol)
                    live_price = float(ticker["markPrice"])
                except Exception:
                    live_price = 0.0

            if not live_price or live_price <= 0:
                log.warning(f"[DRY RUN] No se pudo obtener precio real de {symbol}")
                live_price = 1.0  # fallback de emergencia

            order = self._dry_run_account.create_mock_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=Decimal(str(live_price)),
                order_type="MARKET",
            )
            log.info(
                f"[DRY RUN] Orden MARKET {side} {quantity} {symbol} "
                f"@ ${live_price:.6f} → ID: {order['orderId']}"
            )
            return order

        try:
            params = {
                "symbol": symbol,
                "side": side,
                "type": FUTURE_ORDER_TYPE_MARKET,
                "quantity": str(quantity),
            }
            if reduce_only:
                params["reduceOnly"] = "true"

            order = self._client.futures_create_order(**params)
            log.info(
                f"✓ Orden MARKET {side} {quantity} {symbol} | "
                f"ID: {order['orderId']} | Status: {order['status']}"
            )
            return order
        except BinanceAPIException as e:
            log.error(f"✗ Error colocando orden MARKET {side} {symbol}: {e}")
            raise

    def place_stop_loss_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        stop_price: Decimal,
    ) -> dict:
        """
        Coloca una orden STOP_MARKET para stop loss.

        Args:
            symbol:     Par de trading
            side:       "BUY" para cubrir un short, "SELL" para cubrir un long
            quantity:   Cantidad a cerrar
            stop_price: Precio de activación del stop

        Returns:
            Dict con la respuesta de Binance
        """
        if self.mode == TradingMode.DRY_RUN:
            order = self._dry_run_account.create_mock_order(
                symbol=symbol, side=side, quantity=quantity,
                price=stop_price, order_type="STOP_MARKET"
            )
            log.info(f"[DRY RUN] Stop Loss {side} {symbol} @ {stop_price}")
            return order

        try:
            order = self._client.futures_create_order(
                symbol=symbol,
                side=side,
                type="STOP_MARKET",
                stopPrice=str(stop_price),
                quantity=str(quantity),
                reduceOnly=True,
                workingType="MARK_PRICE",
            )
            log.info(f"✓ Stop Loss colocado {symbol} @ {stop_price}")
            return order
        except BinanceAPIException as e:
            log.error(f"✗ Error colocando Stop Loss {symbol}: {e}")
            raise

    def place_take_profit_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        take_profit_price: Decimal,
    ) -> dict:
        """Coloca una orden TAKE_PROFIT_MARKET."""
        if self.mode == TradingMode.DRY_RUN:
            order = self._dry_run_account.create_mock_order(
                symbol=symbol, side=side, quantity=quantity,
                price=take_profit_price, order_type="TAKE_PROFIT_MARKET"
            )
            log.info(f"[DRY RUN] Take Profit {side} {symbol} @ {take_profit_price}")
            return order

        try:
            order = self._client.futures_create_order(
                symbol=symbol,
                side=side,
                type="TAKE_PROFIT_MARKET",
                stopPrice=str(take_profit_price),
                quantity=str(quantity),
                reduceOnly=True,
                workingType="MARK_PRICE",
            )
            log.info(f"✓ Take Profit colocado {symbol} @ {take_profit_price}")
            return order
        except BinanceAPIException as e:
            log.error(f"✗ Error colocando Take Profit {symbol}: {e}")
            raise

    def cancel_all_orders(self, symbol: str) -> bool:
        """Cancela todas las órdenes abiertas de un símbolo."""
        if self.mode == TradingMode.DRY_RUN:
            log.info(f"[DRY RUN] Cancelando todas las órdenes de {symbol}")
            return True

        try:
            self._client.futures_cancel_all_open_orders(symbol=symbol)
            log.info(f"✓ Todas las órdenes de {symbol} canceladas")
            return True
        except BinanceAPIException as e:
            log.error(f"Error cancelando órdenes {symbol}: {e}")
            return False

    # ── Utilidades ────────────────────────────────────────────────────────────

    @property
    def is_dry_run(self) -> bool:
        return self.mode == TradingMode.DRY_RUN

    @property
    def is_ready(self) -> bool:
        """Indica si el cliente está inicializado y listo."""
        return self._initialized

    def get_server_time(self) -> datetime:
        """Obtiene el tiempo del servidor de Binance."""
        if self.mode == TradingMode.DRY_RUN:
            return datetime.utcnow()
        ts = self._client.futures_time()["serverTime"]
        return datetime.fromtimestamp(ts / 1000)

    def __repr__(self) -> str:
        return (
            f"<BinanceFuturesClient mode={self.mode.value} "
            f"ready={self._initialized}>"
        )
