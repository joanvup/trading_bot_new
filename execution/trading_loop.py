"""
=============================================================
execution/trading_loop.py
Loop principal de trading — orquesta todas las fases.

Por cada vela cerrada del timeframe primario:
  1. Sincronizar balance y posiciones abiertas
  2. Monitorear posiciones abiertas (trailing, SL/TP hits)
  3. Para cada top simbolo sin posicion: predict + evaluate
  4. Ejecutar decisiones aprobadas
  5. Guardar snapshot periodico

Este modulo conecta: DataPipeline -> Predictor -> StrategyEngine
                     -> OrderExecutor -> PortfolioManager
=============================================================
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional

from config.settings import Settings
from data.data_pipeline import DataPipeline
from ai.predictor import RealTimePredictor
from ai.model_trainer import ModelTrainer
from ai.auto_retrain import AutoRetrainer
from strategy.strategy_engine import StrategyEngine
from execution.order_executor import OrderExecutor
from execution.portfolio_manager import PortfolioManager
from risk.risk_manager import RiskManager
from risk.sl_tp_calculator import SLTPCalculator
from database.connection import DatabaseManager
from core.binance_client import BinanceFuturesClient
from utils.logger import get_logger

log = get_logger(__name__)


class TradingLoop:
    """
    Orquestador principal que une todas las fases del bot.

    Instancia y conecta todos los componentes, luego corre
    el loop de trading respondiendo a velas cerradas.

    Uso (desde main.py):
        loop = TradingLoop(settings, client, db)
        await loop.initialize()
        await loop.run_forever()
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

        # Instanciar todos los componentes
        self.risk_mgr  = RiskManager(settings)
        self.sl_tp     = SLTPCalculator(
            atr_sl_multiplier  = settings.sl_atr_multiplier,
            atr_tp_multiplier  = settings.tp_atr_multiplier,
            min_risk_reward    = settings.min_risk_reward,
            min_sl_pct         = settings.min_sl_pct,
            max_sl_pct         = settings.max_sl_pct,
        )
        self.executor  = OrderExecutor(settings, client, db)
        self.pipeline  = DataPipeline(settings, client, db)
        self.predictor = RealTimePredictor(
            settings=settings,
            db=db,
            min_confidence=0.58,
        )
        self.strategy  = StrategyEngine(
            settings=settings,
            risk_mgr=self.risk_mgr,
            sl_tp_calc=self.sl_tp,
            leverage=settings.max_leverage,
            client=client,
        )
        self.portfolio = PortfolioManager(
            settings=settings,
            client=client,
            db=db,
            executor=self.executor,
            risk_mgr=self.risk_mgr,
        )
        self.trainer   = ModelTrainer(settings, db)
        self.retrainer = AutoRetrainer(
            settings=settings,
            db=db,
            trainer=self.trainer,
            predictor=self.predictor,
        )

        self._running     = False
        self._cycle_count = 0
        self._last_candle_times: dict[str, int] = {}

    # ── Inicializacion ────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Inicializa todos los componentes en orden."""
        log.info("Inicializando TradingLoop...")

        # 1. Pipeline de datos
        await self.pipeline.start()

        # 2. Cargar modelo IA
        model_ready = await self.predictor.initialize()
        if not model_ready:
            log.warning(
                "Modelo no encontrado. Ejecutando entrenamiento inicial..."
            )
            symbols = self.pipeline.get_top_symbol_names(10)
            if symbols and not self.client.is_dry_run:
                await self.trainer.train(
                    symbols=symbols,
                    timeframe=self.settings.primary_timeframe.value,
                    use_optuna=False,  # Rapido para el primer entrenamiento
                )
                await self.predictor.initialize()
            else:
                log.warning(
                    "Sin datos para entrenar en dry_run. "
                    "El bot correra sin modelo (solo HOLD)."
                )

        # 3. Sincronizar balance inicial
        await self.portfolio.sync()

        # 3. Reconciliar posiciones abiertas con Binance
        log.info("Reconciliando posiciones con Binance...")
        recovered = await self.executor.reconcile_positions()
        if recovered > 0:
            log.info(
                f"Se recuperaron {recovered} posicion(es) de la sesion anterior. "
                f"El bot las gestionara normalmente."
            )

        log.info(
            f"TradingLoop listo | "
            f"Modelo: {'OK' if self.predictor.is_ready else 'NO DISPONIBLE'} | "
            f"Balance: ${self.risk_mgr.state.total_balance:.2f} | "
            f"Posiciones recuperadas: {recovered}"
        )

    # ── Loop principal ─────────────────────────────────────────────────────────

    async def run_forever(self) -> None:
        """
        Loop principal de trading.
        Corre hasta recibir senal de parada.
        """
        self._running = True

        # Tareas paralelas
        tasks = [
            asyncio.create_task(self.pipeline.run_forever(),       name="data_pipeline"),
            asyncio.create_task(self._position_monitor_loop(),     name="position_monitor"),
            asyncio.create_task(self._signal_analysis_loop(),      name="signal_analysis"),
            asyncio.create_task(self._portfolio_snapshot_loop(),   name="snapshots"),
            asyncio.create_task(
                self.retrainer.run_forever(self.pipeline.get_top_symbol_names),
                name="auto_retrain"
            ),
        ]

        log.info(f"TradingLoop corriendo con {len(tasks)} tareas paralelas")

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            log.info("TradingLoop cancelado")
        finally:
            for t in tasks:
                t.cancel()

    async def _position_monitor_loop(self) -> None:
        """
        Loop RAPIDO — corre cada 10 segundos.
        Solo revisa posiciones abiertas: SL hit, TP hit, trailing stop.
        Usa el precio en tiempo real del WebSocket (sin consultar la BD).
        """
        log.info("Monitor de posiciones iniciado (cada 10s)")

        while self._running:
            try:
                if self.executor.open_count > 0:
                    await self._check_open_positions()
                await asyncio.sleep(10)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error en monitor de posiciones: {e}", exc_info=True)
                await asyncio.sleep(10)

    async def _check_open_positions(self) -> None:
        """
        Revisa todas las posiciones abiertas con precio en tiempo real.
        Se ejecuta cada 10 segundos desde _position_monitor_loop.

        Orden de prioridad:
          1. Breakeven: si el trade llega a BREAKEVEN_PCT de ganancia,
             mover SL al precio de entrada (nunca perder en este trade)
          2. Trailing: si el trade llega a TRAIL_ACTIVATION_PCT de ganancia,
             activar el trailing stop que persigue el precio
          3. SL/TP hit: cerrar si el precio cruza alguno de los niveles
        """
        tf = self.settings.primary_timeframe.value

        current_prices = {}
        for symbol in list(self.executor.open_positions.keys()):
            price = self.pipeline.get_live_price(symbol)
            if price and price > 0:
                current_prices[symbol] = price

        if not current_prices:
            return

        # ATR para trailing stop — si BD no tiene suficiente, pedir a Binance
        dfs_for_trailing = {}
        for symbol in current_prices:
            df_raw = await self.pipeline.get_df(symbol, tf, limit=60, with_indicators=False)

            if df_raw.empty or len(df_raw) < 50:
                # BD insuficiente — descargar directamente de Binance
                try:
                    klines = self.client.get_klines(
                        symbol=symbol, interval=tf, limit=100
                    )
                    if klines:
                        import pandas as pd
                        df_api = pd.DataFrame(klines, columns=[
                            "open_time","open","high","low","close","volume",
                            "close_time","quote_volume","trades",
                            "taker_buy_volume","taker_buy_quote","ignore"
                        ])
                        for col in ["open","high","low","close","volume","quote_volume"]:
                            df_api[col] = pd.to_numeric(df_api[col], errors="coerce").fillna(0.0)
                        df_api["open_time"] = pd.to_datetime(df_api["open_time"], unit="ms", utc=True)
                        df_api.set_index("open_time", inplace=True)
                        df_raw = df_api
                except Exception as e:
                    log.debug(f"No se pudo obtener velas de API para {symbol}: {e}")

            if not df_raw.empty and len(df_raw) >= 14:
                df_ind = self.pipeline.ti.add_all(df_raw)
                if not df_ind.empty and "atr_14" in df_ind.columns:
                    dfs_for_trailing[symbol] = df_ind
            # Con menos de 14 velas no hay ATR posible — trailing omitido este ciclo

        # Procesar cada posicion individualmente
        for symbol, pos in list(self.executor.open_positions.items()):
            price = current_prices.get(symbol)
            if not price:
                continue

            pos.update_price_extremes(price)

            # Obtener ATR actual para este simbolo
            atr = 0.0
            if symbol in dfs_for_trailing:
                df = dfs_for_trailing[symbol]
                if not df.empty and "atr_14" in df.columns:
                    atr = float(df["atr_14"].iloc[-1])

            # Distancia que el precio ha recorrido a favor desde la entrada
            if pos.direction == "long":
                price_move = price - pos.entry_price   # positivo si el precio subio
            else:
                price_move = pos.entry_price - price   # positivo si el precio bajo


            # ── 1. BREAKEVEN ───────────────────────────────────────────────────
            # El SL se coloca en la entrada + un buffer de 0.1 * ATR
            # para absorber slippage y no cerrar inmediatamente
            be_atr = self.settings.breakeven_atr
            if be_atr > 0 and atr > 0 and price_move >= be_atr * atr:
                if pos.direction == "long" and pos.current_sl < pos.entry_price:
                    # Buffer = 0.1 ATR por encima de la entrada para absorber slippage
                    be_buffer = atr * 0.1
                    new_sl = pos.entry_price + be_buffer
                    await self.executor._update_sl_order(pos, new_sl)
                    pos.current_sl  = new_sl
                    pos.take_profit = float("inf")
                    await self._save_position_state(pos, be_activated=True)
                    log.info(
                        f"BREAKEVEN {symbol} | "
                        f"SL -> {new_sl:.8f} (entrada {pos.entry_price:.8f} + {be_buffer:.8f} buffer) | "
                        f"TP eliminado — solo trailing stop cerrara la posicion"
                    )
                elif pos.direction == "short" and pos.current_sl > pos.entry_price:
                    be_buffer = atr * 0.1
                    new_sl = pos.entry_price - be_buffer
                    await self.executor._update_sl_order(pos, new_sl)
                    pos.current_sl  = new_sl
                    pos.take_profit = 0.0
                    await self._save_position_state(pos, be_activated=True)
                    log.info(
                        f"BREAKEVEN {symbol} | "
                        f"SL -> {new_sl:.8f} (entrada {pos.entry_price:.8f} - {be_buffer:.8f} buffer) | "
                        f"TP eliminado — solo trailing stop cerrara la posicion"
                    )

            # ── 2. TRAILING STOP ───────────────────────────────────────────────
            # Se activa con TRAIL_ACTIVATION_ATR * ATR de movimiento favorable.
            # Continua activo INCLUSO despues del breakeven — es el unico
            # mecanismo de cierre una vez que el TP fue eliminado.
            trail_atr_activation = self.settings.trail_activation_atr
            trail_ok = (
                trail_atr_activation == 0
                or (atr > 0 and price_move >= trail_atr_activation * atr)
            )

            if trail_ok and atr > 0:
                # Usar multiplicador mas agresivo si el BE ya esta activo
                be_active = (
                    (pos.direction == "long"  and pos.take_profit == float("inf")) or
                    (pos.direction == "short" and pos.take_profit == 0.0)
                )
                trail_mult = (
                    self.settings.trail_atr_after_be
                    if be_active
                    else self.settings.trail_atr_multiplier
                )
                new_sl = self.sl_tp.update_trailing_stop(
                    direction=pos.direction,
                    current_price=price,
                    current_sl=pos.current_sl,
                    entry_price=pos.entry_price,
                    atr=atr,
                    trail_atr_mult=trail_mult,
                )
                if new_sl != pos.current_sl:
                    await self.executor._update_sl_order(pos, new_sl)
                    pos.current_sl = new_sl
                    await self._save_position_state(pos)   # persistir SL en BD

            # ── 3. SL / TP HIT ─────────────────────────────────────────────────
            hit_reason  = None
            close_price = price  # precio por defecto = precio actual del WebSocket

            if pos.direction == "long":
                if price <= pos.current_sl:
                    hit_reason  = "sl_hit"
                    # En dry_run usar el precio del SL como exit (evita slippage negativo)
                    # En testnet/live Binance ejecuta la orden al precio de mercado real
                    close_price = pos.current_sl if self.client.is_dry_run else price
                elif pos.take_profit != float("inf") and price >= pos.take_profit:
                    hit_reason  = "tp_hit"
                    close_price = pos.take_profit if self.client.is_dry_run else price
            else:
                if price >= pos.current_sl:
                    hit_reason  = "sl_hit"
                    close_price = pos.current_sl if self.client.is_dry_run else price
                elif pos.take_profit > 0 and price <= pos.take_profit:
                    hit_reason  = "tp_hit"
                    close_price = pos.take_profit if self.client.is_dry_run else price

            if hit_reason:
                closed = await self.executor.close_position(pos, hit_reason, close_price)
                if closed:
                    pnl = pos.calc_unrealized_pnl(close_price)
                    won = pnl > 0
                    self.risk_mgr.record_trade_closed(won=won, pnl=pnl)
                    await self.portfolio.sync_after_close(
                        pnl=pnl,
                        current_prices=current_prices,
                    )

    async def _save_position_state(
        self, pos, be_activated: bool = False
    ) -> None:
        """
        Persiste el estado actual de una posicion en BD (current_sl, be_activated).
        Permite recuperar el estado exacto tras un reinicio del bot.
        """
        try:
            from sqlalchemy import update as sa_update
            from database.models import Trade
            from decimal import Decimal as _Dec
            async with self.db.async_session() as session:
                be_now = be_activated or (
                    (pos.direction == "long"  and pos.take_profit == float("inf")) or
                    (pos.direction == "short" and pos.take_profit == 0.0)
                )
                tp_val = (
                    None if pos.take_profit in (float("inf"), 0.0)
                    else _Dec(str(round(pos.take_profit, 8)))
                )
                stmt = (
                    sa_update(Trade)
                    .where(Trade.id == pos.trade_id)
                    .values(
                        stop_loss=_Dec(str(round(pos.current_sl, 8))),
                        take_profit=tp_val,
                        trade_metadata=Trade.trade_metadata.op("||")(
                            {"current_sl": pos.current_sl, "be_activated": be_now}
                        ),
                    )
                )
                await session.execute(stmt)
        except Exception as e:
            log.debug(f"Error guardando estado de posicion {pos.symbol}: {e}")

    async def _signal_analysis_loop(self) -> None:
        """
        Loop LENTO — corre cada N minutos (timeframe principal).
        Analiza el mercado y busca nuevas entradas cuando se cierra una vela.
        """
        tf            = self.settings.primary_timeframe.value
        interval_secs = self._tf_to_seconds(tf)

        log.info(f"Analisis de señales cada {interval_secs}s ({tf})")

        while self._running:
            try:
                await self._run_signal_cycle()
                self._cycle_count += 1
                await asyncio.sleep(interval_secs)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error en ciclo de señales: {e}", exc_info=True)
                await asyncio.sleep(30)

    async def _run_signal_cycle(self) -> None:
        """
        Ciclo de analisis completo por vela cerrada:
          1. Sincronizar balance
          2. Evaluar nuevas entradas para simbolos sin posicion
          3. Ejecutar decisiones aprobadas
        """
        tf         = self.settings.primary_timeframe.value
        tf_htf     = self.settings.secondary_timeframe.value
        top_symbols = self.pipeline.get_top_symbol_names()

        if not top_symbols:
            return

        # 1. Sincronizar balance
        current_prices = {
            s: (self.pipeline.get_live_price(s) or 0)
            for s in top_symbols
        }
        await self.portfolio.sync(current_prices)

        # 2. Evaluar nuevas entradas
        available_slots = (
            self.settings.max_open_positions - self.executor.open_count
        )

        if available_slots > 0 and not self.risk_mgr.state.is_paused:
            symbols_to_scan = [
                s for s in top_symbols
                if s not in self.executor.open_positions
            ][:available_slots + 3]

            signals = await self.predictor.predict_batch(
                symbols=symbols_to_scan,
                timeframe=tf,
            )

            for signal in signals[:available_slots]:
                if self.executor.open_count >= self.settings.max_open_positions:
                    break

                df     = await self.pipeline.get_df(signal.symbol, tf,     limit=200)
                df_htf = await self.pipeline.get_df(signal.symbol, tf_htf, limit=100)

                if df.empty:
                    continue

                symbol_info = await self._get_symbol_info(signal.symbol)

                # Agregar la direccion del ranking al symbol_info
                top_symbol = next(
                    (s for s in self.pipeline.get_top_symbols()
                     if s.symbol == signal.symbol), None
                )
                if top_symbol:
                    symbol_info["ranking_direction"] = top_symbol.direction

                decision = self.strategy.evaluate(
                    signal=signal,
                    df=df,
                    df_htf=df_htf if not df_htf.empty else None,
                    current_price=current_prices.get(signal.symbol),
                    symbol_info=symbol_info,
                )

                if decision.is_actionable:
                    await self.executor.open_position(decision)
                    await asyncio.sleep(0.5)

        if self._cycle_count % 10 == 0:
            self._log_status(current_prices)

    async def _portfolio_snapshot_loop(self) -> None:
        """Guarda snapshots del portfolio cada 5 minutos."""
        while self._running:
            try:
                await asyncio.sleep(300)
                current_prices = {
                    s: (self.pipeline.get_live_price(s) or 0)
                    for s in self.executor.open_positions.keys()
                }
                await self.portfolio.save_snapshot(current_prices)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(f"Error en snapshot loop: {e}")

    # ── Utilidades ─────────────────────────────────────────────────────────────

    async def _get_symbol_info(self, symbol: str) -> dict:
        """Obtiene info del simbolo (step_size) desde la BD."""
        try:
            from sqlalchemy import select
            async with self.db.async_session() as session:
                from database.models import Symbol as SymModel
                stmt   = select(SymModel).where(SymModel.symbol == symbol)
                result = await session.execute(stmt)
                sym    = result.scalar_one_or_none()
                if sym and sym.step_size:
                    return {"step_size": sym.step_size, "min_qty": sym.min_qty}
        except Exception:
            pass
        return {}

    def _tf_to_seconds(self, tf: str) -> int:
        mapping = {
            "1m": 60, "3m": 180, "5m": 300,
            "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400,
        }
        return mapping.get(tf, 900)

    def _log_status(self, current_prices: dict) -> None:
        state = self.risk_mgr.state
        upnl  = self.executor.get_total_unrealized_pnl(current_prices)
        log.info(
            f"[CICLO {self._cycle_count}] "
            f"Balance: ${state.total_balance:.2f} | "
            f"uPnL: ${upnl:+.2f} | "
            f"Posiciones: {self.executor.open_count}/{self.settings.max_open_positions} | "
            f"DD: {state.current_drawdown*100:.2f}% | "
            f"{'PAUSADO' if state.is_paused else 'ACTIVO'}"
        )

    async def stop(self) -> None:
        """Detiene el loop limpiamente."""
        self._running = False
        await self.pipeline.stop()
        log.info("TradingLoop detenido")

    def get_status(self) -> dict:
        return {
            "running":       self._running,
            "cycle_count":   self._cycle_count,
            "open_positions": self.executor.open_count,
            "risk":          self.risk_mgr.get_status(),
            "pipeline":      self.pipeline.get_status(),
            "model_version": self.predictor.model_version,
            "retrain":       self.retrainer.get_status(),
        }
