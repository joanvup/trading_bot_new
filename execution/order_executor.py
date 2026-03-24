"""
=============================================================
execution/order_executor.py
Motor de ejecucion de ordenes en Binance Futuros.

Responsabilidades:
  - Abrir posicion (market order + SL + TP)
  - Cancelar ordenes pendientes si cambia el plan
  - Cerrar posicion (market order + cancel SL/TP abiertos)
  - Registro completo en PostgreSQL
  - Soporte identico para testnet, dry_run y live
=============================================================
"""

import asyncio
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from core.binance_client import BinanceFuturesClient
from database.connection import DatabaseManager
from database.models import Trade, Symbol
from strategy.strategy_engine import TradeDecision
from config.settings import Settings
from utils.logger import get_logger

log = get_logger(__name__)


class OpenPosition:
    """Estado de una posicion abierta en memoria."""

    __slots__ = (
        "trade_id", "symbol", "direction",
        "entry_price", "quantity", "leverage",
        "stop_loss", "take_profit",
        "sl_order_id", "tp_order_id", "entry_order_id",
        "opened_at", "current_sl", "original_sl",
        "emergency_sl",
        "highest_price", "lowest_price",
        "signal_confidence",
    )

    def __init__(
        self,
        trade_id:          int,
        symbol:            str,
        direction:         str,
        entry_price:       float,
        quantity:          Decimal,
        leverage:          int,
        stop_loss:         float,
        take_profit:       float,
        signal_confidence: float = 0.0,
    ):
        self.trade_id          = trade_id
        self.symbol            = symbol
        self.direction         = direction
        self.entry_price       = entry_price
        self.quantity          = quantity
        self.leverage          = leverage
        self.stop_loss         = stop_loss
        self.take_profit       = take_profit
        self.sl_order_id:  Optional[str] = None
        self.tp_order_id:  Optional[str] = None
        self.entry_order_id: Optional[str] = None
        self.opened_at         = datetime.now(timezone.utc)
        self.current_sl        = stop_loss
        self.original_sl       = stop_loss   # SL inicial para detectar si trailing se movio
        self.emergency_sl      = 0.0         # SL de emergencia colocado en Binance
        self.highest_price     = entry_price
        self.lowest_price      = entry_price
        self.signal_confidence = signal_confidence

    def update_price_extremes(self, current_price: float) -> None:
        self.highest_price = max(self.highest_price, current_price)
        self.lowest_price  = min(self.lowest_price,  current_price)

    def calc_unrealized_pnl(self, current_price: float) -> float:
        if self.direction == "long":
            raw = (current_price - self.entry_price) * float(self.quantity) * self.leverage
        else:
            raw = (self.entry_price - current_price) * float(self.quantity) * self.leverage
        return raw

    def calc_pnl_pct(self, current_price: float) -> float:
        if self.direction == "long":
            return (current_price - self.entry_price) / self.entry_price
        return (self.entry_price - current_price) / self.entry_price

    def __repr__(self) -> str:
        return (
            f"<Position {self.direction.upper()} {self.symbol} "
            f"x{self.leverage} @ {self.entry_price} qty={self.quantity}>"
        )


class OrderExecutor:
    """
    Ejecuta ordenes en Binance Futuros y gestiona el ciclo de vida
    completo de cada posicion abierta.

    Uso:
        executor = OrderExecutor(settings, client, db)

        # Abrir posicion
        pos = await executor.open_position(decision)

        # Cerrar posicion
        if pos:
            await executor.close_position(pos, reason="sl_hit", current_price=49200)
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

        # Posiciones abiertas en memoria {symbol: OpenPosition}
        self._open_positions: dict[str, OpenPosition] = {}
        self._symbol_id_cache: dict[str, int] = {}

    async def reconcile_positions(self) -> int:
        """
        Reconcilia posiciones abiertas al arrancar el bot.
        - dry_run:  lee trades 'open' de la BD (no hay Binance real)
        - testnet/live: compara Binance con BD y reconstruye
        """

        # ── DRY RUN: recuperar desde BD ───────────────────────────────────────
        if self.client.is_dry_run:
            log.info("Reconciliando posiciones (dry_run) desde BD...")
            recovered = 0
            from sqlalchemy import select as sa_select
            async with self.db.async_session() as session:
                stmt = (
                    sa_select(Trade, Symbol.symbol)
                    .join(Symbol, Trade.symbol_id == Symbol.id)
                    .where(Trade.status == "open", Trade.is_dry_run == True)
                    .order_by(Trade.opened_at.desc())
                )
                result = await session.execute(stmt)
                rows   = result.all()

            if not rows:
                log.info("No hay posiciones dry_run abiertas en BD")
                return 0

            for trade, sym_name in rows:
                if sym_name in self._open_positions:
                    continue

                sl = float(trade.stop_loss)   if trade.stop_loss   else float(trade.entry_price) * 0.97
                tp = float(trade.take_profit) if trade.take_profit else float(trade.entry_price) * 1.04
                meta = trade.trade_metadata or {}

                pos = OpenPosition(
                    trade_id=trade.id,
                    symbol=sym_name,
                    direction=trade.direction,
                    entry_price=float(trade.entry_price),
                    quantity=trade.quantity,
                    leverage=trade.leverage,
                    stop_loss=sl,
                    take_profit=tp,
                    signal_confidence=meta.get("signal_confidence", 0.0),
                )
                # Restaurar estado de BE/TS si estaba activo
                saved_sl = meta.get("current_sl")
                if saved_sl:
                    pos.current_sl = float(saved_sl)
                    pos.original_sl = sl

                # Restaurar si el TP fue eliminado por breakeven
                if meta.get("be_activated"):
                    if pos.direction == "long":
                        pos.take_profit = float("inf")
                    else:
                        pos.take_profit = 0.0

                self._open_positions[sym_name] = pos
                recovered += 1
                log.info(
                    f"Posicion dry_run recuperada: {trade.direction.upper()} {sym_name} | "
                    f"entry={trade.entry_price} | SL={pos.current_sl:.8f} | "
                    f"trade_id={trade.id}"
                )

            log.info(f"Reconciliacion dry_run: {recovered} posicion(es) recuperada(s)")
            return recovered

        # ── TESTNET / LIVE: comparar con Binance ──────────────────────────────
        log.info("Reconciliando posiciones con Binance...")
        try:
            binance_positions = self.client.get_open_positions()
        except Exception as e:
            log.error(f"No se pudieron obtener posiciones de Binance: {e}")
            return 0

        if not binance_positions:
            log.info("No hay posiciones abiertas en Binance")
            return 0

        log.info(f"Binance reporta {len(binance_positions)} posicion(es) abierta(s)")

        from sqlalchemy import select as sa_sel2
        recovered = 0

        for bp in binance_positions:
            symbol = bp.get("symbol", "")
            amt    = float(bp.get("positionAmt", 0))
            if amt == 0 or symbol in self._open_positions:
                continue

            direction   = "long" if amt > 0 else "short"
            entry_price = float(bp.get("entryPrice", 0))
            leverage    = int(float(bp.get("leverage", 1)))
            quantity    = Decimal(str(abs(amt)))

            log.warning(
                f"Posicion sin seguimiento: {direction.upper()} {symbol} | "
                f"qty={abs(amt)} | entry={entry_price:.8f}"
            )

            matching_trade = None
            async with self.db.async_session() as session:
                stmt = (
                    sa_sel2(Trade, Symbol.symbol)
                    .join(Symbol, Trade.symbol_id == Symbol.id)
                    .where(Trade.status == "open", Symbol.symbol == symbol)
                    .order_by(Trade.opened_at.desc())
                    .limit(1)
                )
                result = await session.execute(stmt)
                row    = result.one_or_none()
                if row:
                    matching_trade = row[0]

            if matching_trade:
                sl = float(matching_trade.stop_loss)   if matching_trade.stop_loss   else entry_price * (0.97 if direction == "long" else 1.03)
                tp = float(matching_trade.take_profit) if matching_trade.take_profit else entry_price * (1.04 if direction == "long" else 0.96)
                trade_id = matching_trade.id
                meta     = matching_trade.trade_metadata or {}
            else:
                sl = entry_price * (0.97 if direction == "long" else 1.03)
                tp = entry_price * (1.04 if direction == "long" else 0.96)
                notional = abs(amt) * entry_price
                async with self.db.async_session() as session:
                    from sqlalchemy import select as sa_sel3
                    res    = await session.execute(sa_sel3(Symbol.id).where(Symbol.symbol == symbol))
                    sym_id = res.scalar_one_or_none()
                    if sym_id is None:
                        sym = Symbol(symbol=symbol, base_asset=symbol.replace("USDT",""), quote_asset="USDT", is_active=True)
                        session.add(sym)
                        await session.flush()
                        sym_id = sym.id
                    new_trade = Trade(
                        symbol_id=sym_id, direction=direction, status="open",
                        order_type="market", entry_price=Decimal(str(entry_price)),
                        stop_loss=Decimal(str(round(sl,8))), take_profit=Decimal(str(round(tp,8))),
                        quantity=quantity, leverage=leverage,
                        notional_value=Decimal(str(round(notional,4))),
                        margin_used=Decimal(str(round(notional/leverage,4))),
                        opened_at=datetime.now(timezone.utc), is_dry_run=False,
                        trade_metadata={"recovered": True},
                    )
                    session.add(new_trade)
                    await session.flush()
                    trade_id = new_trade.id
                meta = {}

            pos = OpenPosition(
                trade_id=trade_id, symbol=symbol, direction=direction,
                entry_price=entry_price, quantity=quantity, leverage=leverage,
                stop_loss=sl, take_profit=tp, signal_confidence=0.0,
            )
            saved_sl = meta.get("current_sl")
            if saved_sl:
                pos.current_sl = float(saved_sl)
            if meta.get("be_activated"):
                pos.take_profit = float("inf") if direction == "long" else 0.0

            self._open_positions[symbol] = pos
            recovered += 1
            log.info(f"Posicion recuperada: {direction.upper()} {symbol} | SL={pos.current_sl:.8f}")

        # Marcar trades BD que ya no existen en Binance
        binance_symbols = {bp.get("symbol") for bp in binance_positions if float(bp.get("positionAmt",0)) != 0}
        async with self.db.async_session() as session:
            from sqlalchemy import select as sa_sel4
            stmt = sa_sel4(Trade, Symbol.symbol).join(Symbol, Trade.symbol_id == Symbol.id).where(Trade.status == "open", Trade.is_dry_run == False)
            result = await session.execute(stmt)
            for trade, sym_name in result.all():
                if sym_name not in binance_symbols:
                    log.warning(f"Trade {trade.id} ({sym_name}) cerrado mientras bot estaba offline")
                    trade.status = "closed"
                    trade.close_reason = "closed_while_bot_offline"
                    trade.closed_at = datetime.now(timezone.utc)

        log.info(f"Reconciliacion completada: {recovered} posicion(es) recuperada(s)")
        return recovered

        log.info("Reconciliando posiciones con Binance...")

        # 1. Posiciones reales en Binance
        try:
            binance_positions = self.client.get_open_positions()
        except Exception as e:
            log.error(f"No se pudieron obtener posiciones de Binance: {e}")
            return 0

        if not binance_positions:
            log.info("No hay posiciones abiertas en Binance")
            return 0

        log.info(f"Binance reporta {len(binance_positions)} posicion(es) abierta(s)")

        # 2. Trades abiertos en BD
        from sqlalchemy import select as sa_select
        async with self.db.async_session() as session:
            stmt   = sa_select(Trade).where(Trade.status == "open")
            result = await session.execute(stmt)
            open_trades = {t.id: t for t in result.scalars().all()}

        recovered = 0

        for bp in binance_positions:
            symbol = bp.get("symbol", "")
            amt    = float(bp.get("positionAmt", 0))

            if amt == 0 or symbol in self._open_positions:
                continue

            direction   = "long" if amt > 0 else "short"
            entry_price = float(bp.get("entryPrice", 0))
            leverage    = int(float(bp.get("leverage", 1)))
            quantity    = Decimal(str(abs(amt)))
            mark_price  = float(bp.get("markPrice", entry_price))

            log.warning(
                f"Posicion sin seguimiento encontrada: "
                f"{direction.upper()} {symbol} | "
                f"qty={abs(amt)} | entry={entry_price:.8f} | "
                f"mark={mark_price:.8f}"
            )

            # 3. Buscar en BD si ya existe un trade abierto para este simbolo
            matching_trade = None
            async with self.db.async_session() as session:
                from sqlalchemy import select as sa_select2
                stmt = (
                    sa_select2(Trade, Symbol.symbol)
                    .join(Symbol, Trade.symbol_id == Symbol.id)
                    .where(
                        Trade.status == "open",
                        Symbol.symbol == symbol,
                    )
                    .order_by(Trade.opened_at.desc())
                    .limit(1)
                )
                result = await session.execute(stmt)
                row    = result.one_or_none()
                if row:
                    matching_trade = row[0]

            if matching_trade:
                # Reconstruir desde BD
                sl_price = float(matching_trade.stop_loss) if matching_trade.stop_loss else entry_price * (0.97 if direction == "long" else 1.03)
                tp_price = float(matching_trade.take_profit) if matching_trade.take_profit else entry_price * (1.04 if direction == "long" else 0.96)
                trade_id = matching_trade.id
                log.info(f"Reconstruyendo desde BD: trade_id={trade_id} | SL={sl_price:.8f} | TP={tp_price:.8f}")
            else:
                # No existe en BD — crear registro nuevo
                from decimal import Decimal as Dec
                notional = abs(amt) * entry_price
                async with self.db.async_session() as session:
                    # Obtener o crear symbol
                    from sqlalchemy import select as sa_sel3
                    stmt = sa_sel3(Symbol.id).where(Symbol.symbol == symbol)
                    res  = await session.execute(stmt)
                    sym_id = res.scalar_one_or_none()
                    if sym_id is None:
                        sym = Symbol(symbol=symbol, base_asset=symbol.replace("USDT",""), quote_asset="USDT", is_active=True)
                        session.add(sym)
                        await session.flush()
                        sym_id = sym.id

                    sl_price = entry_price * (0.97 if direction == "long" else 1.03)
                    tp_price = entry_price * (1.04 if direction == "long" else 0.96)

                    new_trade = Trade(
                        symbol_id=sym_id,
                        direction=direction,
                        status="open",
                        order_type="market",
                        entry_price=Dec(str(entry_price)),
                        stop_loss=Dec(str(round(sl_price, 8))),
                        take_profit=Dec(str(round(tp_price, 8))),
                        quantity=quantity,
                        leverage=leverage,
                        notional_value=Dec(str(round(notional, 4))),
                        margin_used=Dec(str(round(notional / leverage, 4))),
                        opened_at=datetime.now(timezone.utc),
                        is_dry_run=False,
                        trade_metadata={"recovered": True, "source": "reconciliation"},
                    )
                    session.add(new_trade)
                    await session.flush()
                    trade_id = new_trade.id
                    log.warning(
                        f"Posicion sin registro en BD recuperada: {symbol} | "
                        f"Nuevo trade_id={trade_id} | "
                        f"SL/TP calculados automaticamente"
                    )

            # 4. Reconstruir OpenPosition en memoria
            pos = OpenPosition(
                trade_id=trade_id,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                quantity=quantity,
                leverage=leverage,
                stop_loss=sl_price,
                take_profit=tp_price,
                signal_confidence=0.0,
            )
            pos.emergency_sl = sl_price  # usar sl como referencia

            self._open_positions[symbol] = pos
            recovered += 1

            log.info(
                f"Posicion recuperada y bajo gestion del bot: "
                f"{direction.upper()} {symbol} | "
                f"SL={sl_price:.8f} | TP={tp_price:.8f}"
            )

        # 5. Detectar trades en BD que ya no existen en Binance (se cerraron mientras el bot estaba caido)
        binance_symbols = {bp.get("symbol") for bp in binance_positions if float(bp.get("positionAmt", 0)) != 0}
        async with self.db.async_session() as session:
            from sqlalchemy import select as sa_sel4
            stmt = (
                sa_sel4(Trade, Symbol.symbol)
                .join(Symbol, Trade.symbol_id == Symbol.id)
                .where(Trade.status == "open")
            )
            result = await session.execute(stmt)
            for trade, sym_name in result.all():
                if sym_name not in binance_symbols:
                    log.warning(
                        f"Trade {trade.id} ({sym_name}) estaba 'open' en BD "
                        f"pero ya no existe en Binance — marcando como cerrado"
                    )
                    trade.status       = "closed"
                    trade.close_reason = "closed_while_bot_offline"
                    trade.closed_at    = datetime.now(timezone.utc)

        if recovered > 0:
            log.info(f"Reconciliacion completada: {recovered} posicion(es) recuperada(s)")
        else:
            log.info("Reconciliacion completada: estado consistente con Binance")

        return recovered

    # ── Apertura de posicion ──────────────────────────────────────────────────

    async def open_position(
        self, decision: TradeDecision
    ) -> Optional[OpenPosition]:
        """
        Abre una posicion en Binance segun la TradeDecision.

        Secuencia:
          1. Configurar leverage
          2. Colocar market order de entrada
          3. Colocar STOP_MARKET para SL
          4. Colocar TAKE_PROFIT_MARKET para TP
          5. Registrar en PostgreSQL

        Args:
            decision: TradeDecision aprobada por el StrategyEngine

        Returns:
            OpenPosition si tuvo exito, None si fallo
        """
        if not decision.is_actionable:
            return None

        symbol    = decision.symbol
        direction = decision.direction
        qty       = decision.quantity
        leverage  = decision.leverage

        # Verificar que no hay posicion ya abierta en este simbolo
        if symbol in self._open_positions:
            log.warning(f"Ya hay posicion abierta en {symbol}, ignorando")
            return None

        # ── Obtener precision del simbolo (tick_size y step_size) ─────────────
        tick_size, step_size = await self._get_symbol_precision(symbol)

        # Redondear cantidad al step_size
        if step_size and step_size > 0:
            qty = self._round_to_step(qty, step_size)

        # Redondear SL y TP al tick_size
        sl_price = decision.stop_loss
        tp_price = decision.take_profit
        if tick_size and tick_size > 0:
            sl_price = self._round_to_tick(sl_price, tick_size)
            tp_price = self._round_to_tick(tp_price, tick_size)

        if qty <= 0:
            log.error(f"Cantidad invalida tras redondear {symbol}: {qty}")
            return None

        log.info(
            f"Abriendo posicion: {direction.upper()} {symbol} | "
            f"qty={qty} | lev={leverage}x | "
            f"SL={sl_price} | TP={tp_price} | "
            f"tick={tick_size} | step={step_size}"
        )

        try:
            # 1. Configurar leverage
            self.client.set_leverage(symbol, leverage)
            await asyncio.sleep(0.1)

            # 2. Orden de entrada (MARKET)
            side_entry = "BUY" if direction == "long" else "SELL"
            entry_order = self.client.place_market_order(
                symbol=symbol,
                side=side_entry,
                quantity=qty,
            )

            await asyncio.sleep(0.3)  # Dar tiempo a Binance para registrar el fill

            # ── Obtener precio REAL de ejecucion desde Binance ────────────────
            # avgPrice en testnet viene en 0 cuando status=NEW, por eso
            # consultamos la posicion directamente para obtener el entryPrice real
            entry_price = 0.0
            try:
                open_pos = self.client.get_open_positions()
                pos_data  = next(
                    (p for p in open_pos if p.get("symbol") == symbol
                     and float(p.get("positionAmt", 0)) != 0),
                    None
                )
                if pos_data:
                    entry_price = float(pos_data.get("entryPrice", 0))
            except Exception:
                pass

            # Fallbacks si no se pudo obtener de la posicion
            if entry_price == 0:
                entry_price = float(entry_order.get("avgPrice") or 0)
            if entry_price == 0:
                entry_price = float(self.client.get_mark_price(symbol))
            if entry_price == 0:
                entry_price = decision.entry_price

            log.info(f"Precio real de entrada {symbol}: {entry_price:.8f}")

            # ── SL/TP internos del bot (los reales, gestionados por el monitor) ─
            # Estos son los que aparecen en el dashboard y se evaluan cada 10s
            internal_sl = float(sl_price)
            internal_tp = float(tp_price)

            # ── SL de EMERGENCIA en Binance (proteccion si el bot cae) ─────────
            # Se coloca MAS LEJOS que el SL real para no interferir con el bot.
            # Solo se activa si el bot pierde conexion o se detiene.
            # El bot cierra las posiciones via market order antes de que llegue aqui.
            EMERGENCY_SL_MULT = 2.0  # 2x la distancia del SL real

            if direction == "long":
                sl_dist     = entry_price - internal_sl
                emergency_sl = entry_price - (sl_dist * EMERGENCY_SL_MULT)
            else:
                sl_dist      = internal_sl - entry_price
                emergency_sl = entry_price + (sl_dist * EMERGENCY_SL_MULT)

            # Redondear emergency SL al tick_size
            if tick_size and tick_size > 0:
                emergency_sl_dec = self._round_to_tick(emergency_sl, tick_size)
            else:
                emergency_sl_dec = Decimal(str(round(emergency_sl, 8)))

            # Validar que el SL de emergencia este en el lado correcto
            current_mark = float(self.client.get_mark_price(symbol)) or entry_price
            if direction == "long" and float(emergency_sl_dec) >= current_mark:
                emergency_sl_dec = self._round_to_tick(current_mark * 0.95, tick_size) if tick_size else Decimal(str(current_mark * 0.95))
            elif direction == "short" and float(emergency_sl_dec) <= current_mark:
                emergency_sl_dec = self._round_to_tick(current_mark * 1.05, tick_size) if tick_size else Decimal(str(current_mark * 1.05))

            # 3. Colocar SOLO el SL de emergencia en Binance (sin TP)
            side_sl  = "SELL" if direction == "long" else "BUY"
            sl_order = self.client.place_stop_loss_order(
                symbol=symbol,
                side=side_sl,
                quantity=qty,
                stop_price=emergency_sl_dec,
            )
            log.info(
                f"SL emergencia colocado {symbol} @ {emergency_sl_dec} "
                f"(SL real del bot: {internal_sl:.8f})"
            )

            # NO se coloca TP en Binance — el bot gestiona el cierre via market order
            tp_order = {"orderId": "", "status": "MANAGED_BY_BOT"}

            # 4. Crear objeto de posicion con precios REALES y SL/TP INTERNOS
            trade_id = await self._save_trade_opened(
                decision=decision,
                entry_price=entry_price,
                entry_order=entry_order,
                sl_order=sl_order,
                tp_order=tp_order,
                internal_sl=internal_sl,
                internal_tp=internal_tp,
            )

            pos = OpenPosition(
                trade_id=trade_id,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,      # precio real de Binance
                quantity=qty,
                leverage=leverage,
                stop_loss=internal_sl,         # SL real del bot
                take_profit=internal_tp,        # TP real del bot
                signal_confidence=decision.signal_confidence,
            )
            pos.sl_order_id      = str(sl_order.get("orderId", ""))
            pos.tp_order_id      = ""  # sin TP en Binance
            pos.entry_order_id   = str(entry_order.get("orderId", ""))
            pos.emergency_sl     = float(emergency_sl_dec)

            self._open_positions[symbol] = pos

            log.info(
                f"Posicion abierta: {direction.upper()} {symbol} | "
                f"entry={entry_price:.8f} | "
                f"SL_bot={internal_sl:.8f} | TP_bot={internal_tp:.8f} | "
                f"SL_emergencia={emergency_sl_dec} | lev={leverage}x"
            )
            return pos

        except Exception as e:
            log.error(f"Error abriendo posicion {symbol}: {e}", exc_info=True)
            try:
                open_pos = self.client.get_open_positions()
                orphan   = next(
                    (p for p in open_pos
                     if p.get("symbol") == symbol
                     and float(p.get("positionAmt", 0)) != 0),
                    None
                )
                if orphan:
                    amt        = float(orphan.get("positionAmt", 0))
                    close_side = "BUY" if amt < 0 else "SELL"
                    log.warning(f"POSICION HUERFANA {symbol} — cerrando")
                    self.client.cancel_all_orders(symbol)
                    self.client.place_market_order(
                        symbol=symbol,
                        side=close_side,
                        quantity=Decimal(str(abs(amt))),
                        reduce_only=True,
                    )
                else:
                    self.client.cancel_all_orders(symbol)
            except Exception as cleanup_err:
                log.error(f"Error limpiando {symbol}: {cleanup_err}")
            return None

    # ── Cierre de posicion ────────────────────────────────────────────────────

    async def close_position(
        self,
        position:      OpenPosition,
        reason:        str,
        current_price: float,
    ) -> bool:
        """
        Cierra una posicion abierta.

        Secuencia:
          1. Cancelar SL y TP pendientes
          2. Colocar market order de cierre (reduce-only)
          3. Calcular PnL real
          4. Actualizar trade en PostgreSQL

        Args:
            position:      Posicion a cerrar
            reason:        Motivo ("sl_hit", "tp_hit", "signal_exit", "manual", "drawdown")
            current_price: Precio actual para calcular PnL

        Returns:
            True si se cerro correctamente
        """
        symbol    = position.symbol
        direction = position.direction

        log.info(
            f"Cerrando posicion: {symbol} | "
            f"razon={reason} | precio={current_price:.4f}"
        )

        try:
            # 1. Cancelar todas las ordenes pendientes (SL y TP)
            self.client.cancel_all_orders(symbol)
            await asyncio.sleep(0.1)

            # 2. Orden de cierre (reduce-only)
            side_close = "SELL" if direction == "long" else "BUY"
            close_order = self.client.place_market_order(
                symbol=symbol,
                side=side_close,
                quantity=position.quantity,
                reduce_only=True,
            )

            # 3. Calcular PnL
            exit_price = float(
                close_order.get("avgPrice") or close_order.get("price") or current_price
            )
            if exit_price == 0:
                exit_price = current_price

            pnl     = position.calc_unrealized_pnl(exit_price)
            pnl_pct = position.calc_pnl_pct(exit_price)

            # Estimacion de fees (0.04% taker * 2 lados)
            notional  = float(position.quantity) * exit_price
            fees_paid = notional * 0.0004 * 2
            net_pnl   = pnl - fees_paid

            duration = int(
                (datetime.now(timezone.utc) - position.opened_at).total_seconds()
            )

            # 4. Actualizar BD
            await self._save_trade_closed(
                trade_id=position.trade_id,
                exit_price=exit_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                fees_paid=fees_paid,
                net_pnl=net_pnl,
                close_reason=reason,
                duration_seconds=duration,
            )

            # 5. Eliminar de posiciones abiertas
            self._open_positions.pop(symbol, None)

            won = net_pnl > 0
            log.info(
                f"Posicion cerrada: {symbol} | "
                f"{'GANADO' if won else 'PERDIDO'} | "
                f"PnL=${net_pnl:+.2f} ({pnl_pct:+.2%}) | "
                f"Razon: {reason} | "
                f"Duracion: {duration}s"
            )
            return True

        except Exception as e:
            log.error(f"Error cerrando posicion {symbol}: {e}", exc_info=True)
            return False

    # ── Gestion de posiciones abiertas ────────────────────────────────────────

    async def monitor_positions(
        self,
        current_prices: dict[str, float],
        sl_tp_calculator=None,
        dfs: Optional[dict] = None,
        trail_atr_mult: float = 1.2,
    ) -> list[tuple[OpenPosition, str]]:
        """
        Monitorea todas las posiciones abiertas y detecta eventos.

        Verifica para cada posicion:
          - Si el precio toco el SL o TP (por si Binance no ejecuto la orden)
          - Actualiza trailing stop si corresponde
          - Retorna lista de (posicion, razon_cierre) para cerrar

        Args:
            current_prices: {symbol: precio_actual}
            sl_tp_calculator: Para actualizar trailing stop
            dfs:            {symbol: DataFrame} para trailing stop ATR

        Returns:
            Lista de (OpenPosition, reason) que deben cerrarse
        """
        to_close = []

        for symbol, pos in list(self._open_positions.items()):
            price = current_prices.get(symbol)
            if price is None or price <= 0:
                continue

            pos.update_price_extremes(price)

            # Verificar SL hit
            if pos.direction == "long" and price <= pos.current_sl:
                to_close.append((pos, "sl_hit"))
                continue
            elif pos.direction == "short" and price >= pos.current_sl:
                to_close.append((pos, "sl_hit"))
                continue

            # Verificar TP hit
            if pos.direction == "long" and price >= pos.take_profit:
                to_close.append((pos, "tp_hit"))
                continue
            elif pos.direction == "short" and price <= pos.take_profit:
                to_close.append((pos, "tp_hit"))
                continue

            # Actualizar trailing stop
            if sl_tp_calculator and dfs and symbol in dfs:
                df = dfs[symbol]
                if not df.empty and "atr_14" in df.columns:
                    atr = float(df["atr_14"].iloc[-1])
                    new_sl = sl_tp_calculator.update_trailing_stop(
                        direction=pos.direction,
                        current_price=price,
                        current_sl=pos.current_sl,
                        entry_price=pos.entry_price,
                        atr=atr,
                        trail_atr_mult=trail_atr_mult,
                    )
                    if new_sl != pos.current_sl:
                        await self._update_sl_order(pos, new_sl)
                        pos.current_sl = new_sl

        return to_close

    async def _update_sl_order(
        self, position: OpenPosition, new_sl: float
    ) -> None:
        """
        Actualiza el SL interno del bot.

        El SL en Binance es solo de emergencia y NO se actualiza con el trailing
        para no generar ruido de ordenes ni costos adicionales.
        El bot gestiona el cierre via market order cuando el SL interno se toca.
        """
        # Solo actualizar la variable en memoria — el monitor hara el cierre
        log.debug(
            f"SL interno actualizado {position.symbol}: "
            f"{position.current_sl:.8f} -> {new_sl:.8f}"
        )

    # ── Persistencia en BD ────────────────────────────────────────────────────

    async def _save_trade_opened(
        self,
        decision:    TradeDecision,
        entry_price: float,
        entry_order: dict,
        sl_order:    dict,
        tp_order:    dict,
        internal_sl: float = 0.0,
        internal_tp: float = 0.0,
    ) -> int:
        """Guarda el trade abierto en PostgreSQL y retorna el ID."""
        async with self.db.async_session() as session:
            from sqlalchemy import select

            if decision.symbol not in self._symbol_id_cache:
                stmt   = select(Symbol.id).where(Symbol.symbol == decision.symbol)
                result = await session.execute(stmt)
                sym_id = result.scalar_one_or_none()
                if sym_id is None:
                    sym = Symbol(
                        symbol=decision.symbol,
                        base_asset=decision.symbol.replace("USDT", ""),
                        quote_asset="USDT",
                        is_active=True,
                    )
                    session.add(sym)
                    await session.flush()
                    sym_id = sym.id
                self._symbol_id_cache[decision.symbol] = sym_id

            sym_id   = self._symbol_id_cache[decision.symbol]
            notional = float(decision.quantity) * entry_price

            # Usar SL/TP internos del bot (los reales) para la BD
            sl_to_save = internal_sl if internal_sl > 0 else decision.stop_loss
            tp_to_save = internal_tp if internal_tp > 0 else decision.take_profit

            trade = Trade(
                symbol_id=sym_id,
                direction=decision.direction,
                status="open",
                order_type="market",
                entry_price=Decimal(str(entry_price)),      # precio real de ejecucion
                stop_loss=Decimal(str(round(sl_to_save, 8))),
                take_profit=Decimal(str(round(tp_to_save, 8))),
                quantity=decision.quantity,
                leverage=decision.leverage,
                notional_value=Decimal(str(round(notional, 4))),
                margin_used=Decimal(str(round(decision.margin_used, 4))),
                opened_at=datetime.now(timezone.utc),
                is_dry_run=self.client.is_dry_run,
                binance_order_id=str(entry_order.get("orderId", "")),
                trade_metadata={
                    "sl_order_id":       str(sl_order.get("orderId", "")),
                    "tp_order_id":       str(tp_order.get("orderId", "")),
                    "signal_confidence": decision.signal_confidence,
                    "filters_passed":    decision.filters_passed,
                    "risk_reward":       decision.risk_reward,
                    "sl_is_emergency":   True,   # el SL en Binance es de emergencia
                    "tp_managed_by_bot": True,   # el TP lo gestiona el bot
                },
            )
            session.add(trade)
            await session.flush()
            trade_id = trade.id

        return trade_id

    async def _save_trade_closed(
        self,
        trade_id:         int,
        exit_price:       float,
        pnl:              float,
        pnl_pct:          float,
        fees_paid:        float,
        net_pnl:          float,
        close_reason:     str,
        duration_seconds: int,
    ) -> None:
        """Actualiza el trade cerrado en PostgreSQL."""
        from sqlalchemy import select

        async with self.db.async_session() as session:
            stmt   = select(Trade).where(Trade.id == trade_id)
            result = await session.execute(stmt)
            trade  = result.scalar_one_or_none()

            if trade is None:
                log.warning(f"Trade {trade_id} no encontrado en BD")
                return

            trade.exit_price       = Decimal(str(round(exit_price, 8)))
            trade.pnl              = Decimal(str(round(pnl, 4)))
            trade.pnl_pct          = Decimal(str(round(pnl_pct, 6)))
            trade.fees_paid        = Decimal(str(round(fees_paid, 8)))
            trade.net_pnl          = Decimal(str(round(net_pnl, 4)))
            trade.status           = "closed"
            trade.close_reason     = close_reason
            trade.closed_at        = datetime.now(timezone.utc)
            trade.duration_seconds = duration_seconds

    # ── Consultas ─────────────────────────────────────────────────────────────

    @property
    def open_positions(self) -> dict[str, OpenPosition]:
        return dict(self._open_positions)

    @property
    def open_count(self) -> int:
        return len(self._open_positions)

    def get_total_unrealized_pnl(self, current_prices: dict[str, float]) -> float:
        total = 0.0
        for symbol, pos in self._open_positions.items():
            price = current_prices.get(symbol, pos.entry_price)
            total += pos.calc_unrealized_pnl(price)
        return total

    # ── Precision de simbolos ─────────────────────────────────────────────────

    async def _get_symbol_precision(
        self, symbol: str
    ) -> tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Retorna (tick_size, step_size) del simbolo.
        Busca primero en BD, luego en Binance si no encuentra.
        """
        try:
            from sqlalchemy import select as sa_select
            async with self.db.async_session() as session:
                stmt   = sa_select(Symbol).where(Symbol.symbol == symbol)
                result = await session.execute(stmt)
                sym    = result.scalar_one_or_none()
                if sym and sym.tick_size and sym.step_size:
                    return sym.tick_size, sym.step_size
        except Exception as e:
            log.warning(f"Error leyendo precision de BD para {symbol}: {e}")

        # Fallback: obtener de Binance directamente
        try:
            info = self.client._client.futures_exchange_info()
            for s in info.get("symbols", []):
                if s["symbol"] != symbol:
                    continue
                tick_size = None
                step_size = None
                for f in s.get("filters", []):
                    if f["filterType"] == "PRICE_FILTER":
                        tick_size = Decimal(f["tickSize"])
                    if f["filterType"] == "LOT_SIZE":
                        step_size = Decimal(f["stepSize"])
                if tick_size and step_size:
                    log.info(f"Precision de {symbol} obtenida de Binance: tick={tick_size} step={step_size}")
                    return tick_size, step_size
        except Exception as e:
            log.warning(f"No se pudo obtener precision de Binance para {symbol}: {e}")

        return None, None

    def _round_to_tick(self, price: float, tick_size: Decimal) -> Decimal:
        """Redondea un precio al tick_size permitido por Binance."""
        tick = float(tick_size)
        if tick <= 0:
            return Decimal(str(price))
        # Calcular decimales del tick_size
        import math
        decimals = max(0, -int(math.floor(math.log10(tick))))
        rounded  = round(round(price / tick) * tick, decimals)
        return Decimal(str(rounded))

    def _round_to_step(self, qty: Decimal, step_size: Decimal) -> Decimal:
        """Redondea una cantidad al step_size permitido por Binance."""
        step = float(step_size)
        if step <= 0:
            return qty
        import math
        decimals = max(0, -int(math.floor(math.log10(step))))
        rounded  = round(round(float(qty) / step) * step, decimals)
        return Decimal(str(rounded))
