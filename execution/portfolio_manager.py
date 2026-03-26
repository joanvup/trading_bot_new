"""
=============================================================
execution/portfolio_manager.py
Gestor del estado del portfolio y snapshots periodicos.

Responsabilidades:
  - Sincronizar balance real con Binance
  - Guardar snapshots periodicos en PostgreSQL
  - Calcular metricas de rendimiento (Sharpe, win rate, etc.)
  - Alimentar al RiskManager con datos actualizados
=============================================================
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from core.binance_client import BinanceFuturesClient
from database.connection import DatabaseManager
from database.models import PortfolioSnapshot
from execution.order_executor import OrderExecutor
from risk.risk_manager import RiskManager
from config.settings import Settings
from utils.logger import get_logger

log = get_logger(__name__)


class PortfolioManager:
    """
    Mantiene el estado del portfolio sincronizado con Binance
    y registra snapshots periodicos para analisis de rendimiento.

    Uso:
        pm = PortfolioManager(settings, client, db, executor, risk_mgr)
        await pm.sync()   # Sincronizar balance actual
        await pm.save_snapshot()  # Guardar snapshot
    """

    def __init__(
        self,
        settings:  Settings,
        client:    BinanceFuturesClient,
        db:        DatabaseManager,
        executor:  OrderExecutor,
        risk_mgr:  RiskManager,
    ):
        self.settings  = settings
        self.client    = client
        self.db        = db
        self.executor  = executor
        self.risk_mgr  = risk_mgr

        self._last_balance: float = settings.initial_capital
        self._peak_balance: float = settings.initial_capital
        self._last_sync:    Optional[datetime] = None

    async def sync(self, current_prices: Optional[dict] = None) -> dict:
        """
        Sincroniza el balance y actualiza el RiskManager.

        REGLAS:
        - dry_run:     balance = INITIAL_CAPITAL + suma de PnL de trades CERRADOS
                       drawdown se calcula sobre este balance realizado
        - testnet/live: balance viene de Binance (wallet balance, sin unrealized)
                       drawdown se calcula sobre wallet balance

        El PnL no realizado (posiciones abiertas) NO afecta el balance ni el drawdown.
        Solo es informativo para el dashboard.
        """
        try:
            unrealized_pnl = 0.0

            if self.client.is_dry_run:
                # ── Balance real en dry_run = capital inicial + PnL realizado ──
                realized_pnl = await self._get_realized_pnl()
                total_balance     = self.settings.initial_capital + realized_pnl
                available_balance = total_balance  # simplificacion para dry_run

                # PnL no realizado es solo informativo
                if current_prices:
                    unrealized_pnl = self.executor.get_total_unrealized_pnl(current_prices)

            else:
                # ── Testnet/Live: obtener wallet balance de Binance ────────────
                raw = self.client.get_account_balance()
                # totalWalletBalance = balance real sin incluir unrealized PnL
                total_balance     = float(raw.get("totalWalletBalance", self._last_balance))
                available_balance = float(raw.get("availableBalance", total_balance))
                unrealized_pnl    = float(raw.get("totalUnrealizedProfit", 0))

                # Sincronizar entry_price real desde Binance
                try:
                    binance_positions = self.client.get_open_positions()
                    for bp in binance_positions:
                        sym   = bp.get("symbol", "")
                        entry = float(bp.get("entryPrice", 0))
                        if sym in self.executor.open_positions and entry > 0:
                            pos = self.executor.open_positions[sym]
                            if abs(pos.entry_price - entry) > entry * 0.001:
                                pos.entry_price = entry
                except Exception:
                    pass

            self._last_balance = total_balance
            self._last_sync    = datetime.now(timezone.utc)

            # Actualizar RiskManager con balance REALIZADO (sin unrealized)
            self.risk_mgr.update_balance(
                total_balance=total_balance,
                available_balance=available_balance,
                unrealized_pnl=unrealized_pnl,
            )

            heat = self._calc_portfolio_heat(total_balance, current_prices or {})
            self.risk_mgr.update_open_positions(
                count=self.executor.open_count,
                heat=heat,
            )

            return {
                "total_balance":     round(total_balance, 2),
                "available_balance": round(available_balance, 2),
                "unrealized_pnl":    round(unrealized_pnl, 2),
                "open_positions":    self.executor.open_count,
                "portfolio_heat":    round(heat * 100, 2),
                "drawdown_pct":      round(self.risk_mgr.state.current_drawdown * 100, 2),
            }

        except Exception as e:
            log.warning(f"Error sincronizando balance: {e}")
            return {}

    async def sync_after_close(self, pnl: float, current_prices: Optional[dict] = None) -> None:
        """
        Llamar DESPUES de cerrar un trade para actualizar balance y drawdown.
        Este es el unico momento donde el drawdown debe recalcularse.
        """
        if self.client.is_dry_run:
            realized_pnl  = await self._get_realized_pnl()
            total_balance = self.settings.initial_capital + realized_pnl
        else:
            raw = self.client.get_account_balance()
            total_balance = float(raw.get("totalWalletBalance", self._last_balance))

        self._last_balance = total_balance
        available_balance  = total_balance

        # Actualizar peak y drawdown SOLO con balance realizado post-cierre
        self.risk_mgr.state.total_balance     = total_balance
        self.risk_mgr.state.available_balance = available_balance
        self.risk_mgr.state.update_peak()
        self.risk_mgr.state.calc_drawdown()
        self.risk_mgr._check_pause_conditions()

        log.info(
            f"Balance actualizado post-cierre: ${total_balance:.2f} | "
            f"PnL trade: ${pnl:+.2f} | "
            f"DD: {self.risk_mgr.state.current_drawdown*100:.2f}%"
        )

    async def _get_realized_pnl(self) -> float:
        """Suma el net_pnl de todos los trades cerrados en la BD."""
        try:
            from database.models import Trade
            from sqlalchemy import select, func
            async with self.db.async_session() as session:
                stmt = select(func.coalesce(func.sum(Trade.net_pnl), 0)).where(
                    Trade.status == "closed",
                    Trade.is_dry_run == True,
                )
                result = await session.execute(stmt)
                total  = result.scalar() or 0
                return float(total)
        except Exception as e:
            log.warning(f"Error obteniendo PnL realizado: {e}")
            return 0.0

    def _calc_portfolio_heat(
        self, total_balance: float, current_prices: dict
    ) -> float:
        """
        Calcula el % del capital actualmente en riesgo
        considerando todas las posiciones abiertas y sus SL.
        Siempre intenta obtener el precio live del WebSocket
        independientemente de si el simbolo esta en el top ranking.
        """
        from data.websocket_manager import stream_buffer

        total_risk = 0.0
        for symbol, pos in self.executor.open_positions.items():
            # 1. Precio del dict pasado (top symbols)
            price = current_prices.get(symbol, 0)
            # 2. Si es 0 o no esta, buscar en el stream buffer
            if not price:
                price = stream_buffer.get_last_price(symbol) or 0
            # 3. Si sigue sin precio, usar el de entrada
            if not price:
                price = pos.entry_price

            if price <= 0:
                continue

            sl_dist = abs(price - pos.current_sl) / price
            risk    = float(pos.quantity) * price * sl_dist
            total_risk += risk

        if total_balance <= 0:
            return 0.0
        return min(total_risk / total_balance, 1.0)

    async def save_snapshot(
        self,
        current_prices: Optional[dict] = None,
    ) -> None:
        """Guarda un snapshot del portfolio en PostgreSQL."""
        try:
            state = self.risk_mgr.state

            # Calcular win rate del dia
            recent = state.recent_results
            win_rate = (sum(recent) / len(recent)) if recent else None

            async with self.db.async_session() as session:
                snap = PortfolioSnapshot(
                    timestamp=datetime.now(timezone.utc),
                    total_balance=Decimal(str(round(state.total_balance, 4))),
                    available_balance=Decimal(str(round(state.available_balance, 4))),
                    unrealized_pnl=Decimal(str(round(
                        self.executor.get_total_unrealized_pnl(current_prices or {}), 4
                    ))),
                    realized_pnl_day=Decimal(str(round(state.daily_pnl, 4))),
                    open_positions_count=self.executor.open_count,
                    peak_balance=Decimal(str(round(state.peak_balance, 4))),
                    current_drawdown=Decimal(str(round(state.current_drawdown, 6))),
                    total_trades_day=state.total_trades_today,
                    win_rate_day=Decimal(str(round(win_rate, 4))) if win_rate else None,
                )
                session.add(snap)

            log.debug(
                f"Snapshot guardado | "
                f"Balance: ${state.total_balance:.2f} | "
                f"DD: {state.current_drawdown*100:.2f}%"
            )

        except Exception as e:
            log.warning(f"Error guardando snapshot: {e}")

    async def run_snapshot_loop(self, interval_secs: int = 300) -> None:
        """Guarda snapshots periodicamente (cada 5 min por defecto)."""
        while True:
            try:
                await asyncio.sleep(interval_secs)
                await self.save_snapshot()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(f"Error en snapshot loop: {e}")
