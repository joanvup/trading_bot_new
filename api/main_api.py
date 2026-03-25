"""
=============================================================
api/main_api.py
API REST con FastAPI para el dashboard del trading bot.

Endpoints:
  GET  /api/status          Estado general del sistema
  GET  /api/portfolio       Balance, drawdown, metricas
  GET  /api/positions       Posiciones abiertas con PnL live
  GET  /api/trades          Historial de trades con filtros
  GET  /api/signals         Señales recientes del modelo IA
  GET  /api/top-symbols     Ranking actual de activos
  GET  /api/performance     Metricas de rendimiento (Sharpe, win rate)
  GET  /api/model           Info del modelo IA activo
  POST /api/control         Comandos: start, stop, pause, retrain
  WS   /ws                  WebSocket para updates en tiempo real
=============================================================
"""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import select, desc, func, and_

from database.models import Trade, Signal, PortfolioSnapshot, Symbol, ModelTrainingRun
from utils.logger import get_logger

log = get_logger(__name__)

# Directorio del dashboard
DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"

app = FastAPI(
    title="Trading Bot IA — Dashboard API",
    description="API REST para monitorear y controlar el bot de trading",
    version="1.0.0",
)

# CORS: allow_credentials debe ser False cuando allow_origins es ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir el dashboard como archivos estáticos
if DASHBOARD_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR)), name="dashboard")


@app.get("/", include_in_schema=False)
async def root():
    """Redirige al dashboard."""
    index = DASHBOARD_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Dashboard no encontrado. Abre dashboard/index.html manualmente."}

# ── Referencia global al TradingLoop (se inyecta desde main.py) ───────────────
_trading_loop = None
_db_manager   = None


def set_trading_loop(loop, db):
    """Inyecta la referencia al TradingLoop activo."""
    global _trading_loop, _db_manager
    _trading_loop = loop
    _db_manager   = db


# ── Modelos Pydantic ──────────────────────────────────────────────────────────

class ControlCommand(BaseModel):
    action: str   # "pause", "resume", "stop", "retrain"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_db():
    if _db_manager is None:
        raise HTTPException(503, "Base de datos no disponible")
    return _db_manager


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    """Estado general del sistema en tiempo real."""
    from config.settings import get_settings
    settings = get_settings()

    if _trading_loop is None:
        return {
            "running":    False,
            "mode":       settings.trading_mode.value,
            "uptime_secs": 0,
            "message":    "TradingLoop no inicializado",
        }

    status = _trading_loop.get_status()
    return {
        "running":        status["running"],
        "mode":           settings.trading_mode.value,
        "cycle_count":    status["cycle_count"],
        "open_positions": status["open_positions"],
        "model_version":  status["model_version"],
        "is_paused":      status["risk"]["is_paused"],
        "pause_reason":   status["risk"]["pause_reason"],
        "top_symbols":    status["pipeline"]["top_symbols"],
        "ws_messages":    status["pipeline"].get("ws_messages", 0),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/portfolio")
async def get_portfolio():
    """Balance actual, drawdown y metricas del portfolio."""
    db = _get_db()

    # Estado en memoria del RiskManager
    risk_data = {}
    if _trading_loop:
        risk_data = _trading_loop.get_status().get("risk", {})

    # Ultimos 48 snapshots para el grafico (cada 5min = 4h de historia)
    async with db.async_session() as session:
        stmt = (
            select(PortfolioSnapshot)
            .order_by(desc(PortfolioSnapshot.timestamp))
            .limit(96)
        )
        result    = await session.execute(stmt)
        snapshots = result.scalars().all()

    history = [
        {
            "timestamp":    s.timestamp.isoformat(),
            "balance":      float(s.total_balance),
            "drawdown_pct": float(s.current_drawdown) * 100,
            "daily_pnl":    float(s.realized_pnl_day),
            "positions":    s.open_positions_count,
        }
        for s in reversed(snapshots)
    ]

    return {
        "balance":          risk_data.get("balance", 0),
        "available":        risk_data.get("available", 0),
        "peak_balance":     risk_data.get("peak_balance", 0),
        "drawdown_pct":     risk_data.get("drawdown_pct", 0),
        "daily_pnl":        risk_data.get("daily_pnl", 0),
        "portfolio_heat":   risk_data.get("portfolio_heat_pct", 0),
        "consecutive_losses": risk_data.get("consecutive_losses", 0),
        "win_streak":       risk_data.get("win_streak", 0),
        "trades_today":     risk_data.get("trades_today", 0),
        "is_paused":        risk_data.get("is_paused", False),
        "history":          history,
    }


@app.get("/api/positions")
async def get_positions():
    """Posiciones abiertas con PnL en tiempo real."""
    if _trading_loop is None:
        return {"positions": [], "total_unrealized_pnl": 0}

    executor = _trading_loop.executor
    positions_out = []

    for symbol, pos in executor.open_positions.items():
        live_price = _trading_loop.pipeline.get_live_price(symbol) or pos.entry_price
        upnl = pos.calc_unrealized_pnl(live_price)
        upnl_pct = pos.calc_pnl_pct(live_price)

        # Detectar si el trailing stop se ha movido respecto al SL original
        trailing_active  = round(pos.current_sl, 8) != round(pos.original_sl, 8)
        trailing_moved   = 0.0
        if trailing_active:
            trailing_moved = abs(pos.current_sl - pos.original_sl)

        # Distancia actual al SL y TP en % del precio actual
        sl_dist_pct = abs(live_price - pos.current_sl) / live_price * 100 if live_price > 0 else 0
        # Detectar si el BE esta activo (take_profit = inf significa sin TP)
        be_active_tp = (
            pos.take_profit == float("inf") or
            (pos.direction == "short" and pos.take_profit == 0.0)
        )
        # Usar None para el TP cuando BE esta activo — JSON no acepta inf
        tp_for_display = None if be_active_tp else pos.take_profit
        tp_for_display_safe = tp_for_display or 0.0

        tp_dist_pct = (
            abs(live_price - tp_for_display_safe) / live_price * 100
            if live_price > 0 and not be_active_tp else 0
        )

        # ATR — si BD no tiene suficientes velas, obtener directo de Binance
        atr_value   = 0.0
        sl_atr_dist = 0.0
        if _trading_loop:
            df_atr = await _trading_loop.pipeline.get_df(symbol, "15m", limit=60, with_indicators=False)
            if df_atr.empty or len(df_atr) < 14:
                try:
                    import pandas as _pd
                    klines = _trading_loop.client.get_klines(symbol=symbol, interval="15m", limit=100)
                    if klines:
                        df_api = _pd.DataFrame(klines, columns=[
                            "open_time","open","high","low","close","volume",
                            "close_time","quote_volume","trades",
                            "taker_buy_volume","taker_buy_quote","ignore"
                        ])
                        for col in ["open","high","low","close","volume","quote_volume"]:
                            df_api[col] = _pd.to_numeric(df_api[col], errors="coerce").fillna(0.0)
                        df_api["open_time"] = _pd.to_datetime(df_api["open_time"], unit="ms", utc=True)
                        df_api.set_index("open_time", inplace=True)
                        df_atr = df_api
                except Exception:
                    pass
            if not df_atr.empty and len(df_atr) >= 14:
                df_ind = _trading_loop.pipeline.ti.add_all(df_atr)
                if not df_ind.empty and "atr_14" in df_ind.columns:
                    atr_value = float(df_ind["atr_14"].iloc[-1])
                    if atr_value > 0:
                        sl_atr_dist = abs(live_price - pos.current_sl) / atr_value

        # Detectar si el SL ya paso el breakeven (SL >= entry para long, <= para short)
        if pos.direction == "long":
            be_activated  = pos.current_sl >= pos.entry_price * 0.9999
            price_move    = live_price - pos.entry_price
        else:
            be_activated  = pos.current_sl <= pos.entry_price * 1.0001
            price_move    = pos.entry_price - live_price

        # ATR faltante para activar breakeven y trailing
        be_atr_setting    = _trading_loop.settings.breakeven_atr if _trading_loop else 1.0
        trail_atr_setting = _trading_loop.settings.trail_activation_atr if _trading_loop else 0.5
        atr_to_breakeven  = max(0.0, round((be_atr_setting * atr_value - price_move) / atr_value, 2)) if atr_value > 0 else 0
        atr_to_trail      = max(0.0, round((trail_atr_setting * atr_value - price_move) / atr_value, 2)) if atr_value > 0 else 0
        price_move_in_atr = round(price_move / atr_value, 2) if atr_value > 0 else 0

        # ── Campos para la ficha de posicion ──────────────────────────────────
        # R value: distancia recorrida en multiples del riesgo original
        sl_dist_entry = abs(pos.entry_price - pos.original_sl)
        r_value = round(price_move / sl_dist_entry, 2) if sl_dist_entry > 0 else 0

        # Meta BE y Meta TS — precios donde se activan los mecanismos
        if pos.direction == "long":
            meta_be_price = round(pos.entry_price + be_atr_setting * atr_value, 8) if atr_value > 0 else 0
            meta_ts_price = round(pos.entry_price + trail_atr_setting * atr_value, 8) if atr_value > 0 else 0
        else:
            meta_be_price = round(pos.entry_price - be_atr_setting * atr_value, 8) if atr_value > 0 else 0
            meta_ts_price = round(pos.entry_price - trail_atr_setting * atr_value, 8) if atr_value > 0 else 0

        # Barra de progreso: 0%=SL, 100%=TP
        total_range = abs(tp_for_display_safe - pos.current_sl) if not be_active_tp else abs(pos.current_sl * 0.1)

        def _to_bar(price):
            if total_range <= 0 or not price:
                return 0.0
            if pos.direction == "long":
                pct = (price - pos.current_sl) / total_range * 100
            else:
                pct = (pos.current_sl - price) / total_range * 100
            return max(0.0, min(100.0, round(pct, 1)))

        progress_pct  = _to_bar(live_price)
        entry_bar_pct = _to_bar(pos.entry_price)
        meta_be_pct   = _to_bar(meta_be_price) if meta_be_price > 0 else 0
        meta_ts_pct   = _to_bar(meta_ts_price) if meta_ts_price > 0 else 0

        # Funding rate actual del simbolo
        funding_rate = 0.0
        funding_next = ""
        if _trading_loop and not _trading_loop.client.is_dry_run:
            try:
                fr = _trading_loop.client.get_funding_rate(symbol)
                funding_rate = fr["funding_rate"]
                import datetime as _dt
                if fr.get("next_funding_time"):
                    funding_next = _dt.datetime.fromtimestamp(
                        fr["next_funding_time"] / 1000
                    ).strftime("%H:%M UTC")
            except Exception:
                pass

        # Costo de funding estimado por dia (3 pagos de 8h)
        notional_usdt   = float(pos.quantity) * live_price
        funding_cost_8h = abs(funding_rate) * notional_usdt
        funding_cost_day = funding_cost_8h * 3
        # Positivo = te pagan, negativo = pagas
        rate_for_direction = funding_rate if pos.direction == "short" else -funding_rate
        funding_favorable  = rate_for_direction >= 0

        positions_out.append({
            "symbol":        symbol,
            "direction":     pos.direction,
            "entry_price":   pos.entry_price,
            "current_price": live_price,
            "quantity":      float(pos.quantity),
            "leverage":      pos.leverage,

            # SL info
            "stop_loss":          pos.current_sl,
            "stop_loss_original": pos.original_sl,
            "sl_distance_pct":    round(sl_dist_pct, 3),

            # TP info — None cuando BE esta activo (sin limite de ganancia)
            "take_profit":        tp_for_display,   # None si BE activo
            "tp_distance_pct":    round(tp_dist_pct, 3),
            "be_active_no_tp":    be_active_tp,     # indica al dashboard que no hay TP

            # Trailing stop
            "trailing_active":    trailing_active,
            "trailing_moved":     round(trailing_moved, 8),
            "trailing_moved_pct": round(trailing_moved / pos.original_sl * 100, 3) if pos.original_sl > 0 else 0,

            # ATR info
            "atr_value":           round(atr_value, 8),
            "sl_atr_distance":     round(sl_atr_dist, 2),
            "breakeven_activated": be_activated,
            "price_move_atr":      price_move_in_atr,
            "atr_to_breakeven":    atr_to_breakeven,
            "atr_to_trail":        atr_to_trail,

            # Ficha visual
            "r_value":       r_value,
            "meta_be_price": meta_be_price,
            "meta_ts_price": meta_ts_price,
            "meta_be_atr":       be_atr_setting,
            "meta_ts_atr":       trail_atr_setting,
            "trail_atr_after_be": _trading_loop.settings.trail_atr_after_be if _trading_loop else 0.8,
            "progress_pct":  progress_pct,
            "entry_bar_pct": entry_bar_pct,
            "meta_be_pct":   meta_be_pct,
            "meta_ts_pct":   meta_ts_pct,

            # Funding rate
            "funding_rate":        round(funding_rate * 100, 4),   # en %
            "funding_rate_8h":     round(funding_cost_8h, 4),      # costo en USDT cada 8h
            "funding_cost_day":    round(funding_cost_day, 4),      # costo estimado diario
            "funding_favorable":   funding_favorable,               # True si te pagan
            "funding_next":        funding_next,                    # hora del proximo cobro

            # PnL
            "unrealized_pnl":     round(upnl, 2),
            "unrealized_pnl_pct": round(upnl_pct * 100, 2),

            # Metadata
            "opened_at":      pos.opened_at.isoformat(),
            "highest_price":  pos.highest_price,
            "lowest_price":   pos.lowest_price,
            "confidence":     pos.signal_confidence,
        })

    current_prices = {
        s: (_trading_loop.pipeline.get_live_price(s) or 0)
        for s in executor.open_positions
    }
    total_upnl = executor.get_total_unrealized_pnl(current_prices)

    return {
        "positions":            positions_out,
        "total_unrealized_pnl": round(total_upnl, 2),
        "count":                len(positions_out),
    }


@app.get("/api/trades")
async def get_trades(
    limit:     int = 50,
    offset:    int = 0,
    symbol:    Optional[str] = None,
    direction: Optional[str] = None,
    days:      int = 30,
):
    """Historial de trades cerrados con filtros."""
    db = _get_db()

    since = datetime.now(timezone.utc) - timedelta(days=days)

    async with db.async_session() as session:
        # Base query
        stmt = (
            select(Trade, Symbol.symbol)
            .join(Symbol, Trade.symbol_id == Symbol.id)
            .where(
                and_(
                    Trade.status == "closed",
                    Trade.opened_at >= since,
                )
            )
        )

        if symbol:
            stmt = stmt.where(Symbol.symbol == symbol)
        if direction:
            stmt = stmt.where(Trade.direction == direction)

        stmt = stmt.order_by(desc(Trade.opened_at)).limit(limit).offset(offset)
        result = await session.execute(stmt)
        rows   = result.all()

        # Estadisticas globales del periodo
        stats_stmt = (
            select(
                func.count(Trade.id).label("total"),
                func.sum(Trade.net_pnl).label("total_pnl"),
                func.avg(Trade.net_pnl).label("avg_pnl"),
            )
            .where(
                and_(Trade.status == "closed", Trade.opened_at >= since)
            )
        )
        stats_result = await session.execute(stats_stmt)
        stats_row    = stats_result.one()

        wins_stmt = (
            select(func.count(Trade.id))
            .where(
                and_(
                    Trade.status == "closed",
                    Trade.opened_at >= since,
                    Trade.net_pnl > 0,
                )
            )
        )
        wins_result = await session.execute(wins_stmt)
        wins        = wins_result.scalar() or 0

    trades_out = []
    for trade, sym_name in rows:
        trades_out.append({
            "id":           trade.id,
            "symbol":       sym_name,
            "direction":    trade.direction,
            "entry_price":  float(trade.entry_price),
            "exit_price":   float(trade.exit_price) if trade.exit_price else None,
            "quantity":     float(trade.quantity),
            "leverage":     trade.leverage,
            "pnl":          float(trade.pnl)     if trade.pnl     else None,
            "net_pnl":      float(trade.net_pnl) if trade.net_pnl else None,
            "pnl_pct":      float(trade.pnl_pct) * 100 if trade.pnl_pct else None,
            "fees_paid":    float(trade.fees_paid),
            "close_reason": trade.close_reason,
            "duration_secs": trade.duration_seconds,
            "opened_at":    trade.opened_at.isoformat(),
            "closed_at":    trade.closed_at.isoformat() if trade.closed_at else None,
            "is_dry_run":   trade.is_dry_run,
        })

    total     = stats_row.total or 0
    total_pnl = float(stats_row.total_pnl or 0)
    win_rate  = (wins / total * 100) if total > 0 else 0

    return {
        "trades":    trades_out,
        "stats": {
            "total_trades": total,
            "wins":         wins,
            "losses":       total - wins,
            "win_rate_pct": round(win_rate, 1),
            "total_pnl":    round(total_pnl, 2),
            "avg_pnl":      round(float(stats_row.avg_pnl or 0), 2),
        },
        "pagination": {"limit": limit, "offset": offset},
    }


@app.get("/api/signals")
async def get_signals(limit: int = 30):
    """Señales recientes del modelo IA."""
    db = _get_db()

    async with db.async_session() as session:
        stmt = (
            select(Signal, Symbol.symbol)
            .join(Symbol, Signal.symbol_id == Symbol.id)
            .where(Signal.signal_type != "hold")
            .order_by(desc(Signal.created_at))
            .limit(limit)
        )
        result = await session.execute(stmt)
        rows   = result.all()

    return {
        "signals": [
            {
                "id":            s.id,
                "symbol":        sym_name,
                "signal_type":   s.signal_type,
                "confidence":    float(s.confidence),
                "price":         float(s.price_at_signal),
                "timeframe":     s.timeframe,
                "indicators":    s.indicators or {},
                "model_version": s.model_version,
                "created_at":    s.created_at.isoformat(),
                "was_profitable": s.was_profitable,
            }
            for s, sym_name in rows
        ]
    }


@app.get("/api/top-symbols")
async def get_top_symbols():
    """Ranking actual de activos con scores."""
    if _trading_loop is None:
        return {"symbols": []}

    top = _trading_loop.pipeline.get_top_symbols()
    return {
        "symbols": [s.to_dict() for s in top],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/performance")
async def get_performance(days: int = 30):
    """Metricas de rendimiento del periodo."""
    db = _get_db()
    since = datetime.now(timezone.utc) - timedelta(days=days)

    async with db.async_session() as session:
        # Todos los trades cerrados del periodo
        stmt = (
            select(Trade)
            .where(and_(Trade.status == "closed", Trade.opened_at >= since))
            .order_by(Trade.opened_at)
        )
        result = await session.execute(stmt)
        trades = result.scalars().all()

        # Snapshots para curva de equity
        snaps_stmt = (
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.timestamp >= since)
            .order_by(PortfolioSnapshot.timestamp)
            .limit(288)  # max 1 dia de snapshots cada 5min
        )
        snaps_result = await session.execute(snaps_stmt)
        snapshots    = snaps_result.scalars().all()

    if not trades:
        return {"message": "Sin trades en el periodo", "days": days}

    pnls     = [float(t.net_pnl or 0) for t in trades]
    wins     = [p for p in pnls if p > 0]
    losses   = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) if pnls else 0

    avg_win  = sum(wins)   / len(wins)   if wins   else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0

    # Sharpe ratio aproximado (diario)
    import numpy as np
    if len(pnls) > 1:
        arr     = np.array(pnls)
        sharpe  = (arr.mean() / (arr.std() + 1e-10)) * (252 ** 0.5)
    else:
        sharpe = 0

    equity_curve = [
        {"timestamp": s.timestamp.isoformat(), "balance": float(s.total_balance)}
        for s in snapshots
    ]

    return {
        "period_days":    days,
        "total_trades":   len(trades),
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate_pct":   round(win_rate * 100, 1),
        "total_pnl":      round(sum(pnls), 2),
        "avg_win":        round(avg_win, 2),
        "avg_loss":       round(avg_loss, 2),
        "profit_factor":  round(profit_factor, 2),
        "sharpe_ratio":   round(float(sharpe), 2),
        "max_drawdown_pct": round(
            max((float(s.current_drawdown) for s in snapshots), default=0) * 100, 2
        ),
        "equity_curve":   equity_curve,
    }


@app.get("/api/model")
async def get_model_info():
    """Informacion del modelo IA activo."""
    db = _get_db()

    model_data = {}
    if _trading_loop and _trading_loop.predictor.is_ready:
        m = _trading_loop.predictor.model
        model_data = {
            "version":       m.version,
            "is_trained":    m.is_trained,
            "test_accuracy": m.metrics.get("test_accuracy"),
            "f1_weighted":   m.metrics.get("f1_weighted"),
            "f1_buy":        m.metrics.get("f1_buy"),
            "f1_sell":       m.metrics.get("f1_sell"),
            "n_features":    len(m.feature_names),
            "top_features":  dict(list((m.metrics.get("top_features") or {}).items())[:10]),
        }

    async with db.async_session() as session:
        stmt = (
            select(ModelTrainingRun)
            .order_by(desc(ModelTrainingRun.trained_at))
            .limit(5)
        )
        result = await session.execute(stmt)
        runs   = result.scalars().all()

    history = [
        {
            "version":        r.version,
            "trained_at":     r.trained_at.isoformat(),
            "test_accuracy":  float(r.test_accuracy) if r.test_accuracy else None,
            "f1_score":       float(r.f1_score)       if r.f1_score      else None,
            "train_samples":  r.training_samples,
            "is_deployed":    r.is_deployed,
            "duration_secs":  r.duration_seconds,
        }
        for r in runs
    ]

    return {
        "active_model": model_data,
        "training_history": history,
        "retrain_status": _trading_loop.retrainer.get_status() if _trading_loop else {},
    }


@app.post("/api/control")
async def control(cmd: ControlCommand):
    """Comandos de control del bot."""
    if _trading_loop is None:
        raise HTTPException(503, "TradingLoop no disponible")

    action = cmd.action.lower()

    if action == "pause":
        _trading_loop.risk_mgr._pause("Pausa manual desde dashboard")
        return {"ok": True, "message": "Bot pausado"}

    elif action == "resume":
        _trading_loop.risk_mgr._resume()
        return {"ok": True, "message": "Bot reanudado"}

    elif action == "retrain":
        symbols = _trading_loop.pipeline.get_top_symbol_names(10)
        asyncio.create_task(
            _trading_loop.retrainer.retrain_now(symbols)
        )
        return {"ok": True, "message": f"Reentrenamiento iniciado con {len(symbols)} simbolos"}

    else:
        raise HTTPException(400, f"Accion desconocida: {action}")


# ── WebSocket para updates en tiempo real ─────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active = [w for w in self.active if w != ws]

    async def broadcast(self, data: dict):
        import json
        msg = json.dumps(data)
        for ws in list(self.active):
            try:
                await ws.send_text(msg)
            except Exception:
                self.disconnect(ws)


ws_manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket para actualizaciones en tiempo real cada 2 segundos."""
    await ws_manager.connect(websocket)
    try:
        while True:
            if _trading_loop:
                status  = _trading_loop.get_status()
                risk    = status["risk"]
                ex      = _trading_loop.executor

                current_prices = {
                    s: (_trading_loop.pipeline.get_live_price(s) or 0)
                    for s in ex.open_positions
                }

                positions = [
                    {
                        "symbol":    sym,
                        "direction": pos.direction,
                        "entry_price": pos.entry_price,
                        "current_price": current_prices.get(sym, pos.entry_price),
                        "pnl": round(pos.calc_unrealized_pnl(
                            current_prices.get(sym, pos.entry_price)
                        ), 2),
                    }
                    for sym, pos in ex.open_positions.items()
                ]

                await websocket.send_json({
                    "type":          "update",
                    "balance":       risk["balance"],
                    "drawdown_pct":  risk["drawdown_pct"],
                    "positions":     positions,
                    "is_paused":     risk["is_paused"],
                    "cycle_count":   status["cycle_count"],
                    "timestamp":     datetime.now(timezone.utc).isoformat(),
                })

            await asyncio.sleep(2)

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}
