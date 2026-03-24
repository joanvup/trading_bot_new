"""
=============================================================
backtest.py
Evaluacion de efectividad del modelo IA con datos reales.

Descarga datos historicos de Binance, aplica el modelo y
los filtros de estrategia, simula las operaciones y genera
un reporte completo de rendimiento.

Uso:
    # Test rapido (5 simbolos, 30 dias, sin filtros)
    python backtest.py

    # Test completo con filtros de estrategia
    python backtest.py --symbols BTCUSDT ETHUSDT SOLUSDT --days 90 --strategy

    # Comparar con/sin filtros
    python backtest.py --days 60 --compare

    # Probar distintos umbrales de confianza
    python backtest.py --sweep-confidence

Metricas reportadas:
    - Accuracy del modelo (BUY/SELL/HOLD correcto)
    - Win rate de los trades
    - Profit Factor
    - Sharpe Ratio
    - Max Drawdown
    - Distribucion de señales
    - Mejor/peor simbolo
    - Impacto de cada filtro de estrategia
=============================================================
"""

import sys
import os
import time
import warnings
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Optional

import click
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import box

warnings.filterwarnings("ignore")

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("TRADING_MODE", "dry_run")
os.environ.setdefault("DB_PASSWORD",  "postgres")

from config.settings  import get_settings
from core.binance_client import BinanceFuturesClient
from analysis.indicators import TechnicalIndicators
from ai.feature_engineer import FeatureEngineer
from ai.ml_model         import TradingMLModel
from risk.sl_tp_calculator import SLTPCalculator

console = Console()


# ── Descarga de datos ──────────────────────────────────────────────────────────

def fetch_klines(client: BinanceFuturesClient, symbol: str,
                 interval: str, days: int) -> pd.DataFrame:
    """Descarga velas historicas de Binance y retorna DataFrame OHLCV."""
    start_ms = int((time.time() - days * 86400) * 1000)
    all_klines = []
    limit = 1500

    while True:
        klines = client.get_klines(
            symbol=symbol, interval=interval,
            limit=limit, start_time=start_ms,
        )
        if not klines:
            break
        all_klines.extend(klines)
        last_ts = klines[-1][6]   # close_time
        if len(klines) < limit:
            break
        start_ms = last_ts + 1
        time.sleep(0.05)

    if not all_klines:
        return pd.DataFrame()

    df = pd.DataFrame(all_klines, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades",
        "taker_buy_volume","taker_buy_quote","ignore"
    ])
    # Convertir TODAS las columnas numericas a float
    for col in ["open","high","low","close","volume",
                "quote_volume","taker_buy_volume","taker_buy_quote"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").fillna(0).astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    return df


# ── Simulacion de un trade ─────────────────────────────────────────────────────

def simulate_trade(
    df:          pd.DataFrame,
    entry_idx:   int,
    direction:   str,
    entry_price: float,
    sl:          float,
    tp:          float,
    max_candles: int = 40,
) -> dict:
    """
    Simula el resultado de un trade a partir de una señal.
    Evalua vela a vela si se toco el SL o el TP.

    Returns:
        dict con outcome, pnl_pct, candles_held, exit_price, exit_reason
    """
    sl_dist = abs(entry_price - sl)

    for i in range(1, max_candles + 1):
        if entry_idx + i >= len(df):
            break

        candle = df.iloc[entry_idx + i]
        high   = candle["high"]
        low    = candle["low"]

        if direction == "long":
            if low  <= sl:  return _trade_result("sl", sl,          entry_price, direction, sl_dist, i)
            if high >= tp:  return _trade_result("tp", tp,          entry_price, direction, sl_dist, i)
        else:
            if high >= sl:  return _trade_result("sl", sl,          entry_price, direction, sl_dist, i)
            if low  <= tp:  return _trade_result("tp", tp,          entry_price, direction, sl_dist, i)

    # Timeout — cerrar al precio del cierre de la ultima vela evaluada
    exit_candle = min(entry_idx + max_candles, len(df) - 1)
    exit_price  = df.iloc[exit_candle]["close"]
    return _trade_result("timeout", exit_price, entry_price, direction, sl_dist, max_candles)


def _trade_result(reason, exit_price, entry_price, direction, sl_dist, candles):
    if direction == "long":
        pnl_pct = (exit_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - exit_price) / entry_price
    won = pnl_pct > 0
    r   = pnl_pct * entry_price / sl_dist if sl_dist > 0 else 0
    return {
        "won":         won,
        "pnl_pct":     pnl_pct,
        "r_value":     r,
        "exit_price":  exit_price,
        "exit_reason": reason,
        "candles":     candles,
    }


# ── Filtros de estrategia ──────────────────────────────────────────────────────

def apply_strategy_filters(row: pd.Series, direction: str, settings) -> tuple[bool, list]:
    """
    Aplica los mismos filtros que StrategyEngine.evaluate().
    Retorna (pasa, lista_de_filtros_fallados).
    """
    failed = []

    # RSI extremo
    if "rsi_14" in row.index and not pd.isna(row["rsi_14"]):
        rsi = float(row["rsi_14"])
        if direction == "long"  and rsi > 75: failed.append(f"RSI sobrecomprado ({rsi:.1f})")
        if direction == "short" and rsi < 25: failed.append(f"RSI sobrevendido ({rsi:.1f})")

    # Volumen bajo
    if "vol_ratio" in row.index and not pd.isna(row["vol_ratio"]):
        if float(row["vol_ratio"]) < 0.8: failed.append(f"Vol bajo ({row['vol_ratio']:.2f}x)")

    # Bollinger Bands
    if "bb_position" in row.index and not pd.isna(row["bb_position"]):
        bb = float(row["bb_position"])
        if direction == "long"  and bb > 0.85: failed.append(f"BB alto ({bb:.2f})")
        if direction == "short" and bb < 0.15: failed.append(f"BB bajo ({bb:.2f})")

    return len(failed) == 0, failed


# ── Motor de backtest ──────────────────────────────────────────────────────────

def run_backtest(
    symbols:          list[str],
    days:             int,
    timeframe:        str,
    min_confidence:   float,
    use_filters:      bool,
    initial_capital:  float,
    risk_per_trade:   float,
    leverage:         int,
    label:            str = "",
) -> dict:
    """
    Corre el backtest completo y retorna las metricas.
    """
    settings   = get_settings()
    client     = BinanceFuturesClient(settings)
    client.initialize()

    models_dir = settings.get_models_path()

    ti      = TechnicalIndicators()
    fe      = FeatureEngineer(models_dir)
    model   = TradingMLModel(models_dir)
    model.load()

    sl_calc = SLTPCalculator(
        atr_sl_multiplier = settings.sl_atr_multiplier,
        atr_tp_multiplier = settings.tp_atr_multiplier,
        min_risk_reward   = settings.min_risk_reward,
    )

    if not model.is_trained:
        console.print("[red]✗ No hay modelo entrenado. Ejecuta --init primero.[/red]")
        sys.exit(1)

    all_trades    = []
    signal_counts = defaultdict(int)
    filter_rejections = defaultdict(int)
    symbol_stats  = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Backtesting {len(symbols)} simbolos...", total=len(symbols))

        for symbol in symbols:
            progress.update(task, description=f"[cyan]{symbol}[/cyan]")

            # 1. Descargar datos
            df_raw = fetch_klines(client, symbol, timeframe, days + 10)
            if df_raw.empty or len(df_raw) < 100:
                progress.advance(task)
                continue

            # 2. Calcular indicadores tecnicos
            df = ti.add_all(df_raw.copy())
            if df.empty or len(df) < 100:
                progress.advance(task)
                continue

            # 3. Preparar features para el modelo (inferencia completa sobre todo el df)
            # prepare_inference devuelve un array 2D con todas las filas posibles
            try:
                # Preparar features fila a fila no es eficiente —
                # mejor preparar todo el df y luego iterar por indices
                X_all = []
                valid_indices = []
                for idx in range(50, len(df) - 45):
                    row_df = df.iloc[max(0, idx-50):idx+1]
                    x = fe.prepare_inference(row_df)
                    if x is not None and len(x) > 0:
                        X_all.append(x[0])
                        valid_indices.append(idx)

                if not X_all:
                    progress.advance(task)
                    continue

                X = np.array(X_all)
            except Exception as e:
                progress.advance(task)
                continue

            # 4. Predicciones del modelo (batch sobre todo X)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    probas = model.model.predict_proba(X)
                    labels = model.model.predict(X)
            except Exception:
                progress.advance(task)
                continue

            # Mapear clases del modelo a indices
            classes  = list(model.model.classes_)
            idx_buy  = classes.index(1)  if  1 in classes else None
            idx_sell = classes.index(-1) if -1 in classes else None

            sym_trades = []
            for batch_i, df_idx in enumerate(valid_indices):
                pred_label = labels[batch_i]
                if pred_label == 0:
                    signal_counts["HOLD"] += 1
                    continue

                direction  = "long" if pred_label == 1 else "short"
                idx_dir    = idx_buy if direction == "long" else idx_sell
                confidence = float(probas[batch_i][idx_dir]) if idx_dir is not None else 0.0

                # Filtro de confianza
                if confidence < min_confidence:
                    signal_counts["LOW_CONF"] += 1
                    continue

                signal_counts["BUY" if direction == "long" else "SELL"] += 1

                row        = df.iloc[df_idx]
                price      = float(row["close"])

                # Filtros de estrategia
                if use_filters:
                    passes, failed = apply_strategy_filters(row, direction, settings)
                    if not passes:
                        for f in failed:
                            filter_rejections[f.split("(")[0].strip()] += 1
                        continue

                # Calcular SL y TP con ATR
                try:
                    levels = sl_calc.calculate(
                        direction=direction,
                        entry_price=price,
                        df=df.iloc[:df_idx+1],
                    )
                    sl = levels.stop_loss
                    tp = levels.take_profit
                except Exception:
                    sl_pct = 0.015
                    sl = price * (1 - sl_pct) if direction == "long" else price * (1 + sl_pct)
                    tp = price * (1 + sl_pct * 2) if direction == "long" else price * (1 - sl_pct * 2)

                # Simular el trade
                result = simulate_trade(
                    df=df, entry_idx=df_idx,
                    direction=direction,
                    entry_price=price, sl=sl, tp=tp,
                )
                result["symbol"]     = symbol
                result["direction"]  = direction
                result["confidence"] = confidence
                result["entry_price"]= price
                result["timestamp"]  = df.index[df_idx]
                sym_trades.append(result)

            all_trades.extend(sym_trades)
            symbol_stats[symbol] = _calc_symbol_stats(sym_trades)
            progress.advance(task)

    # ── Calcular metricas globales ──────────────────────────────────────────────
    return _calc_metrics(
        trades=all_trades,
        signal_counts=signal_counts,
        filter_rejections=filter_rejections,
        symbol_stats=symbol_stats,
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        label=label,
    )


def _calc_symbol_stats(trades: list) -> dict:
    if not trades:
        return {"n":0,"wins":0,"win_rate":0,"total_pnl":0,"avg_r":0}
    wins = [t for t in trades if t["won"]]
    pnls = [t["pnl_pct"] for t in trades]
    rs   = [t["r_value"] for t in trades]
    return {
        "n":        len(trades),
        "wins":     len(wins),
        "win_rate": len(wins)/len(trades),
        "total_pnl":sum(pnls),
        "avg_r":    np.mean(rs),
    }


def _calc_metrics(trades, signal_counts, filter_rejections,
                  symbol_stats, initial_capital, risk_per_trade, label) -> dict:
    if not trades:
        return {"label": label, "n_trades": 0, "error": "Sin trades generados"}

    wins   = [t for t in trades if t["won"]]
    losses = [t for t in trades if not t["won"]]
    pnls   = [t["pnl_pct"] for t in trades]
    rs     = [t["r_value"] for t in trades]

    win_rate      = len(wins) / len(trades)
    avg_win_pnl   = np.mean([t["pnl_pct"] for t in wins])   if wins   else 0
    avg_loss_pnl  = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    profit_factor = abs(sum(t["pnl_pct"] for t in wins) / sum(t["pnl_pct"] for t in losses)) \
                    if losses and sum(t["pnl_pct"] for t in losses) != 0 else 0

    # Simulacion de equity con riesgo fijo por trade
    capital  = initial_capital
    equity   = [capital]
    peak     = capital
    max_dd   = 0.0
    for t in trades:
        risk_usdt = capital * risk_per_trade
        pnl_usdt  = risk_usdt * t["r_value"]
        capital   = max(0, capital + pnl_usdt)
        equity.append(capital)
        peak      = max(peak, capital)
        dd        = (peak - capital) / peak
        max_dd    = max(max_dd, dd)

    final_capital = capital
    total_return  = (final_capital - initial_capital) / initial_capital

    # Sharpe ratio (diario aproximado)
    if len(pnls) > 2:
        arr    = np.array(pnls)
        sharpe = (arr.mean() / (arr.std() + 1e-10)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Por motivo de cierre
    by_reason = defaultdict(list)
    for t in trades:
        by_reason[t["exit_reason"]].append(t["pnl_pct"])

    # Por direccion
    longs  = [t for t in trades if t["direction"]=="long"]
    shorts = [t for t in trades if t["direction"]=="short"]

    return {
        "label":          label,
        "n_trades":       len(trades),
        "n_longs":        len(longs),
        "n_shorts":       len(shorts),
        "win_rate":       win_rate,
        "wins":           len(wins),
        "losses":         len(losses),
        "avg_win_pct":    avg_win_pnl  * 100,
        "avg_loss_pct":   avg_loss_pnl * 100,
        "profit_factor":  profit_factor,
        "sharpe":         sharpe,
        "max_drawdown":   max_dd * 100,
        "total_return":   total_return * 100,
        "final_capital":  final_capital,
        "initial_capital":initial_capital,
        "avg_r":          np.mean(rs),
        "median_r":       float(np.median(rs)),
        "avg_candles":    np.mean([t["candles"] for t in trades]),
        "signal_counts":  dict(signal_counts),
        "filter_rejections": dict(filter_rejections),
        "symbol_stats":   symbol_stats,
        "by_reason":      {k: {"n": len(v), "avg_pnl": np.mean(v)*100} for k,v in by_reason.items()},
        "long_win_rate":  len([t for t in longs  if t["won"]]) / len(longs)  if longs  else 0,
        "short_win_rate": len([t for t in shorts if t["won"]]) / len(shorts) if shorts else 0,
        "equity_curve":   equity,
        "trades":         trades,
    }


# ── Impresion de resultados ────────────────────────────────────────────────────

def print_report(m: dict, show_symbols: bool = True):
    if "error" in m:
        console.print(f"[red]{m['error']}[/red]")
        return

    label = f" — {m['label']}" if m["label"] else ""

    # Cabecera
    pnl_color = "green" if m["total_return"] >= 0 else "red"
    console.print(Panel(
        f"[bold]Resultado del backtest{label}[/bold]\n"
        f"[{pnl_color}]Retorno total: {m['total_return']:+.2f}%  "
        f"(${m['initial_capital']:,.0f} → ${m['final_capital']:,.2f})[/{pnl_color}]",
        border_style="cyan",
    ))

    # Metricas principales
    t1 = Table(box=box.ROUNDED, show_header=False, border_style="dim", padding=(0,1))
    t1.add_column("Metrica", style="dim", width=26)
    t1.add_column("Valor",   style="bold")
    t1.add_column("",        style="dim", width=26)
    t1.add_column("",        style="bold")

    wr_color = "green" if m["win_rate"] >= 0.5 else "red"
    pf_color = "green" if m["profit_factor"] >= 1.0 else "red"
    sh_color = "green" if m["sharpe"] >= 1.0 else "yellow" if m["sharpe"] >= 0 else "red"
    dd_color = "red"   if m["max_drawdown"] > 20 else "yellow" if m["max_drawdown"] > 10 else "green"

    t1.add_row("Total trades",    str(m["n_trades"]),
               "Win rate",        f"[{wr_color}]{m['win_rate']*100:.1f}%[/{wr_color}]")
    t1.add_row("Longs / Shorts",  f"{m['n_longs']} / {m['n_shorts']}",
               "Profit Factor",   f"[{pf_color}]{m['profit_factor']:.2f}[/{pf_color}]")
    t1.add_row("Win rate LONG",   f"{m['long_win_rate']*100:.1f}%",
               "Sharpe ratio",    f"[{sh_color}]{m['sharpe']:.2f}[/{sh_color}]")
    t1.add_row("Win rate SHORT",  f"{m['short_win_rate']*100:.1f}%",
               "Max drawdown",    f"[{dd_color}]{m['max_drawdown']:.1f}%[/{dd_color}]")
    t1.add_row("Avg ganancia",    f"{m['avg_win_pct']:+.2f}%",
               "Avg R por trade", f"{m['avg_r']:+.3f}R")
    t1.add_row("Avg perdida",     f"{m['avg_loss_pct']:+.2f}%",
               "Duracion media",  f"{m['avg_candles']:.1f} velas")
    console.print(t1)

    # Por motivo de cierre
    console.print("\n[bold dim]Cierres por motivo:[/bold dim]")
    t2 = Table(box=box.SIMPLE, show_header=True, border_style="dim")
    t2.add_column("Motivo",     style="dim")
    t2.add_column("N trades",   justify="right")
    t2.add_column("PnL medio",  justify="right")
    for reason, stats in sorted(m["by_reason"].items(), key=lambda x: -x[1]["n"]):
        color = "green" if stats["avg_pnl"] >= 0 else "red"
        t2.add_row(reason, str(stats["n"]),
                   f"[{color}]{stats['avg_pnl']:+.2f}%[/{color}]")
    console.print(t2)

    # Distribucion de señales
    sc = m["signal_counts"]
    total_raw = sum(sc.values())
    console.print(f"\n[bold dim]Señales generadas (total raw: {total_raw:,}):[/bold dim]")
    for k,v in sorted(sc.items(), key=lambda x: -x[1]):
        pct = v/total_raw*100 if total_raw else 0
        console.print(f"  {k:12s} {v:6,}  ({pct:.1f}%)")

    # Filtros que rechazaron señales
    if m["filter_rejections"]:
        console.print(f"\n[bold dim]Señales rechazadas por filtro:[/bold dim]")
        for k,v in sorted(m["filter_rejections"].items(), key=lambda x: -x[1]):
            console.print(f"  {k:30s} {v:5,} rechazos")

    # Top/bottom simbolos
    if show_symbols and m["symbol_stats"]:
        stats_sorted = sorted(
            [(s, d) for s,d in m["symbol_stats"].items() if d["n"] > 0],
            key=lambda x: -x[1]["avg_r"]
        )
        console.print(f"\n[bold dim]Rendimiento por simbolo (top 5 / bottom 5):[/bold dim]")
        t3 = Table(box=box.SIMPLE, show_header=True, border_style="dim")
        t3.add_column("Simbolo",  width=14)
        t3.add_column("Trades",   justify="right")
        t3.add_column("Win rate", justify="right")
        t3.add_column("Avg R",    justify="right")
        t3.add_column("PnL total",justify="right")

        top    = stats_sorted[:5]
        bottom = stats_sorted[-5:]
        shown  = set()
        for sym, s in top + bottom:
            if sym in shown: continue
            shown.add(sym)
            wr_c  = "green" if s["win_rate"] >= 0.5 else "red"
            r_c   = "green" if s["avg_r"]    >= 0   else "red"
            pnl_c = "green" if s["total_pnl"]>= 0   else "red"
            t3.add_row(
                sym,
                str(s["n"]),
                f"[{wr_c}]{s['win_rate']*100:.0f}%[/{wr_c}]",
                f"[{r_c}]{s['avg_r']:+.2f}R[/{r_c}]",
                f"[{pnl_c}]{s['total_pnl']*100:+.1f}%[/{pnl_c}]",
            )
        console.print(t3)


def print_comparison(m1: dict, m2: dict):
    """Imprime tabla comparativa sin filtros vs con filtros."""
    console.print(Panel("[bold]Comparacion: sin filtros vs con filtros[/bold]", border_style="cyan"))

    t = Table(box=box.ROUNDED, border_style="dim", padding=(0,1))
    t.add_column("Metrica",        style="dim", width=22)
    t.add_column("Sin filtros",    justify="right")
    t.add_column("Con filtros",    justify="right")
    t.add_column("Diferencia",     justify="right")

    def row(label, key, fmt_fn, better="higher"):
        v1 = m1.get(key, 0)
        v2 = m2.get(key, 0)
        diff = v2 - v1
        if better == "higher":
            color = "green" if diff > 0 else "red" if diff < 0 else "dim"
        else:
            color = "green" if diff < 0 else "red" if diff > 0 else "dim"
        t.add_row(label, fmt_fn(v1), fmt_fn(v2),
                  f"[{color}]{'+' if diff>=0 else ''}{fmt_fn(diff)}[/{color}]")

    row("Total trades",    "n_trades",       lambda v: str(int(v)))
    row("Win rate",        "win_rate",       lambda v: f"{v*100:.1f}%")
    row("Profit factor",   "profit_factor",  lambda v: f"{v:.2f}")
    row("Sharpe ratio",    "sharpe",         lambda v: f"{v:.2f}")
    row("Max drawdown",    "max_drawdown",   lambda v: f"{v:.1f}%", better="lower")
    row("Retorno total",   "total_return",   lambda v: f"{v:+.1f}%")
    row("Avg R",           "avg_r",          lambda v: f"{v:+.3f}R")
    t.add_row("Capital final",
              f"${m1.get('final_capital',0):,.2f}",
              f"${m2.get('final_capital',0):,.2f}",
              "")
    console.print(t)


def print_confidence_sweep(results: list[dict]):
    """Muestra como cambian las metricas al variar el umbral de confianza."""
    console.print(Panel("[bold]Sweep de umbral de confianza[/bold]", border_style="cyan"))

    t = Table(box=box.ROUNDED, border_style="dim", padding=(0,1))
    t.add_column("Confianza", style="bold", justify="right")
    t.add_column("Trades",    justify="right")
    t.add_column("Win rate",  justify="right")
    t.add_column("Prof. Factor", justify="right")
    t.add_column("Sharpe",    justify="right")
    t.add_column("Max DD",    justify="right")
    t.add_column("Retorno",   justify="right")

    for m in results:
        if "error" in m: continue
        wr_c = "green" if m["win_rate"] >= 0.5 else "red"
        ret_c = "green" if m["total_return"] >= 0 else "red"
        t.add_row(
            m["label"],
            str(m["n_trades"]),
            f"[{wr_c}]{m['win_rate']*100:.1f}%[/{wr_c}]",
            f"{m['profit_factor']:.2f}",
            f"{m['sharpe']:.2f}",
            f"{m['max_drawdown']:.1f}%",
            f"[{ret_c}]{m['total_return']:+.1f}%[/{ret_c}]",
        )
    console.print(t)


# ── CLI ────────────────────────────────────────────────────────────────────────

DEFAULT_SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
    "DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","DOTUSDT",
]

@click.command()
@click.option("--symbols", "-s", multiple=True,
              help="Simbolos a testear (default: top 10 majors)")
@click.option("--days",    "-d", default=60, show_default=True,
              help="Dias de historico a evaluar")
@click.option("--tf", default="15m", show_default=True,
              help="Timeframe")
@click.option("--confidence", "-c", default=None, type=float,
              help="Umbral de confianza (default: del .env)")
@click.option("--strategy/--no-strategy", default=True, show_default=True,
              help="Aplicar filtros de estrategia (RSI, volumen, BB)")
@click.option("--compare", is_flag=True, default=False,
              help="Comparar con y sin filtros de estrategia")
@click.option("--sweep-confidence", is_flag=True, default=False,
              help="Probar distintos umbrales de confianza (50% a 85%)")
@click.option("--capital", default=1000.0, show_default=True,
              help="Capital inicial para simulacion de equity")
@click.option("--risk", default=0.02, show_default=True,
              help="Riesgo por trade como fraccion del capital (0.02 = 2%%)")
def main(symbols, days, tf, confidence, strategy, compare, sweep_confidence, capital, risk):
    """
    Backtest del modelo IA con datos reales de Binance.

    Evalua la efectividad de las señales del modelo sobre datos historicos
    y simula el rendimiento que habrian tenido las operaciones.
    """
    from rich.console import Console as C
    C().print(Panel.fit(
        "[bold cyan]Backtest — Modelo IA Trading Bot[/bold cyan]\n"
        "[dim]Evaluacion con datos reales de Binance[/dim]",
        border_style="cyan", padding=(0,3),
    ))

    settings = get_settings()
    syms     = list(symbols) if symbols else DEFAULT_SYMBOLS
    conf     = confidence or settings.min_signal_confidence

    console.print(f"\n  Simbolos:    [cyan]{', '.join(syms)}[/cyan]")
    console.print(f"  Periodo:     [cyan]{days} dias ({tf})[/cyan]")
    console.print(f"  Confianza:   [cyan]{conf*100:.0f}%[/cyan]")
    console.print(f"  Filtros:     [cyan]{'Si' if strategy else 'No'}[/cyan]")
    console.print(f"  Capital:     [cyan]${capital:,.0f}[/cyan]")
    console.print(f"  Riesgo/trade:[cyan]{risk*100:.1f}%[/cyan]\n")

    # ── Modo sweep de confianza ────────────────────────────────────────────────
    if sweep_confidence:
        thresholds = [0.50, 0.55, 0.57, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
        results    = []
        for thr in thresholds:
            console.print(f"[dim]Probando confianza {thr*100:.0f}%...[/dim]")
            m = run_backtest(
                symbols=syms, days=days, timeframe=tf,
                min_confidence=thr, use_filters=strategy,
                initial_capital=capital, risk_per_trade=risk,
                leverage=settings.max_leverage,
                label=f"{thr*100:.0f}%",
            )
            results.append(m)
        print_confidence_sweep(results)
        return

    # ── Modo comparacion ───────────────────────────────────────────────────────
    if compare:
        console.print("[dim]Corriendo sin filtros...[/dim]")
        m_no_filter = run_backtest(
            symbols=syms, days=days, timeframe=tf,
            min_confidence=conf, use_filters=False,
            initial_capital=capital, risk_per_trade=risk,
            leverage=settings.max_leverage,
            label="Sin filtros",
        )
        console.print("[dim]Corriendo con filtros...[/dim]")
        m_with_filter = run_backtest(
            symbols=syms, days=days, timeframe=tf,
            min_confidence=conf, use_filters=True,
            initial_capital=capital, risk_per_trade=risk,
            leverage=settings.max_leverage,
            label="Con filtros",
        )
        print_report(m_no_filter, show_symbols=False)
        console.print()
        print_report(m_with_filter, show_symbols=False)
        console.print()
        print_comparison(m_no_filter, m_with_filter)
        return

    # ── Modo normal ────────────────────────────────────────────────────────────
    m = run_backtest(
        symbols=syms, days=days, timeframe=tf,
        min_confidence=conf, use_filters=strategy,
        initial_capital=capital, risk_per_trade=risk,
        leverage=settings.max_leverage,
        label="",
    )
    print_report(m)

    # Resumen final
    if m.get("n_trades", 0) > 0:
        console.print(Panel(
            f"[bold]Conclusion:[/bold]\n\n"
            f"El modelo generó [cyan]{m['n_trades']}[/cyan] trades en {days} dias "
            f"sobre {len(syms)} simbolos.\n"
            f"Win rate: [{'green' if m['win_rate']>=0.5 else 'red'}]{m['win_rate']*100:.1f}%[/]\n"
            f"Profit Factor: [{'green' if m['profit_factor']>=1 else 'red'}]{m['profit_factor']:.2f}[/]\n"
            f"Retorno simulado: [{'green' if m['total_return']>=0 else 'red'}]{m['total_return']:+.1f}%[/]\n\n"
            f"[dim]Nota: el backtest NO incluye comisiones, slippage ni funding rates.[/dim]",
            border_style="green" if m.get("total_return",0) >= 0 else "red",
        ))


if __name__ == "__main__":
    main()
