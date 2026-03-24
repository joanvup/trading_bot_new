"""
=============================================================
main.py
Punto de entrada principal del bot de trading.

Uso:
    python main.py --mode testnet     # Modo testnet
    python main.py --mode dry_run     # Modo simulacion (default)
    python main.py --mode live        # Modo real (con precaucion)
    python main.py --check            # Solo verifica conexiones
    python main.py --init             # Inicializacion completa (BD + datos + modelo)
    python main.py --init --symbols 5 # Inicializar con N simbolos top

=============================================================
"""

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import box

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import get_settings, TradingMode
from core.binance_client import BinanceFuturesClient
from database.connection import init_database
from utils.logger import setup_logging, get_logger

console = Console()
log = get_logger(__name__)


def print_banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]🤖 TRADING BOT IA[/bold cyan]\n"
            "[dim]Binance Futures USDT-M | ML + TA + Risk Management[/dim]",
            border_style="cyan",
            padding=(1, 4),
        )
    )


def print_config_summary(settings) -> None:
    mode_colors = {
        TradingMode.DRY_RUN: "yellow",
        TradingMode.TESTNET: "blue",
        TradingMode.LIVE:    "red",
    }
    mode_icons = {
        TradingMode.DRY_RUN: "🟡",
        TradingMode.TESTNET: "🔵",
        TradingMode.LIVE:    "🔴",
    }

    color = mode_colors[settings.trading_mode]
    icon  = mode_icons[settings.trading_mode]

    table = Table(box=box.ROUNDED, show_header=False, border_style="dim", padding=(0, 1))
    table.add_column("Parámetro", style="dim", width=30)
    table.add_column("Valor",     style="bold")

    table.add_row("Modo de trading",        f"[{color}]{icon} {settings.trading_mode.value.upper()}[/{color}]")
    table.add_row("Capital inicial",        f"${settings.initial_capital:,.2f} USDT")
    table.add_row("Riesgo por operación",   f"{settings.max_risk_per_trade*100:.1f}%")
    table.add_row("Drawdown máximo",        f"{settings.max_drawdown*100:.1f}%")
    table.add_row("Posiciones simultáneas", str(settings.max_open_positions))
    table.add_row("Apalancamiento máximo",  f"{settings.max_leverage}x")
    table.add_row("Timeframe principal",    settings.primary_timeframe.value)
    table.add_row("Timeframe secundario",   settings.secondary_timeframe.value)
    table.add_row("Top símbolos a analizar",str(settings.top_symbols_count))
    table.add_row("Base de datos",          f"{settings.db_host}:{settings.db_port}/{settings.db_name}")

    console.print(table)


async def verify_connections(settings) -> bool:
    all_ok = True
    console.print("\n[bold]Verificando conexiones...[/bold]")

    try:
        console.print("  PostgreSQL...", end=" ")
        db = init_database(settings.database_url, settings.async_database_url)
        ok = await db.check_connection()
        if ok:
            console.print("[green]✓ OK[/green]")
        else:
            console.print("[red]✗ FAILED[/red]")
            all_ok = False
    except Exception as e:
        console.print(f"[red]✗ ERROR: {e}[/red]")
        all_ok = False

    try:
        console.print(f"  Binance ({settings.trading_mode.value})...", end=" ")
        client = BinanceFuturesClient(settings)
        client.initialize()
        balance = client.get_account_balance()
        avail   = float(balance.get("availableBalance", 0))
        console.print(f"[green]✓ OK | Balance: ${avail:,.2f} USDT[/green]")
    except Exception as e:
        console.print(f"[red]✗ ERROR: {e}[/red]")
        all_ok = False

    return all_ok


# ── Inicializacion completa ────────────────────────────────────────────────────

async def run_init(settings, n_symbols: int, days_back: int, use_optuna: bool) -> None:
    """
    Inicializacion LIMPIA y completa del sistema.

    Secuencia:
      0. Confirmar con el usuario (accion destructiva)
      1. Borrar modelos guardados del disco
      2. Limpiar base de datos (drop + recrear todas las tablas)
      3. Sincronizar simbolos desde Binance
      4. Descargar historico OHLCV
      5. Entrenar modelo IA inicial
    """
    console.print(
        Panel(
            "[bold yellow]⚠  Inicializacion limpia del sistema[/bold yellow]\n\n"
            "Esta operacion va a:\n"
            "  • Borrar [bold]todos los modelos IA[/bold] guardados en disco\n"
            "  • Borrar [bold]todos los datos[/bold] de la base de datos\n"
            "    (trades, señales, velas, snapshots, simbolos)\n"
            "  • Descargar datos frescos y entrenar un modelo nuevo\n\n"
            "[dim]El sistema quedara como si fuera la primera vez que se ejecuta.[/dim]",
            border_style="yellow",
        )
    )

    if not click.confirm("\n¿Confirmas que quieres borrar todo y empezar de cero?", default=False):
        console.print("[dim]Inicializacion cancelada.[/dim]")
        return

    console.print()

    # ── Paso 0: Verificar conexiones ───────────────────────────────────────────
    console.print("[bold cyan]Paso 1/5 — Verificando conexiones[/bold cyan]")
    ok = await verify_connections(settings)
    if not ok:
        console.print("[bold red]✗ Conexiones fallidas. Revisa tu .env y PostgreSQL.[/bold red]")
        sys.exit(1)

    # ── Paso 1: Borrar modelos del disco ───────────────────────────────────────
    console.print("\n[bold cyan]Paso 2/5 — Limpiando modelos guardados[/bold cyan]")
    models_dir = settings.get_models_path()
    deleted_models = 0

    if models_dir.exists():
        for f in models_dir.iterdir():
            if f.is_file():
                f.unlink()
                deleted_models += 1
                log.debug(f"Borrado: {f.name}")

    if deleted_models > 0:
        console.print(f"  [green]✓ {deleted_models} archivos de modelos eliminados[/green]")
    else:
        console.print("  [dim]No habia modelos guardados[/dim]")

    # ── Paso 2: Limpiar base de datos ──────────────────────────────────────────
    console.print("\n[bold cyan]Paso 3/5 — Limpiando base de datos[/bold cyan]")

    from sqlalchemy import create_engine, text
    from database.models import Base

    console.print("  Eliminando todas las tablas...", end=" ")
    try:
        engine = create_engine(settings.database_url, echo=False)
        with engine.begin() as conn:
            # Deshabilitar foreign keys temporalmente para borrar en cualquier orden
            conn.execute(text("SET session_replication_role = replica"))
            Base.metadata.drop_all(bind=engine)
            conn.execute(text("SET session_replication_role = DEFAULT"))
        console.print("[green]✓[/green]")
    except Exception as e:
        console.print(f"[red]✗ {e}[/red]")
        sys.exit(1)

    console.print("  Recreando tablas limpias...", end=" ")
    try:
        Base.metadata.create_all(bind=engine)
        engine.dispose()
        console.print("[green]✓[/green]")
    except Exception as e:
        console.print(f"[red]✗ {e}[/red]")
        sys.exit(1)

    # Reinicializar el gestor de BD con las tablas nuevas
    db = init_database(settings.database_url, settings.async_database_url)
    console.print("  [green]✓ Base de datos limpia y lista[/green]")

    # ── Paso 3: Sincronizar simbolos ───────────────────────────────────────────
    console.print("\n[bold cyan]Paso 4/5 — Descargando datos de mercado[/bold cyan]")

    client = BinanceFuturesClient(settings)
    client.initialize()

    from data.historic_collector import HistoricCollector
    from analysis.asset_selector import AssetSelector
    from analysis.indicators import TechnicalIndicators

    collector = HistoricCollector(client, db)

    console.print("  Sincronizando simbolos desde Binance...", end=" ")
    await collector.sync_symbols()
    console.print("[green]✓[/green]")

    # Calcular ranking real
    console.print("  Calculando ranking de activos...", end=" ")
    ti       = TechnicalIndicators()
    selector = AssetSelector(client=client, indicators=ti, top_n=n_symbols)
    top      = await selector.get_top_symbols(force_refresh=True)
    top_names = [s.symbol for s in top]
    console.print(f"[green]✓ {len(top_names)} simbolos seleccionados[/green]")

    for s in top[:5]:
        console.print(
            f"    #{s.rank} [cyan]{s.symbol}[/cyan]  "
            f"Score:{s.score:.1f}  Vol:{s.volume_24h_usdt/1e6:.0f}M$  {s.change_24h_pct:+.2f}%"
        )
    if len(top) > 5:
        console.print(f"    [dim]... y {len(top)-5} mas[/dim]")

    # Descargar historico OHLCV
    console.print(f"\n  Descargando {days_back} dias de velas OHLCV...")
    timeframes = [settings.primary_timeframe.value, settings.secondary_timeframe.value]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task          = progress.add_task("Descargando...", total=len(top_names))
        total_candles = 0

        for symbol in top_names:
            progress.update(task, description=f"[cyan]{symbol}[/cyan]")
            for tf in timeframes:
                try:
                    import time as _time
                    start_ms = int((_time.time() - days_back * 86400) * 1000)
                    inserted = await collector._download_symbol_tf(
                        symbol=symbol,
                        timeframe=tf,
                        start_time_ms=start_ms,
                    )
                    total_candles += inserted
                    await asyncio.sleep(0.05)
                except Exception as e:
                    log.warning(f"Error descargando {symbol}/{tf}: {e}")
            progress.advance(task)

    console.print(f"  [green]✓ {total_candles:,} velas insertadas[/green]")

    # ── Paso 4: Entrenar modelo ────────────────────────────────────────────────
    console.print(f"\n[bold cyan]Paso 5/5 — Entrenando modelo IA[/bold cyan]")
    console.print(
        f"  Simbolos: {len(top_names)} | "
        f"Optuna: {'Si (~8-10 min)' if use_optuna else 'No (~1-2 min)'}"
    )

    from ai.model_trainer import ModelTrainer
    trainer = ModelTrainer(settings, db)

    try:
        result = await trainer.train(
            symbols=top_names,
            timeframe=settings.primary_timeframe.value,
            use_optuna=use_optuna,
            n_trials=30 if use_optuna else 0,
        )
        m = result["metrics"]
        console.print(f"  [green]✓ Modelo entrenado v{result['version']}[/green]")
        console.print(f"  Muestras train/test: {result['n_train']:,} / {result['n_test']:,}")
        console.print(f"  Test accuracy:  [cyan]{m['test_accuracy']:.3f}[/cyan]")
        console.print(f"  F1 weighted:    [cyan]{m['f1_weighted']:.3f}[/cyan]")
        console.print(f"  F1 BUY/SELL:    [cyan]{m['f1_buy']:.3f}[/cyan] / [cyan]{m['f1_sell']:.3f}[/cyan]")
        console.print(f"  Duracion:       {result['duration_s']}s")

    except Exception as e:
        console.print(f"  [yellow]⚠ Entrenamiento fallido: {e}[/yellow]")
        console.print("  [dim]Puedes reentrenar mas tarde con el bot corriendo[/dim]")
        log.error(f"Error en entrenamiento inicial: {e}", exc_info=True)

    # ── Resumen ────────────────────────────────────────────────────────────────
    console.print(
        Panel(
            "[bold green]✓ Sistema inicializado desde cero[/bold green]\n\n"
            "Base de datos limpia  ✓\n"
            "Modelos anteriores eliminados  ✓\n"
            f"Historico descargado ({total_candles:,} velas)  ✓\n"
            "Modelo IA entrenado  ✓\n\n"
            "Ahora puedes arrancar el bot con:\n"
            "  [cyan]python main.py[/cyan]",
            border_style="green",
        )
    )


# ── CLI con Click ──────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--mode",
    type=click.Choice(["testnet", "dry_run", "live"]),
    default=None,
    help="Modo de operacion (sobreescribe el .env)",
)
@click.option(
    "--check",
    is_flag=True,
    default=False,
    help="Solo verifica conexiones y sale",
)
@click.option(
    "--init",
    is_flag=True,
    default=False,
    help="Inicializacion completa: BD + datos historicos + modelo IA",
)
@click.option(
    "--symbols",
    type=int,
    default=15,
    show_default=True,
    help="Numero de simbolos top a usar en --init",
)
@click.option(
    "--days",
    type=int,
    default=90,
    show_default=True,
    help="Dias de historico a descargar en --init",
)
@click.option(
    "--optuna",
    is_flag=True,
    default=False,
    help="Usar Optuna para optimizar hiperparametros en --init (mas lento)",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default=None,
    help="Nivel de logging",
)
def main(mode: str, check: bool, init: bool, symbols: int,
         days: int, optuna: bool, log_level: str):
    """🤖 Bot de Trading IA para Binance Futures USDT-M"""

    print_banner()

    try:
        settings = get_settings()
    except Exception as e:
        console.print(f"\n[bold red]Error cargando configuracion:[/bold red] {e}")
        console.print(
            "\n[dim]Asegurate de que existe el archivo [bold].env[/bold]\n"
            "Copia [bold].env.example[/bold] como [bold].env[/bold] y rellena tus valores.[/dim]"
        )
        sys.exit(1)

    if mode:
        import os
        os.environ["TRADING_MODE"] = mode
        get_settings.cache_clear()
        settings = get_settings()

    setup_logging(
        log_level=log_level or settings.log_level,
        log_file=settings.log_file,
        log_rotation=settings.log_rotation,
        log_retention=settings.log_retention,
    )

    console.print()
    print_config_summary(settings)

    # ── Modo --check ───────────────────────────────────────────────────────────
    if check:
        console.print()
        ok = asyncio.run(verify_connections(settings))
        if ok:
            console.print("\n[bold green]✓ Todas las conexiones estan OK[/bold green]\n")
            sys.exit(0)
        else:
            console.print("\n[bold red]✗ Hay errores en las conexiones[/bold red]\n")
            sys.exit(1)

    # ── Modo --init ────────────────────────────────────────────────────────────
    if init:
        console.print()
        try:
            asyncio.run(run_init(settings, n_symbols=symbols, days_back=days, use_optuna=optuna))
        except KeyboardInterrupt:
            console.print("\n[yellow]Inicializacion interrumpida[/yellow]")
        except Exception as e:
            log.error(f"Error en inicializacion: {e}", exc_info=True)
            console.print(f"\n[bold red]Error en inicializacion: {e}[/bold red]")
            sys.exit(1)
        sys.exit(0)

    # ── Modo live: confirmacion extra ──────────────────────────────────────────
    if settings.is_live:
        console.print("\n[bold red]⚠️  MODO LIVE - DINERO REAL ⚠️[/bold red]")
        if not click.confirm("¿Confirmas operar con dinero real?", default=False):
            console.print("[dim]Cancelado.[/dim]")
            sys.exit(0)

    # ── Arrancar el bot ────────────────────────────────────────────────────────
    console.print(
        f"\n[bold green]Iniciando bot en modo "
        f"{settings.trading_mode.value.upper()}...[/bold green]"
    )
    log.info(f"Bot iniciando | Modo: {settings.trading_mode.value}")

    try:
        asyncio.run(run_bot(settings))
    except KeyboardInterrupt:
        console.print("\n[yellow]Bot detenido por el usuario[/yellow]")
        log.info("Bot detenido por KeyboardInterrupt")
    except Exception as e:
        log.error(f"Error critico en el bot: {e}", exc_info=True)
        console.print(f"\n[bold red]Error critico: {e}[/bold red]")
        sys.exit(1)


async def run_bot(settings) -> None:
    """Funcion principal asincrona del bot."""
    log.info("Inicializando componentes del bot...")

    db = init_database(settings.database_url, settings.async_database_url)
    if not await db.check_connection():
        raise RuntimeError("No se pudo conectar a PostgreSQL")

    client = BinanceFuturesClient(settings)
    client.initialize()

    balance = client.get_account_balance()
    log.info(
        f"Balance: Total={balance.get('totalWalletBalance','N/A')} USDT | "
        f"Disponible={balance.get('availableBalance','N/A')} USDT"
    )

    from execution.trading_loop import TradingLoop
    trading_loop = TradingLoop(settings, client, db)

    console.print("\n[bold cyan]Inicializando sistema completo...[/bold cyan]")
    await trading_loop.initialize()

    status = trading_loop.get_status()
    risk   = status["risk"]

    console.print(f"\n[bold green]Sistema listo:[/bold green]")
    console.print(f"  Balance:    [cyan]${risk['balance']:,.2f} USDT[/cyan]")
    console.print(f"  Modelo IA:  [cyan]{status['model_version'] or 'sin modelo'}[/cyan]")
    console.print(f"  Simbolos:   [cyan]{status['pipeline']['top_symbols']}[/cyan]")
    console.print(f"  Modo:       [cyan]{settings.trading_mode.value.upper()}[/cyan]")

    from api.server import start_api_server
    console.print(
        f"  Dashboard:  [cyan]http://{settings.api_host}:{settings.api_port}[/cyan]"
    )

    console.print("\n[bold green]Bot activo — todas las fases corriendo[/bold green]")
    console.print("[dim]Presiona Ctrl+C para detener[/dim]\n")

    await asyncio.gather(
        trading_loop.run_forever(),
        start_api_server(settings, trading_loop, db),
    )


if __name__ == "__main__":
    main()



def print_banner() -> None:
    """Imprime el banner de inicio del bot."""
    console.print(
        Panel.fit(
            "[bold cyan]🤖 TRADING BOT IA[/bold cyan]\n"
            "[dim]Binance Futures USDT-M | ML + TA + Risk Management[/dim]",
            border_style="cyan",
            padding=(1, 4),
        )
    )


def print_config_summary(settings) -> None:
    """Muestra un resumen de la configuración activa."""
    mode_colors = {
        TradingMode.DRY_RUN: "yellow",
        TradingMode.TESTNET: "blue",
        TradingMode.LIVE:    "red",
    }
    mode_icons = {
        TradingMode.DRY_RUN: "🟡",
        TradingMode.TESTNET: "🔵",
        TradingMode.LIVE:    "🔴",
    }

    color = mode_colors[settings.trading_mode]
    icon = mode_icons[settings.trading_mode]

    table = Table(
        box=box.ROUNDED,
        show_header=False,
        border_style="dim",
        padding=(0, 1),
    )
    table.add_column("Parámetro", style="dim", width=30)
    table.add_column("Valor", style="bold")

    table.add_row(
        "Modo de trading",
        f"[{color}]{icon} {settings.trading_mode.value.upper()}[/{color}]"
    )
    table.add_row("Capital inicial", f"${settings.initial_capital:,.2f} USDT")
    table.add_row("Riesgo por operación", f"{settings.max_risk_per_trade*100:.1f}%")
    table.add_row("Drawdown máximo", f"{settings.max_drawdown*100:.1f}%")
    table.add_row("Posiciones simultáneas", str(settings.max_open_positions))
    table.add_row("Apalancamiento máximo", f"{settings.max_leverage}x")
    table.add_row("Timeframe principal", settings.primary_timeframe.value)
    table.add_row("Timeframe secundario", settings.secondary_timeframe.value)
    table.add_row("Top símbolos a analizar", str(settings.top_symbols_count))
    table.add_row("Base de datos", f"{settings.db_host}:{settings.db_port}/{settings.db_name}")

    console.print(table)


async def verify_connections(settings) -> bool:
    """
    Verifica todas las conexiones necesarias.

    Returns:
        True si todo está OK, False si hay algún error
    """
    all_ok = True
    console.print("\n[bold]Verificando conexiones...[/bold]")

    # ── 1. Base de datos ───────────────────────────────────────────────────────
    try:
        console.print("  PostgreSQL...", end=" ")
        db = init_database(settings.database_url, settings.async_database_url)
        ok = await db.check_connection()
        if ok:
            console.print("[green]✓ OK[/green]")
        else:
            console.print("[red]✗ FAILED[/red]")
            all_ok = False
    except Exception as e:
        console.print(f"[red]✗ ERROR: {e}[/red]")
        all_ok = False

    # ── 2. Binance API ─────────────────────────────────────────────────────────
    try:
        console.print(f"  Binance ({settings.trading_mode.value})...", end=" ")

        if settings.is_dry_run:
            console.print("[yellow]✓ DRY RUN (sin conexión real)[/yellow]")
        else:
            client = BinanceFuturesClient(settings)
            client.initialize()

            # Verificar balance
            balance = client.get_account_balance()
            avail = float(balance.get("availableBalance", 0))
            console.print(f"[green]✓ OK | Balance: ${avail:,.2f} USDT[/green]")

    except Exception as e:
        console.print(f"[red]✗ ERROR: {e}[/red]")
        all_ok = False

    return all_ok


# ── CLI con Click ──────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--mode",
    type=click.Choice(["testnet", "dry_run", "live"]),
    default=None,
    help="Modo de operación (sobreescribe el .env)",
)
@click.option(
    "--check",
    is_flag=True,
    default=False,
    help="Solo verifica conexiones y sale",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default=None,
    help="Nivel de logging",
)
def main(mode: str, check: bool, log_level: str):
    """🤖 Bot de Trading IA para Binance Futures USDT-M"""

    # Imprimir banner
    print_banner()

    # Cargar configuración
    try:
        settings = get_settings()
    except Exception as e:
        console.print(f"\n[bold red]Error cargando configuración:[/bold red] {e}")
        console.print(
            "\n[dim]Asegúrate de que existe el archivo [bold].env[/bold] "
            "en el directorio raíz del proyecto.[/dim]\n"
            "[dim]Copia [bold].env.example[/bold] como [bold].env[/bold] "
            "y rellena tus valores.[/dim]"
        )
        sys.exit(1)

    # Sobreescribir modo si se especificó por CLI
    if mode:
        import os
        os.environ["TRADING_MODE"] = mode
        from functools import lru_cache
        get_settings.cache_clear()
        settings = get_settings()

    # Configurar logging
    setup_logging(
        log_level=log_level or settings.log_level,
        log_file=settings.log_file,
        log_rotation=settings.log_rotation,
        log_retention=settings.log_retention,
    )

    # Mostrar configuración
    console.print()
    print_config_summary(settings)

    # Modo verificación solamente
    if check:
        console.print()
        ok = asyncio.run(verify_connections(settings))
        if ok:
            console.print("\n[bold green]✓ Todas las conexiones están OK[/bold green]\n")
            sys.exit(0)
        else:
            console.print("\n[bold red]✗ Hay errores en las conexiones[/bold red]\n")
            sys.exit(1)

    # Advertencia para modo live
    if settings.is_live:
        console.print(
            "\n[bold red]⚠️  MODO LIVE ACTIVADO - OPERANDO CON DINERO REAL ⚠️[/bold red]"
        )
        confirm = click.confirm(
            "¿Estás seguro de que quieres operar con dinero real?",
            default=False
        )
        if not confirm:
            console.print("[dim]Operación cancelada.[/dim]")
            sys.exit(0)

    # Iniciar el bot
    console.print(
        f"\n[bold green]Iniciando bot en modo "
        f"{settings.trading_mode.value.upper()}...[/bold green]"
    )
    log.info(f"Bot iniciando | Modo: {settings.trading_mode.value}")

    try:
        asyncio.run(run_bot(settings))
    except KeyboardInterrupt:
        console.print("\n[yellow]Bot detenido por el usuario[/yellow]")
        log.info("Bot detenido por KeyboardInterrupt")
    except Exception as e:
        log.error(f"Error crítico en el bot: {e}", exc_info=True)
        console.print(f"\n[bold red]Error crítico: {e}[/bold red]")
        sys.exit(1)


async def run_bot(settings) -> None:
    """Funcion principal asincrona del bot."""
    log.info("Inicializando componentes del bot...")

    # 1. Base de datos
    db = init_database(settings.database_url, settings.async_database_url)
    if not await db.check_connection():
        raise RuntimeError("No se pudo conectar a PostgreSQL")

    # 2. Cliente Binance
    client = BinanceFuturesClient(settings)
    client.initialize()

    balance = client.get_account_balance()
    log.info(
        f"Balance: Total={balance.get('totalWalletBalance','N/A')} USDT | "
        f"Disponible={balance.get('availableBalance','N/A')} USDT"
    )

    # 3. Trading Loop — orquesta todas las fases
    from execution.trading_loop import TradingLoop
    trading_loop = TradingLoop(settings, client, db)

    console.print("\n[bold cyan]Inicializando sistema completo...[/bold cyan]")
    await trading_loop.initialize()

    status = trading_loop.get_status()
    risk   = status["risk"]

    console.print(f"\n[bold green]Sistema listo:[/bold green]")
    console.print(f"  Balance:    [cyan]${risk['balance']:,.2f} USDT[/cyan]")
    console.print(f"  Modelo IA:  [cyan]{status['model_version'] or 'sin modelo'}[/cyan]")
    console.print(f"  Simbolos:   [cyan]{status['pipeline']['top_symbols']}[/cyan]")
    console.print(f"  Modo:       [cyan]{settings.trading_mode.value.upper()}[/cyan]")

    # 4. Servidor API (FastAPI + dashboard)
    from api.server import start_api_server
    console.print(
        f"  Dashboard:  [cyan]http://{settings.api_host}:{settings.api_port}[/cyan]  "
        f"[dim](abre dashboard/index.html en tu navegador)[/dim]"
    )

    console.print("\n[bold green]Bot activo — todas las fases corriendo[/bold green]")
    console.print("[dim]Presiona Ctrl+C para detener[/dim]\n")

    # Correr bot + API en paralelo
    await asyncio.gather(
        trading_loop.run_forever(),
        start_api_server(settings, trading_loop, db),
    )


if __name__ == "__main__":
    main()
