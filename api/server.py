"""
=============================================================
api/server.py
Lanzador del servidor FastAPI integrado con el TradingLoop.

Puede correr en dos modos:
  1. Junto al bot (mismo proceso, hilo separado)
  2. Standalone (solo la API, para desarrollo del dashboard)

Uso desde main.py:
    from api.server import start_api_server
    asyncio.create_task(start_api_server(settings, trading_loop, db))

Uso standalone para desarrollo:
    python api/server.py
=============================================================
"""

import asyncio
import sys
from pathlib import Path

import uvicorn

from api.main_api import app, set_trading_loop
from config.settings import Settings
from database.connection import DatabaseManager
from utils.logger import get_logger

log = get_logger(__name__)


async def start_api_server(
    settings:     Settings,
    trading_loop,
    db:           DatabaseManager,
) -> None:
    """
    Arranca el servidor FastAPI como tarea asyncio.
    Se llama desde el TradingLoop para correr en paralelo.
    """
    # Inyectar referencias globales
    set_trading_loop(trading_loop, db)

    config = uvicorn.Config(
        app=app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="warning",   # Silenciar logs de uvicorn (usamos loguru)
        access_log=False,
    )
    server = uvicorn.Server(config)

    log.info(f"API REST iniciada en http://{settings.api_host}:{settings.api_port}")
    log.info(f"Docs: http://{settings.api_host}:{settings.api_port}/docs")

    await server.serve()


# ── Standalone para desarrollo ────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Modo standalone: arranca solo la API sin el bot activo.
    Util para desarrollar el frontend sin correr el bot completo.

    Uso:
        python api/server.py
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    from config.settings import get_settings
    from database.connection import init_database

    settings = get_settings()
    db       = init_database(settings.database_url, settings.async_database_url)

    # Sin TradingLoop — la API devuelve datos de BD solamente
    set_trading_loop(None, db)

    print(f"\n API standalone en http://{settings.api_host}:{settings.api_port}")
    print(f" Docs en http://{settings.api_host}:{settings.api_port}/docs\n")

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="info",
    )
