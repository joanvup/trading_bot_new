"""
=============================================================
utils/logger.py
Sistema de logging profesional usando Loguru.
Proporciona logs estructurados con rotación automática,
niveles diferenciados por módulo y salida a consola y archivo.
=============================================================
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_rotation: str = "10 MB",
    log_retention: str = "30 days",
    enable_console: bool = True,
) -> None:
    """
    Configura el sistema de logging global.

    Args:
        log_level:      Nivel de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file:       Ruta del archivo de log (None = solo consola)
        log_rotation:   Tamaño o tiempo para rotar el archivo
        log_retention:  Tiempo de retención de logs viejos
        enable_console: Si True, también imprime en consola
    """
    # Eliminar handlers por defecto de loguru
    logger.remove()

    # ── Formato para consola (con colores) ────────────────────────────────────
    console_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # ── Formato para archivo (más detallado, sin colores) ─────────────────────
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )

    if enable_console:
        logger.add(
            sys.stdout,
            format=console_format,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            str(log_path),
            format=file_format,
            level=log_level,
            rotation=log_rotation,
            retention=log_retention,
            compression="zip",
            backtrace=True,
            diagnose=True,
            encoding="utf-8",
        )

        # Archivo separado solo para errores críticos
        error_log = log_path.parent / "errors.log"
        logger.add(
            str(error_log),
            format=file_format,
            level="ERROR",
            rotation="5 MB",
            retention="60 days",
            compression="zip",
            encoding="utf-8",
        )


def get_logger(name: str):
    """
    Retorna un logger con contexto de módulo.

    Uso:
        from utils.logger import get_logger
        log = get_logger(__name__)
        log.info("Bot iniciado")
        log.warning("Señal débil detectada")
        log.error("Error de conexión", exc_info=True)

    Args:
        name: Nombre del módulo (usa __name__)

    Returns:
        Logger de loguru con binding del nombre del módulo
    """
    return logger.bind(module=name)


# ── Decorador para logging de funciones ───────────────────────────────────────

def log_execution(func):
    """
    Decorador que registra la ejecución de funciones con tiempo.

    Uso:
        @log_execution
        def calcular_indicadores(df):
            ...
    """
    import time
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug(f"{func.__qualname__} completado en {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"{func.__qualname__} falló en {elapsed:.3f}s: {e}")
            raise

    return wrapper


# ── Inicializar logging con configuración por defecto ─────────────────────────
# Se sobreescribe desde main.py con los valores del .env

setup_logging()
