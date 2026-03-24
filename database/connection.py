"""
=============================================================
database/connection.py
Gestor de conexión a PostgreSQL con SQLAlchemy 2.0.
Soporta sesiones síncronas y asíncronas.
=============================================================
"""

from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from database.models import Base
from utils.logger import get_logger

log = get_logger(__name__)


class DatabaseManager:
    """
    Gestor centralizado de conexiones a PostgreSQL.
    Soporta tanto operaciones síncronas como asíncronas.

    Uso:
        db = DatabaseManager(settings)
        await db.initialize()

        # Sesión asíncrona
        async with db.async_session() as session:
            result = await session.execute(select(Trade))

        # Sesión síncrona
        with db.sync_session() as session:
            trades = session.query(Trade).all()
    """

    def __init__(self, database_url: str, async_database_url: str):
        self.database_url = database_url
        self.async_database_url = async_database_url
        self._sync_engine = None
        self._async_engine = None
        self._sync_session_factory = None
        self._async_session_factory = None

    def _create_sync_engine(self):
        """Crea el motor síncrono con pool optimizado."""
        engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,      # Verifica conexiones antes de usarlas
            pool_recycle=3600,       # Recicla conexiones cada hora
            echo=False,
        )

        # Listener para registrar errores de conexión
        @event.listens_for(engine, "connect")
        def on_connect(dbapi_con, _):
            log.debug("Nueva conexión PostgreSQL establecida")

        return engine

    def _create_async_engine(self):
        """Crea el motor asíncrono para operaciones de alta frecuencia."""
        return create_async_engine(
            self.async_database_url,
            pool_size=20,
            max_overflow=40,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
        )

    def initialize(self) -> None:
        """
        Inicializa los motores y factories de sesión.
        Crea las tablas si no existen.
        """
        log.info("Inicializando conexión a PostgreSQL...")

        self._sync_engine = self._create_sync_engine()
        self._async_engine = self._create_async_engine()

        self._sync_session_factory = sessionmaker(
            bind=self._sync_engine,
            expire_on_commit=False,
            autoflush=True,
        )

        self._async_session_factory = async_sessionmaker(
            bind=self._async_engine,
            expire_on_commit=False,
            autoflush=True,
            class_=AsyncSession,
        )

        # Crear tablas si no existen
        self._create_tables()
        log.info("✓ Base de datos inicializada correctamente")

    def _create_tables(self) -> None:
        """Crea todas las tablas definidas en los modelos."""
        Base.metadata.create_all(bind=self._sync_engine)
        log.info("✓ Tablas verificadas/creadas en PostgreSQL")

    @contextmanager
    def sync_session(self) -> Generator[Session, None, None]:
        """
        Context manager para sesiones síncronas.

        Uso:
            with db.sync_session() as session:
                session.add(new_trade)
                session.commit()
        """
        session = self._sync_session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            log.error(f"Error en sesión DB, rollback ejecutado: {e}")
            raise
        finally:
            session.close()

    @asynccontextmanager
    async def async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Context manager para sesiones asíncronas.

        Uso:
            async with db.async_session() as session:
                result = await session.execute(select(Trade))
                trades = result.scalars().all()
        """
        async with self._async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                log.error(f"Error en sesión async DB, rollback ejecutado: {e}")
                raise

    async def check_connection(self) -> bool:
        """Verifica que la conexión a la base de datos sea exitosa."""
        try:
            async with self._async_session_factory() as session:
                await session.execute(text("SELECT 1"))
            log.info("✓ Conexión a PostgreSQL verificada")
            return True
        except Exception as e:
            log.error(f"✗ Error de conexión a PostgreSQL: {e}")
            return False

    async def close(self) -> None:
        """Cierra todas las conexiones de forma limpia."""
        if self._async_engine:
            await self._async_engine.dispose()
        if self._sync_engine:
            self._sync_engine.dispose()
        log.info("Conexiones a base de datos cerradas")


# ── Instancia global (se inicializa en main.py) ───────────────────────────────

_db_manager: DatabaseManager | None = None


def get_db_manager() -> DatabaseManager:
    """Retorna la instancia global del gestor de base de datos."""
    if _db_manager is None:
        raise RuntimeError(
            "DatabaseManager no inicializado. "
            "Llama a init_database() primero."
        )
    return _db_manager


def init_database(database_url: str, async_database_url: str) -> DatabaseManager:
    """
    Inicializa la base de datos global.
    Debe llamarse una sola vez al arrancar la aplicación.
    """
    global _db_manager
    _db_manager = DatabaseManager(database_url, async_database_url)
    _db_manager.initialize()
    return _db_manager
