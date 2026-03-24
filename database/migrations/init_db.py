"""
=============================================================
database/migrations/init_db.py
Script para inicializar y migrar la base de datos.
Ejecutar para crear las tablas en PostgreSQL.

Uso:
    python database/migrations/init_db.py

=============================================================
"""

import sys
from pathlib import Path

# Añadir raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from sqlalchemy import create_engine, text, inspect

from config.settings import get_settings
from database.models import Base

console = Console()


def init_database():
    """Crea todas las tablas en la base de datos PostgreSQL."""
    settings = get_settings()

    console.print("\n[bold cyan]Inicializando base de datos PostgreSQL...[/bold cyan]\n")
    console.print(f"  Host: {settings.db_host}:{settings.db_port}")
    console.print(f"  BD:   {settings.db_name}")
    console.print(f"  User: {settings.db_user}\n")

    try:
        engine = create_engine(settings.database_url, echo=False)

        # Verificar conexión
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        console.print("  [green]✓ Conexión a PostgreSQL exitosa[/green]")

        # Crear todas las tablas
        Base.metadata.create_all(engine)
        console.print("  [green]✓ Tablas creadas/verificadas:[/green]")

        # Listar tablas creadas
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        for table in sorted(tables):
            console.print(f"    · {table}")

        console.print(
            f"\n  [bold green]✓ Base de datos inicializada correctamente "
            f"({len(tables)} tablas)[/bold green]\n"
        )

    except Exception as e:
        console.print(f"\n  [bold red]✗ Error: {e}[/bold red]")
        console.print(
            "\n  [dim]Asegúrate de que:\n"
            "  1. PostgreSQL está corriendo\n"
            "  2. La base de datos existe: CREATE DATABASE trading_bot;\n"
            "  3. Las credenciales en .env son correctas[/dim]\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    init_database()
