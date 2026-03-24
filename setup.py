"""
=============================================================
setup.py
Script de configuración inicial del entorno.
Ejecutar una sola vez para preparar el proyecto.

Uso:
    python setup.py

=============================================================
"""

import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()
BASE_DIR = Path(__file__).parent


def run_command(cmd: list[str], description: str) -> bool:
    """Ejecuta un comando del sistema y reporta el resultado."""
    console.print(f"  {description}...", end=" ")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
        )
        if result.returncode == 0:
            console.print("[green]✓[/green]")
            return True
        else:
            console.print(f"[red]✗[/red]")
            if result.stderr:
                console.print(f"    [dim red]{result.stderr[:200]}[/dim red]")
            return False
    except FileNotFoundError:
        console.print(f"[red]✗ Comando no encontrado[/red]")
        return False


def create_env_file() -> None:
    """Crea el archivo .env si no existe."""
    env_file = BASE_DIR / ".env"
    example_file = BASE_DIR / ".env.example"

    if not env_file.exists():
        if example_file.exists():
            import shutil
            shutil.copy(example_file, env_file)
            console.print(
                "  [yellow]Archivo .env creado desde .env.example[/yellow]\n"
                "  [bold yellow]⚠️  Edita .env con tus credenciales antes de ejecutar el bot[/bold yellow]"
            )
        else:
            console.print("  [red]No se encontró .env.example[/red]")
    else:
        console.print("  [green]✓ .env ya existe[/green]")


def create_directories() -> None:
    """Crea los directorios necesarios."""
    dirs = [
        "logs",
        "models/saved",
        "data/raw",
        "data/processed",
    ]
    for d in dirs:
        path = BASE_DIR / d
        path.mkdir(parents=True, exist_ok=True)
    console.print("  [green]✓ Directorios creados[/green]")


def create_gitignore() -> None:
    """Crea o actualiza .gitignore."""
    gitignore_content = """# Trading Bot - .gitignore

# Credenciales (NUNCA subir)
.env
*.pem
*.key

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.eggs/
venv/
env/
.venv/

# Datos y modelos
data/raw/
data/processed/
models/saved/
logs/

# IDE
.vscode/
.idea/
*.suo
*.user

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/
"""
    gitignore_path = BASE_DIR / ".gitignore"
    gitignore_path.write_text(gitignore_content, encoding="utf-8")
    console.print("  [green]✓ .gitignore creado[/green]")


def create_init_files() -> None:
    """Crea archivos __init__.py en todos los módulos."""
    modules = [
        "config", "core", "data", "analysis", "ai",
        "strategy", "risk", "execution", "api",
        "database", "utils", "tests",
    ]
    for module in modules:
        init_file = BASE_DIR / module / "__init__.py"
        if not init_file.exists():
            init_file.write_text(f'"""Módulo {module} del Trading Bot."""\n')

    console.print("  [green]✓ Archivos __init__.py creados[/green]")


def check_python_version() -> bool:
    """Verifica que Python sea 3.11+."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        console.print(
            f"  [green]✓ Python {version.major}.{version.minor}.{version.micro}[/green]"
        )
        return True
    else:
        console.print(
            f"  [red]✗ Python {version.major}.{version.minor} detectado. "
            f"Se requiere 3.11+[/red]"
        )
        return False


def main():
    console.print(
        Panel.fit(
            "[bold cyan]🔧 Setup inicial - Trading Bot IA[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()

    # 1. Verificar Python
    console.print("[bold]1. Verificando Python...[/bold]")
    if not check_python_version():
        console.print(
            "\n[red]Instala Python 3.11+ desde https://python.org[/red]"
        )
        sys.exit(1)

    # 2. Crear estructura
    console.print("\n[bold]2. Creando estructura de directorios...[/bold]")
    create_directories()

    # 3. Crear __init__.py
    console.print("\n[bold]3. Inicializando módulos Python...[/bold]")
    create_init_files()

    # 4. Crear .env
    console.print("\n[bold]4. Configurando variables de entorno...[/bold]")
    create_env_file()

    # 5. Crear .gitignore
    console.print("\n[bold]5. Configurando .gitignore...[/bold]")
    create_gitignore()

    # 6. Instalar dependencias
    console.print("\n[bold]6. Instalando dependencias...[/bold]")
    console.print("  [dim](Esto puede tardar varios minutos)[/dim]")

    pip_ok = run_command(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        "Instalando paquetes"
    )
    if not pip_ok:
        console.print(
            "\n[yellow]Algunos paquetes pueden requerir instalación manual.[/yellow]\n"
            "[dim]Ejecuta: pip install -r requirements.txt[/dim]"
        )

    # 7. Resumen final
    console.print()
    console.print(
        Panel(
            "[bold green]✓ Setup completado[/bold green]\n\n"
            "Próximos pasos:\n"
            "  1. Edita [bold].env[/bold] con tus credenciales de Binance y PostgreSQL\n"
            "  2. Crea la base de datos: [cyan]CREATE DATABASE trading_bot;[/cyan]\n"
            "  3. Verifica conexiones: [cyan]python main.py --check[/cyan]\n"
            "  4. Ejecuta en dry run: [cyan]python main.py --mode dry_run[/cyan]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
