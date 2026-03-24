# fix_encoding.py
# Ejecuta este script desde el directorio raiz del proyecto:
#   python fix_encoding.py

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

modules = [
    "config",
    "core",
    "data",
    "analysis",
    "ai",
    "strategy",
    "risk",
    "execution",
    "api",
    "database",
    "utils",
    "tests",
    "database/migrations",
]

print("Corrigiendo archivos __init__.py...\n")

for module in modules:
    module_path = BASE_DIR / module
    module_path.mkdir(parents=True, exist_ok=True)

    init_file = module_path / "__init__.py"

    # Contenido sin caracteres especiales
    content = "# {}\n".format(module.replace("/", "."))

    # Escribir explicitamente con UTF-8
    with open(init_file, "w", encoding="utf-8") as f:
        f.write(content)

    print("  OK: {}".format(init_file))

print("\nListo. Ahora vuelve a ejecutar:")
print("  python database/migrations/init_db.py")
