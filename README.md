# 🤖 Trading Bot IA — Binance Futures USDT-M

Bot de trading algorítmico con Inteligencia Artificial para operar en Binance Futuros.
Implementado en Python 3.11+ con PostgreSQL, FastAPI y dashboard React.

---

## Estructura del Proyecto

```
trading_bot/
├── config/
│   └── settings.py          # Configuración centralizada (Pydantic)
├── core/
│   └── binance_client.py    # Cliente Binance Futuros (testnet/live/dry_run)
├── data/                    # Fase 2: Recolección de datos
├── analysis/                # Fase 2: Indicadores técnicos
├── ai/                      # Fase 3: Motor ML/IA
├── strategy/                # Fase 4: Señales y estrategia
├── risk/                    # Fase 4: Gestión de riesgo
├── execution/               # Fase 5: Motor de ejecución de órdenes
├── api/                     # Fase 6: FastAPI REST
├── database/
│   ├── models.py            # Modelos SQLAlchemy (PostgreSQL)
│   ├── connection.py        # Gestor de conexiones
│   └── migrations/
│       └── init_db.py       # Inicialización de tablas
├── utils/
│   └── logger.py            # Sistema de logging (Loguru)
├── tests/
│   └── test_phase1.py       # Tests de Fase 1
├── logs/                    # Archivos de log (auto-generado)
├── models/saved/            # Modelos ML guardados (auto-generado)
├── .env.example             # Plantilla de variables de entorno
├── .env                     # Variables de entorno (NO subir a git)
├── requirements.txt         # Dependencias Python
├── setup.py                 # Script de configuración inicial
└── main.py                  # Punto de entrada principal
```

---

## Instalación Rápida (Windows)

### 1. Prerrequisitos

- **Python 3.11+**: https://python.org/downloads
- **PostgreSQL**: https://postgresql.org/download/windows
- **Git** (opcional): https://git-scm.com

### 2. Configurar el entorno

```bash
# Clonar o descomprimir el proyecto
cd trading_bot

# Ejecutar setup automático
python setup.py
```

### 3. Configurar PostgreSQL

En pgAdmin o psql:
```sql
CREATE DATABASE trading_bot;
```

### 4. Configurar variables de entorno

Edita el archivo `.env` creado por setup.py:

```env
TRADING_MODE=dry_run          # testnet | dry_run | live
BINANCE_TESTNET_API_KEY=...    # tu API key de testnet
BINANCE_TESTNET_API_SECRET=... # tu API secret de testnet
DB_PASSWORD=tu_password_postgres
INITIAL_CAPITAL=1000.0
```

### 5. Inicializar la base de datos

```bash
python database/migrations/init_db.py
```

### 6. Verificar todo

```bash
python main.py --check
```

### 7. Ejecutar el bot

```bash
# Modo simulación (sin dinero real, sin API)
python main.py --mode dry_run

# Modo testnet (API de testnet Binance)
python main.py --mode testnet

# Modo live (¡DINERO REAL - usar con precaución!)
python main.py --mode live
```

---

## Fases de Desarrollo

| Fase | Módulos | Estado |
|------|---------|--------|
| **1** | Config, DB, Binance Client, Logging | ✅ Completada |
| **2** | Data Collector, WebSocket, Indicadores TA | 🔜 Siguiente |
| **3** | Feature Engineering, ML Models, Retraining | 🔜 |
| **4** | Strategy Engine, Risk Management | 🔜 |
| **5** | Order Executor, Position Manager | 🔜 |
| **6** | FastAPI REST, React Dashboard | 🔜 |

---

## Modos de Operación

| Modo | Descripción | Dinero real |
|------|-------------|-------------|
| `dry_run` | Simula todo localmente, sin conexión a Binance | ❌ No |
| `testnet` | Usa la API real de testnet de Binance | ❌ No |
| `live` | Opera en producción real | ✅ Sí |

---

## Parámetros de Riesgo

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `MAX_RISK_PER_TRADE` | 2% | Riesgo máximo por operación |
| `MAX_DRAWDOWN` | 15% | Drawdown máximo antes de pausar |
| `MAX_OPEN_POSITIONS` | 3 | Posiciones simultáneas |
| `MAX_LEVERAGE` | 5x | Apalancamiento máximo |

---

## Tests

```bash
# Tests básicos de Fase 1
python tests/test_phase1.py

# Con pytest
pytest tests/test_phase1.py -v
```

---

## Seguridad

- **NUNCA** subas el archivo `.env` a git
- Usa siempre `dry_run` o `testnet` para probar
- Empieza con capital pequeño al pasar a `live`
- Revisa los logs en `logs/trading_bot.log`
