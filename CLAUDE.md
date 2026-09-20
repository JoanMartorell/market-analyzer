# CLAUDE.md

Guía para trabajar en este repositorio con Claude Code. Lo que no esté aquí se deduce
del código: los docstrings de módulo explican el porqué de cada pieza, léelos antes de
tocarla.

## Qué es

Screener de bolsa multi-región: pipeline diario de 12 pasos por región (calendario →
universo → ingesta → calidad → operaciones corporativas → indicadores → panel →
screener → noticias → dedup y sentimiento → una llamada al modelo → persistencia →
entrega → conciliación al día siguiente). Ver `README.md` para el cuadro completo.

## Normas que no se negocian

- **Todo en castellano**: docstrings, comentarios, mensajes de log, resúmenes, textos de
  la CLI, tests (nombres de test en inglés está bien) y las respuestas al usuario.
- **Nunca `eval()` ni `exec()`.** Las reglas se parsean a AST con lista blanca en
  `src/analyzer/rules/parser.py`. Ruff con reglas `S` (bandit) lo vigila.
- **Se falla en cerrado.** Ante datos incompletos o inconsistentes, el paso lanza
  `StepError` y el pipeline se detiene sin señales. No se rellena, no se interpola, no
  se sigue "a medias". El silencio no es éxito: se notifica el fallo.
- **Determinismo y trazabilidad.** Cada señal guarda hash de regla y de panel. Lo que
  decide el modelo va aparte y se mide después (`reconcile`).
- **Coste de modelo mínimo.** Una llamada por región y día (`max_calls_per_cycle: 1`),
  respuesta guardada y reutilizada si la entrada no cambió. No añadir llamadas.
- **Calendarios con exchange_calendars.** Nunca `weekday() < 5`.
- **Secretos solo en `.env`.** Nunca en YAML, nunca en el log (ni en mensajes de error
  ni en URLs). Los clientes HTTP de httpx tienen el log a WARNING por eso.
- **Cupos de proveedores gratuitos** se cuentan en base de datos y *antes* de llamar
  (`ApiCallStore`), nunca en una variable del proceso.
- **Commit y push solo cuando el usuario lo pida.**

## Comandos

```bash
uv sync --all-extras                     # instalación
uv run analyzer config-check             # valida YAML + .env
uv run analyzer run -r americas -d 2026-09-18
uv run analyzer serve --max-runs 1
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy src tests
uv run pytest -q
```

Las tres puertas (ruff + format, mypy strict, pytest) deben estar en verde antes de dar
algo por terminado. Ejecuta las tres, no solo los tests del módulo tocado. Ruff:
línea de 100, comillas dobles, isort. mypy: `strict`, `warn_unreachable`, plugin
pydantic, pandas-stubs.

Ejecutar un ciclo real (`analyzer run`) gasta cupo de proveedores y dinero de modelo:
hazlo solo cuando haga falta verificar y dilo en la respuesta.

## Estructura

```
src/core/            entrada y orquestación
  cli.py             solo parsea argumentos y traduce a texto y código de salida
  app.py             run_region, reconcile_region, notify, backfill, telegram_chats
  config/            schema.py (pydantic, StrictModel), loader.py (validación cruzada), env.py
  scheduler.py       analyzer serve
  digest.py          el mensaje del día (COMPRAR / VENDER / VIGILAR)
src/analyzer/
  engine.py          Step, StepContext, StepOutcome, StepError, PipelineResult
  steps/__init__.py  ORDEN de los pasos; vive solo aquí
  steps/<paso>/      step.py (adaptador al motor) + service.py (lógica pura, testeable)
  storage/           una clase Table por tabla DuckDB, DDL + migraciones en COLUMNS_ADDED
  rules/             parser AST + funciones permitidas
  universe/          constituyentes point-in-time: fuentes, enriquecedores, build
  sessions.py        qué sesión procesar (as_of nunca es "hoy")
  yahoo.py           sufijos de Yahoo por MIC (compartido por proveedor y universo)
src/delivery/        consola y Telegram; Message + Dispatcher
config/              settings.yaml, sources.yaml, regions/*.yaml, rules/*.yaml
prompts/             analyst_system.md
tests/               espejo de src; conftest con cfg real sin credenciales
```

## Cómo se añade una pieza

- **Un paso nuevo**: carpeta en `steps/`, `service.py` con funciones puras que reciben
  DataFrames y almacenes, `step.py` que lee `ctx.data`, llama al servicio y deja un
  informe en `ctx.data["<paso>"]`; registrarlo en `steps/__init__.py`. Los informes son
  dataclasses congeladas con `summary()` en castellano: ese texto es lo que se ve en el
  log y en la tabla de pasos.
- **Una tabla nueva**: subclase de `Table` en `storage/`, `NAME`, `DDL` con `{table}`,
  columnas añadidas después en `COLUMNS_ADDED` (migración con `ADD COLUMN IF NOT
  EXISTS`). Escritura por `_insert_or_replace`; lectura por `_select` / `_key_filter`.
  Exportar en `storage/__init__.py`.
- **Un proveedor de precios**: módulo en `steps/ingest/providers/` que cumpla
  `PriceProvider` (`fetch(keys, start, end)` → columnas `FETCH_COLUMNS`), fábrica en
  `providers/__init__.py` (recibe `cfg` y `con`; los ignora si no los necesita), entrada
  en `sources.yaml` con `api_key_env` y `daily_call_limit` si tiene cupo.
- **Un universo**: `UniverseSpec` con fuente y enriquecedores en `universe/sources/`,
  registrado en `universe/registry.py`. `history=False` para instantáneas que se
  reconstruyen a diario.
- **Un parámetro de configuración**: campo en `schema.py` (modelos `StrictModel`,
  `extra="forbid"`) con valor por defecto sensato, valor y comentario en el YAML. Lo
  que es secreto va a `env.py` y `.env.example`.

## Convenciones de código

- Tipado completo; `from __future__ import annotations`; dataclasses congeladas para
  informes; `Protocol` para contratos inyectables (proveedores, buscadores, relojes).
- pandas 3 y numpy 2: sin `inplace`, sin `.at` (ruff PD008), fechas con
  `pd.to_datetime`, NaN como ausencia. DuckDB convierte NaN de un frame registrado en
  NULL.
- Dependencias inyectables para poder probar sin red: descargadores, clientes httpx
  (`httpx.MockTransport`), reloj y `sleep` del scheduler.
- Logs con structlog: evento en `modulo.accion` (`ingest.rescue`, `eodhd.budget_exhausted`)
  y campos, no frases. Nunca un secreto ni una URL con token.
- Los mensajes al usuario (digest, CLI, resúmenes) son cortos y en castellano; el
  detalle largo va al log o a la base de datos.

## Tests

- `tests/conftest.py`: `empty_env` (sin claves, sin `.env`) y `cfg` (la configuración
  real del repo). Los tests que necesitan base de datos usan `duckdb.connect(":memory:")`
  o `replace(cfg, env=Env(_env_file=None, ma_data_dir=tmp_path))`.
- Un test por comportamiento, con nombre que lo describa; se comprueban textos de
  resumen porque son parte del contrato con el usuario.
- Nada de red: si un test necesita HTTP, `httpx.MockTransport`; si necesita precios, un
  `FakeProvider`.
- Fecha de referencia habitual en tests: septiembre de 2026.

## Configuración y entorno

- `config-check` es la primera comprobación tras tocar un YAML: valida esquema,
  referencias cruzadas (proveedores, fuentes, reglas, calendarios) y qué variables de
  entorno faltan para lo que está activo.
- Horas en `schedule.run_at` entre comillas ("06:30"); sin comillas YAML lo lee como
  entero.
- Valores marcados `PROVISIONAL` en los YAML son ajustes de prueba con su valor de
  producción al lado; no se dejan olvidados.
- Universos de instantánea (Europa, APAC): reconstruirlos borra y rehace la composición;
  la memoria de tickers `yahoo_symbols.json` de cada carpeta se conserva y puede
  corregirse a mano.

## Estado y límites conocidos

- Américas es la región de referencia. Europa y APAC están activas; Europa depende de
  que Yahoo publique a tiempo (EODHD gratuito solo rescata 20 valores/día).
- `reevaluation_gate` (Europa) existe en el esquema pero ningún paso lo ejecuta.
- FMP y FRED están declarados pero sin uso.
- No hay conversión de divisas: cada señal va en su moneda local.

## Entorno de trabajo

Windows 11 con Git Bash. Los heredocs largos en Bash fallan a veces con "unexpected
EOF"; para ficheros nuevos o ediciones multilínea usa las herramientas de escritura o
un script Python en el scratchpad. Rutas con `P:/market-analyzer`.
