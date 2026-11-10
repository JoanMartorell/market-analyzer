# market-analyzer

Screener de bolsa multi-región con reglas deterministas y una capa de modelo de lenguaje
mínima. Cada día, tras el cierre de cada región, descarga los precios, valida el feed,
calcula indicadores, aplica reglas YAML sobre un panel point-in-time, lee las noticias de
los candidatos, pide **una sola** opinión al modelo y manda al móvil una lista corta de
qué comprar, qué vender y qué vigilar, con un motivo de una línea por valor. Al día
siguiente concilia cada señal con la apertura real.

Principios que gobiernan todo el código:

- **Se falla en cerrado.** Si algo no cuadra (feed incompleto, salto de precio sin
  operación corporativa, vela del día ausente), no hay señales ese día y se avisa.
  El silencio nunca es éxito.
- **Determinismo primero.** Indicadores, reglas y panel son reproducibles: cada señal
  guarda el hash de la regla y del panel del que salió. El modelo solo opina sobre
  candidatos que ya pasaron las reglas.
- **Coste mínimo de modelo.** Una llamada por región y día, con caché de prompt y, si
  el lote tarda, llamada directa. Un ciclo cuesta entre 0.01 y 0.10 EUR.
- **Nunca `eval()`.** Las reglas se parsean a AST y solo se evalúan nodos de una lista
  blanca.

## Requisitos

- Python 3.12 o 3.13 y [uv](https://docs.astral.sh/uv/).
- Cuentas gratuitas: Anthropic (modelo), Finnhub (noticias), FRED (macro), EODHD
  (rescate de precios), un bot de Telegram. yfinance no necesita clave.

## Instalación

```bash
uv sync                       # núcleo
uv sync --extra nlp           # + embeddings y FinBERT (descarga torch para CPU)
uv sync --all-extras          # + Prefect
cp .env.example .env          # y rellena las claves
uv run analyzer config-check  # valida YAML y .env, muestra qué queda activo
```

Sin el extra `nlp`, la deduplicación por embeddings y el sentimiento se desactivan y
el pipeline opera solo con reglas técnicas.

## Uso

```bash
uv run analyzer run -r americas                 # ciclo de la última sesión cerrada
uv run analyzer run -r europe -d 2026-09-18     # ciclo de una sesión concreta
uv run analyzer reconcile -r americas           # paso 12 suelto
uv run analyzer serve                           # residente: cada región a su hora
uv run analyzer universe build -r europe        # reconstruye los constituyentes
uv run analyzer universe show -r apac -d 2026-09-18
uv run analyzer ingest backfill -r americas -y 3
uv run analyzer ingest status -r europe
uv run analyzer telegram chats                  # descubre el chat id del bot
```

Códigos de salida: 0 correcto, 1 configuración inválida o pipeline fallido, 2 faltan
credenciales.

Para Telegram: crea el bot con BotFather, pon `TELEGRAM_BOT_TOKEN` en `.env`, escribe
"Start" al bot desde tu cuenta, ejecuta `analyzer telegram chats` y copia el id a
`TELEGRAM_CHAT_ID`. Después pon `telegram` en `delivery.channels` de `settings.yaml`.

## El ciclo diario

Doce pasos, en este orden y en ningún otro sitio (`src/analyzer/steps/__init__.py`):

| # | Paso | Qué hace |
|---|---|---|
| 0 | `calendar_gate` | ¿Hubo sesión? Lo decide exchange_calendars, nunca `weekday() < 5`. Sin sesión, fin sin error. |
| 0b | `universe` | Constituyentes vigentes ese día (point-in-time), solo de mercados con sesión. |
| 1 | `ingest` | Velas EOD a `staging`. yfinance por lotes; lo que deje sin la vela del día lo rescata `prices_fallback` (EODHD, 20 llamadas/día). |
| 2 | `quality` | Cobertura ≥ 98 %, fecha del feed, saltos, volumen cero. Si pasa, promueve a `prod`. |
| 3 | `corporate_actions` | Splits y dividendos; marca qué histórico hay que recalcular. |
| 4 | `indicators` | Incremental sobre las últimas 250 sesiones (RSI, SMA, MACD...) y patrones chartistas. |
| 5 | `snapshot` | Panel del día con hash sha256; lo que verá el screener. |
| 6 | `screener` | Reglas YAML sobre el panel: candidatos con puntuación y condiciones cumplidas. |
| 7 | `news` | Solo para candidatos: RSS de Yahoo y CNBC, Finnhub. |
| 8 | `dedup_sentiment` | Cascada MinHash → embeddings → DBSCAN; FinBERT sobre el representante. |
| 9 | `llm` | Una llamada con salida estructurada: veredicto, confianza, motivo en una línea. |
| 10 | `persist` | Señal con hash de regla y panel, veredicto y sesión de ejecución. |
| 11 | entrega | Resumen del día y un mensaje con gráfico por cada operación, por consola o Telegram (lo hace `core`, no un paso). |
| 12 | `reconcile` | Al día siguiente: apertura real frente al cierre asumido; `conciliada`, `desviada` o `sin_precio`. |

`as_of` nunca es "hoy": es la última sesión cuyo cierre es anterior al lanzamiento en
todos los mercados de la región. Las señales del día D se ejecutan en la apertura de la
siguiente sesión de su bolsa.

## Patrones chartistas

Además de la regla de indicadores (`oversold_uptrend`), cada región evalúa ocho figuras:

| | COMPRAR | VENDER EN CORTO |
|---|---|---|
| Giro | doble suelo, hombro-cabeza-hombro invertido | doble techo, hombro-cabeza-hombro |
| Continuación | triángulo ascendente, bandera alcista | triángulo descendente, bandera bajista (\*) |

(\*) La bandera bajista se detecta y se mide, pero ninguna región la usa como regla: en
el histórico disponible solo el 22 % llegó al objetivo.

- El detector (`src/analyzer/patterns/`) saca los pivotes del precio con un zigzag en
  ATR y da la figura por completada el día en que el cierre rompe su línea. Lo que se
  decide un día solo usa datos hasta ese día.
- Las banderas no salen de los pivotes (su pausa es demasiado pequeña): se miran en las
  velas anteriores a la que rompe. Mástil de al menos 6 ATR en 15 sesiones, pausa de 4 a
  20 que devuelve como mucho la mitad, y cierre por encima del máximo del mástil.
- El resultado entra en la tabla de indicadores: `pat_<patrón>` vale 1 el día de la
  ruptura, y `pat_entry`, `pat_stop` y `pat_target` son los niveles de ese día. Las
  reglas (`config/rules/double_bottom.yaml`...) los usan como cualquier otra variable.
- La señal guarda el patrón, el stop y el objetivo. El resumen los enseña, y cada
  operación recomendada llega en un mensaje propio con el gráfico de la figura
  (`delivery.notify_operations`).
- `analyzer patterns stats -r americas` pasa el detector por el histórico guardado y
  cuenta cuántos patrones llegaron al objetivo antes que al stop. No descarga nada ni
  llama al modelo; es una medida de frecuencia y desenlace, no un backtest.

Los umbrales del detector están en `PatternParams` (`patterns/model.py`), en código y no
en YAML: las columnas se guardan día a día y un umbral cambiado a mano dejaría un
histórico calculado con dos criterios.

## Regiones

| Región | Mercados | Universo | Hora (Madrid) | Precios |
|---|---|---|---|---|
| americas | NYSE, Nasdaq | S&P 500 con histórico desde 1996 | 06:30 | yfinance |
| europe | 16 bolsas (Xetra, Euronext, LSE, SIX, nórdicas...) | STOXX 600 | 19:30 | yfinance + rescate EODHD |
| apac (desactivada) | Tokio, Hong Kong, Sídney, Seúl | Nikkei 225 + HSI + ASX 200 + KOSPI 200 | 10:30 | yfinance + rescate EODHD |
| latam | B3 (São Paulo), Bolsa Mexicana | Ibovespa + S&P/BMV IPC | 07:00 | yfinance + rescate EODHD |

Los universos de Europa, APAC y Latinoamérica son instantáneas de Wikipedia que se reconstruyen a
diario y acumulan su histórico desde el primer build. Las tablas traen códigos Reuters
y empresas ya absorbidas, así que un enriquecedor corrige cada ticker contra Yahoo y
saca de la composición a las que ya no cotizan. Las resoluciones quedan en
`data/universe/<región>/yahoo_symbols.json`, editable a mano.

Dos regiones activas no pueden compartir bolsa: la clave de una señal es (ticker, mic,
fecha, regla), sin región, y una pisaría las señales de la otra. El cargador lo rechaza.

Límites conocidos:

- Yahoo publica a veces con retraso los cierres de Europa continental. Con el plan
  gratuito de EODHD (20 llamadas al día, sin descargas por bolsa) el rescate cubre 20
  valores; si faltan más, Europa no emite señales ese día y avisa.
- La puerta de reevaluación europea (`reevaluation_gate`, futuros USA a las 08:45)
  está en el esquema de configuración pero ningún paso la ejecuta todavía.
- FMP y FRED están declarados como proveedores pero ningún paso los usa aún.

## Configuración

Todo lo que cambia el comportamiento va versionado en `config/`; los secretos, en
`.env`.

- `settings.yaml`: umbrales de calidad, indicadores, noticias, modelo y presupuesto,
  conciliación, entrega, planificador.
- `sources.yaml`: proveedores de datos y fuentes de noticias, con la variable de
  entorno que necesita cada uno y su cupo diario.
- `regions/<id>.yaml`: mercados, horario, universo, proveedores, reglas.
- `rules/<id>.yaml`: condiciones ponderadas (los pesos suman 1), filtros de universo,
  umbral y dirección. El hash del fichero viaja con cada señal.
- `prompts/analyst_system.md`: prompt de sistema del modelo.

El cargador valida cada fichero y las referencias entre ellos; nunca arranca con una
configuración a medias.

## Datos

Un único fichero DuckDB en `data/market.duckdb` con dos esquemas: `staging` (ingesta
cruda, nadie genera señales desde ahí) y `prod` (solo lo que pasó `quality`). Tablas:
precios, operaciones corporativas, indicadores, paneles, noticias y temas, señales,
llamadas al modelo y llamadas a proveedores con cupo. Cada tabla se crea y migra sola
al instanciarse. `data/` no está en git.

## Desarrollo

```bash
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy src tests
uv run pytest -q
```

Las tres puertas tienen que estar en verde antes de dar algo por hecho. Los tests no
tocan la red: proveedores, buscadores y clientes HTTP se sustituyen por dobles.

Estructura:

```
src/core/        cli, app (orquestación), config (esquema + cargador), scheduler, digest,
                 operations (mensaje por operación), charts (gráfico de la figura)
src/analyzer/    engine, steps/<paso>/{step,service}, storage/<tabla>, rules, universe,
                 patterns (pivotes, figuras, evaluación)
src/delivery/    consola y Telegram
config/          YAML versionado         prompts/   prompt del modelo
tests/           espejo de src           data/      DuckDB, universos (fuera de git)
```

## Licencia

Ver `LICENSE`.
