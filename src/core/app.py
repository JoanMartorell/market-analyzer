"""Arranque y ejecución.

``bootstrap`` carga la configuración y el logging. ``run_region`` monta el
pipeline diario de una región, lo ejecuta y entrega el resultado. Aquí es
donde ``analyzer`` y ``delivery`` se conocen; ninguno de los dos importa al otro.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import structlog

from analyzer.engine import Pipeline, PipelineResult, StepContext
from analyzer.sessions import last_closed_session
from analyzer.steps import build_daily_steps, build_reconcile_steps
from analyzer.steps.ingest import IngestReport, get_price_provider, ingest_prices
from analyzer.steps.ingest.service import universe_keys
from analyzer.storage import PriceStore, connect
from analyzer.universe import load_constituents, members_as_of, members_between
from core.config import AppConfig, ConfigError, Region, load_config
from core.digest import build_digest
from core.logs import configure_logging
from delivery import Dispatcher, Message, TelegramChat, build_dispatcher, required_env
from delivery import telegram_chats as discover_telegram_chats

log = structlog.get_logger(__name__)

# Variables de entorno que delivery puede necesitar. core las lee de Env y
# se las pasa como un mapping plano para que delivery no dependa de core.
_DELIVERY_ENV = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")


def bootstrap(config_dir: Path | None = None) -> AppConfig:
    cfg = load_config(config_dir)
    configure_logging(cfg.env.ma_log_level)
    return cfg


def now_local(cfg: AppConfig) -> datetime:
    return datetime.now(ZoneInfo(cfg.settings.local_timezone))


def default_session(cfg: AppConfig, region_id: str) -> date:
    """Sesión a procesar si no se indica una: la última ya cerrada en la región."""
    region = cfg.regions.get(region_id)
    if region is None:
        raise ConfigError(f"región desconocida: {region_id!r}")
    session = last_closed_session(region, now_local(cfg))
    if session is None:
        raise ConfigError(f"región {region_id!r}: sin sesión cerrada reciente en sus calendarios")
    return session


def delivery_env_requirements(cfg: AppConfig) -> dict[str, list[str]]:
    """Variables que exigen los canales de entrega configurados, por canal."""
    return {
        f"delivery:{channel}": list(names)
        for channel, names in required_env(cfg.settings.delivery.channels).items()
    }


def build_delivery(cfg: AppConfig) -> Dispatcher:
    credentials = {name: cfg.env.get(name) for name in _DELIVERY_ENV}
    return build_dispatcher(cfg.settings.delivery.channels, credentials)


def run_region(
    cfg: AppConfig,
    region_id: str,
    as_of: date,
    *,
    dispatcher: Dispatcher | None = None,
) -> PipelineResult:
    """Ejecuta el ciclo diario de una región, concilia la sesión anterior y entrega el resultado."""
    region = _region(cfg, region_id)
    if not region.enabled:
        raise ConfigError(f"región {region_id!r} está desactivada (enabled: false)")

    # Se construye antes de ejecutar: si faltan credenciales de entrega,
    # mejor fallar ahora que después de 5 minutos de descargas.
    dispatcher = dispatcher if dispatcher is not None else build_delivery(cfg)

    ctx = StepContext(cfg=cfg, region=region, as_of=as_of)
    log.info("pipeline.start", region=region_id, as_of=str(as_of))
    result = Pipeline(build_daily_steps()).run(ctx)
    if result.ok and not result.stopped_early:
        # Paso 12: las señales de la sesión anterior se ejecutaban en la apertura
        # de as_of, que el ciclo acaba de traer a prod. Es un pipeline aparte con su
        # propio contexto; sus pasos se suman al informe para que llegue en un mensaje.
        result.reports.extend(reconcile_region(cfg, region_id, as_of).reports)
    log.info("pipeline.end", region=region_id, ok=result.ok, stopped=result.stopped_early)

    notify(cfg, result, dispatcher, data=ctx.data)
    return result


def reconcile_region(cfg: AppConfig, region_id: str, as_of: date) -> PipelineResult:
    """Concilia las señales que se ejecutaban en la sesión ``as_of`` con su apertura real.

    Lo lanza ``run_region`` al final de cada ciclo y ``analyzer reconcile``
    a mano. Necesita en prod las velas de ``as_of``, que trae el ciclo
    diario de esa sesión; sin ellas el paso falla en cerrado.
    """
    ctx = StepContext(cfg=cfg, region=_region(cfg, region_id), as_of=as_of)
    log.info("reconcile.start", region=region_id, as_of=str(as_of))
    result = Pipeline(build_reconcile_steps()).run(ctx)
    log.info("reconcile.end", region=region_id, ok=result.ok)
    return result


def _region(cfg: AppConfig, region_id: str) -> Region:
    region = cfg.regions.get(region_id)
    if region is None:
        raise ConfigError(f"región desconocida: {region_id!r}")
    return region


def notify(
    cfg: AppConfig,
    result: PipelineResult,
    dispatcher: Dispatcher,
    data: Mapping[str, Any] | None = None,
) -> None:
    """Entrega el ciclo: el resumen de señales si fue bien, la tabla de pasos si no.

    ``data`` es el contexto que dejaron los pasos (señales, análisis, coste);
    sin él solo se puede enviar la tabla de pasos.
    """
    delivery = cfg.settings.delivery
    if not result.ok:
        if delivery.notify_on_failure:
            dispatcher.send(
                Message(
                    subject=f"[{result.region_id}] pipeline fallido en {result.failed_step}",
                    body=result.summary(),
                    severity="error",
                )
            )
        return
    if result.stopped_early:
        if delivery.notify_on_no_signals:
            dispatcher.send(
                Message(subject=f"[{result.region_id}] ciclo {result.as_of}", body=result.summary())
            )
        return
    dispatcher.send(build_digest(result, data if data is not None else {}))


def telegram_chats(cfg: AppConfig) -> list[TelegramChat]:
    """Chats que han escrito al bot de Telegram: de ahí sale ``TELEGRAM_CHAT_ID``."""
    token = cfg.env.get("TELEGRAM_BOT_TOKEN")
    if token is None:
        raise ConfigError("falta TELEGRAM_BOT_TOKEN en .env")
    return discover_telegram_chats(token)


# ---------------------------------------------------------------------------
# Precios: carga histórica y estado (analyzer ingest ...)
# ---------------------------------------------------------------------------


def backfill_prices(cfg: AppConfig, region_id: str, as_of: date, years: int) -> IngestReport:
    """Descarga ``years`` años de velas hasta ``as_of`` para todo valor que estuvo en el índice.

    Incluye a los que salieron del índice durante el rango: sin ellos el
    backtest solo vería supervivientes. Reescribe lo que ya hubiera (upsert).
    """
    region = cfg.regions.get(region_id)
    if region is None:
        raise ConfigError(f"región desconocida: {region_id!r}")
    start = as_of - timedelta(days=365 * years)
    table = load_constituents(cfg.constituents_path(region))
    members = members_between(table, start, as_of)
    log.info(
        "backfill.start", region=region_id, keys=len(members), start=str(start), end=str(as_of)
    )
    with connect(cfg) as con:
        provider = get_price_provider(region.providers.prices, cfg=cfg, con=con)
        fallback = (
            get_price_provider(region.providers.prices_fallback, cfg=cfg, con=con)
            if region.providers.prices_fallback
            else None
        )
        store = PriceStore(con, cfg.settings.data.staging_schema)
        return ingest_prices(
            members,
            as_of,
            provider=provider,
            store=store,
            lookback_days=cfg.settings.indicators.lookback_days,
            start=start,
            fallback=fallback,
        )


def prices_status(cfg: AppConfig, region_id: str, as_of: date) -> dict[str, Any]:
    """Resumen de ``staging.prices`` para el universo vigente de la región a ``as_of``."""
    region = cfg.regions.get(region_id)
    if region is None:
        raise ConfigError(f"región desconocida: {region_id!r}")
    empty: dict[str, Any] = {
        "rows": 0,
        "keys": 0,
        "first_date": None,
        "last_date": None,
        "stale": [],
        "absent": [],
    }
    if not cfg.duckdb_path.is_file():
        return empty
    members = members_as_of(load_constituents(cfg.constituents_path(region)), as_of)
    keys = universe_keys(members)
    with connect(cfg) as con:
        store = PriceStore(con, cfg.settings.data.staging_schema)
        stats = store.stats(keys)
        last = store.last_dates(keys)
    stale = last[last["last_date"] < as_of]
    stats["stale"] = [f"{t}@{m} ({d})" for t, m, d in stale.itertuples(index=False)]
    present = set(zip(last["ticker"], last["mic"], strict=True))
    stats["absent"] = [
        f"{t}@{m}"
        for t, m in zip(keys["ticker"], keys["mic"], strict=True)
        if (t, m) not in present
    ]
    return stats
