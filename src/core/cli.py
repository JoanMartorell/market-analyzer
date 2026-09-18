"""Punto de entrada: ``analyzer <comando>``.

Solo parsea argumentos, llama a ``core.app`` y traduce el resultado a texto
y código de salida. Ninguna lógica de negocio vive aquí.

Códigos de salida: 0 ok, 1 configuración inválida o pipeline fallido,
2 faltan credenciales.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from analyzer.universe import build_universe, load_constituents, members_as_of
from core.app import (
    backfill_prices,
    bootstrap,
    default_session,
    delivery_env_requirements,
    prices_status,
    reconcile_region,
    run_region,
    telegram_chats,
)
from core.config import AppConfig, ConfigError, Region
from core.scheduler import serve as serve_forever
from delivery import DeliveryError

app = typer.Typer(no_args_is_help=True, add_completion=False)
universe_app = typer.Typer(
    no_args_is_help=True, help="Universo de valores: constituyentes point-in-time por región."
)
app.add_typer(universe_app, name="universe")
ingest_app = typer.Typer(
    no_args_is_help=True, help="Precios diarios: carga histórica y estado de staging."
)
app.add_typer(ingest_app, name="ingest")
telegram_app = typer.Typer(no_args_is_help=True, help="Entrega por Telegram: descubrir el chat.")
app.add_typer(telegram_app, name="telegram")

ConfigDirOption = Annotated[
    Path | None,
    typer.Option(
        "--config-dir", "-c", help="Directorio de configuración (por defecto MA_CONFIG_DIR)."
    ),
]
RegionOption = Annotated[
    str, typer.Option("--region", "-r", help="Id de la región, p. ej. americas.")
]
DateOption = Annotated[
    datetime | None,
    typer.Option(
        "--date",
        "-d",
        formats=["%Y-%m-%d"],
        help="Sesión (YYYY-MM-DD). Por defecto, la última ya cerrada.",
    ),
]


def _region(cfg: AppConfig, region_id: str) -> Region:
    region = cfg.regions.get(region_id)
    if region is None:
        raise ConfigError(f"región desconocida: {region_id!r}")
    return region


@app.callback()
def main() -> None:
    """market-analyzer: screener multi-región con reglas deterministas y capa LLM mínima."""


def _fail(message: str, code: int) -> typer.Exit:
    typer.secho(message, fg=typer.colors.RED, err=True)
    return typer.Exit(code=code)


@app.command("config-check")
def config_check(config_dir: ConfigDirOption = None) -> None:
    """Valida config/*.yaml y .env, y muestra qué queda activo."""
    try:
        cfg = bootstrap(config_dir)
        delivery_requirements = delivery_env_requirements(cfg)
    except (ConfigError, DeliveryError) as exc:
        raise _fail(f"Configuración inválida\n{exc}", 1) from None

    llm = cfg.settings.llm
    typer.echo(f"config:  {cfg.config_dir}")
    typer.echo(f"datos:   {cfg.data_dir}")
    typer.echo(f"moneda:  {cfg.settings.base_currency}   tz: {cfg.settings.local_timezone}")
    typer.echo(f"llm:     {'on' if llm.enabled else 'off'} ({llm.model}, effort={llm.effort})")
    typer.echo(f"entrega: {', '.join(cfg.settings.delivery.channels)}")
    typer.echo("")

    for region in cfg.regions.values():
        state = "ON " if region.enabled else "off"
        rules = ", ".join(region.rules)
        typer.echo(f"[{state}] {region.id:10} {region.schedule.run_at:%H:%M}  reglas: {rules}")

    typer.echo("")
    for rule in cfg.rules.values():
        typer.echo(
            f"regla {rule.id:24} hash {rule.source_hash[:12]}  umbral {rule.output.score_threshold}"
        )

    missing = cfg.missing_env_vars(extra=delivery_requirements)
    typer.echo("")
    if missing:
        typer.secho("Faltan variables de entorno:", fg=typer.colors.YELLOW)
        for consumer, names in missing.items():
            typer.echo(f"  {consumer:22} {', '.join(names)}")
        raise typer.Exit(code=2)
    typer.secho("Credenciales completas para la configuración activa.", fg=typer.colors.GREEN)


@app.command()
def run(
    region: RegionOption,
    as_of: DateOption = None,
    config_dir: ConfigDirOption = None,
) -> None:
    """Ejecuta el ciclo diario de una región y entrega el resultado."""
    try:
        cfg = bootstrap(config_dir)
        day = as_of.date() if as_of else default_session(cfg, region)
        result = run_region(cfg, region, day)
    except (ConfigError, DeliveryError) as exc:
        raise _fail(str(exc), 1) from None

    typer.echo(result.summary())
    raise typer.Exit(code=0 if result.ok else 1)


@app.command()
def reconcile(
    region: RegionOption,
    as_of: DateOption = None,
    config_dir: ConfigDirOption = None,
) -> None:
    """Concilia las señales que se ejecutaban en una sesión con su apertura real (paso 12).

    El ciclo diario ya lo hace al terminar; esto sirve para repetirlo o para
    un día en que el ciclo no llegó al final. Necesita las velas de la sesión en prod.
    """
    try:
        cfg = bootstrap(config_dir)
        day = as_of.date() if as_of else default_session(cfg, region)
        result = reconcile_region(cfg, region, day)
    except ConfigError as exc:
        raise _fail(str(exc), 1) from None

    typer.echo(result.summary())
    raise typer.Exit(code=0 if result.ok else 1)


@app.command()
def serve(
    config_dir: ConfigDirOption = None,
    max_runs: Annotated[
        int | None,
        typer.Option("--max-runs", help="Detenerse tras N ciclos (para pruebas)."),
    ] = None,
) -> None:
    """Proceso residente: lanza el ciclo de cada región activa a su hora (schedule.run_at)."""
    try:
        cfg = bootstrap(config_dir)
        serve_forever(cfg, max_runs=max_runs)
    except (ConfigError, DeliveryError) as exc:
        raise _fail(str(exc), 1) from None
    except KeyboardInterrupt:
        typer.echo("serve detenido por el usuario", err=True)
        raise typer.Exit(code=130) from None


@universe_app.command("build")
def universe_build(region: RegionOption, config_dir: ConfigDirOption = None) -> None:
    """Descarga las fuentes y escribe el fichero de constituyentes de la región."""
    try:
        cfg = bootstrap(config_dir)
        target = _region(cfg, region)
        report = build_universe(
            target,
            cfg.constituents_path(target),
            user_agent=cfg.env.get("SEC_EDGAR_USER_AGENT"),
        )
    except (ConfigError, ValueError) as exc:
        raise _fail(str(exc), 1) from None

    typer.echo(f"universo:          {report.universe_id}")
    typer.echo(f"fichero:           {report.path}")
    typer.echo(f"fuente:            {report.source}")
    mode = "histórico" if report.history else "instantánea (acumula cambios desde el primer build)"
    typer.echo(f"modo:              {mode}")
    typer.echo(f"intervalos:        {report.rows}")
    typer.echo(f"miembros actuales: {report.current_members}")
    for enricher, notes in report.notes.items():
        typer.echo(f"{enricher}:")
        for key, value in notes.items():
            shown = ", ".join(map(str, value)) or "-" if isinstance(value, list) else value
            typer.echo(f"  {key:22} {shown}")


@universe_app.command("show")
def universe_show(
    region: RegionOption,
    as_of: DateOption = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Filas a mostrar.")] = 15,
    config_dir: ConfigDirOption = None,
) -> None:
    """Muestra los valores vigentes en la región para una fecha."""
    try:
        cfg = bootstrap(config_dir)
        target = _region(cfg, region)
        day = as_of.date() if as_of else default_session(cfg, region)
        table = load_constituents(cfg.constituents_path(target))
    except (ConfigError, FileNotFoundError) as exc:
        raise _fail(str(exc), 1) from None

    members = members_as_of(table, day)
    typer.echo(f"{target.universe.id} a {day}: {len(members)} valores")
    by_sector = members["sector"].value_counts(dropna=False)
    for sector, count in by_sector.items():
        typer.echo(f"  {sector!s:32} {count}")
    typer.echo("")
    columns = ["ticker", "name", "sector", "mic", "start", "end"]
    typer.echo(members[columns].head(limit).to_string(index=False))


@ingest_app.command("backfill")
def ingest_backfill(
    region: RegionOption,
    years: Annotated[int, typer.Option("--years", "-y", help="Años de histórico.", min=1)] = 3,
    as_of: DateOption = None,
    config_dir: ConfigDirOption = None,
) -> None:
    """Descarga N años de velas hasta la fecha para todo valor que pasó por el índice."""
    try:
        cfg = bootstrap(config_dir)
        day = as_of.date() if as_of else default_session(cfg, region)
        report = backfill_prices(cfg, region, day, years)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        raise _fail(str(exc), 1) from None

    typer.echo(f"proveedor:   {report.provider}")
    typer.echo(f"rango:       {report.start} -> {report.as_of}")
    typer.echo(f"valores:     {report.received}/{report.requested} con datos")
    typer.echo(f"velas:       {report.rows}")
    if report.missing:
        typer.echo(f"sin datos:   {', '.join(report.missing[:20])}")
        if len(report.missing) > 20:
            typer.echo(f"             ... y {len(report.missing) - 20} más")


@ingest_app.command("status")
def ingest_status(
    region: RegionOption,
    as_of: DateOption = None,
    config_dir: ConfigDirOption = None,
) -> None:
    """Muestra qué hay en staging.prices y qué valores no llegan a la sesión indicada."""
    try:
        cfg = bootstrap(config_dir)
        day = as_of.date() if as_of else default_session(cfg, region)
        status = prices_status(cfg, region, day)
    except (ConfigError, FileNotFoundError) as exc:
        raise _fail(str(exc), 1) from None

    typer.echo(f"base:        {cfg.duckdb_path}")
    typer.echo(f"universo:    {region} a {day}")
    typer.echo(f"velas:       {status['rows']}")
    typer.echo(f"valores:     {status['keys']}")
    typer.echo(f"rango:       {status['first_date']} -> {status['last_date']}")
    _echo_list("sin ninguna vela", status["absent"])
    _echo_list(f"sin llegar a {day}", status["stale"])


@telegram_app.command("chats")
def telegram_chats_cmd(config_dir: ConfigDirOption = None) -> None:
    """Muestra los chats que han escrito al bot y el TELEGRAM_CHAT_ID que hay que poner en .env."""
    try:
        cfg = bootstrap(config_dir)
        chats = telegram_chats(cfg)
    except ConfigError as exc:
        raise _fail(str(exc), 2) from None
    except DeliveryError as exc:
        raise _fail(str(exc), 1) from None

    if not chats:
        typer.secho("El bot no ha recibido ningún mensaje.", fg=typer.colors.YELLOW)
        typer.echo("Abre Telegram, busca tu bot, pulsa Start (o escribe cualquier cosa) y")
        typer.echo("vuelve a lanzar este comando. Telegram solo guarda los mensajes 24 horas.")
        raise typer.Exit(code=1)

    typer.echo("Chats que han escrito al bot (el más reciente primero):")
    for chat in chats:
        last = f'  último: "{chat.last_text}"' if chat.last_text else ""
        typer.echo(f"  {chat.id:<16} {chat.kind:10} {chat.title}{last}")
    typer.echo("")
    typer.secho(f"Pon en .env:  TELEGRAM_CHAT_ID={chats[0].id}", fg=typer.colors.GREEN)


def _echo_list(label: str, items: list[str], limit: int = 20) -> None:
    typer.echo(f"{label}: {len(items)}")
    for item in items[:limit]:
        typer.echo(f"  {item}")
    if len(items) > limit:
        typer.echo(f"  ... y {len(items) - limit} más")


if __name__ == "__main__":
    app()
