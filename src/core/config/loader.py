"""Carga config/*.yaml, valida cada fichero y las referencias entre ellos.

Orden: settings.yaml -> sources.yaml -> regions/*.yaml -> rules/*.yaml.
Después se comprueba que cada región apunte a proveedores, fuentes, reglas y
calendarios que existen, y que dos regiones activas no compartan bolsa: la
clave de una señal es (ticker, mic, fecha, regla), sin región, y una pisaría
las señales de la otra. Un fallo en cualquier punto lanza ``ConfigError``
con la ruta del fichero: nunca se arranca con una configuración a medias.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import yaml
from pydantic import BaseModel, ValidationError

from core.config.env import Env
from core.config.schema import Region, Rule, Settings, Sources


class ConfigError(Exception):
    """Configuración inválida o incompleta."""


@dataclass(frozen=True)
class AppConfig:
    root: Path
    config_dir: Path
    env: Env
    settings: Settings
    sources: Sources
    regions: dict[str, Region]
    rules: dict[str, Rule]

    # --- accesos cómodos ---------------------------------------------------

    def enabled_regions(self) -> list[Region]:
        return [r for r in self.regions.values() if r.enabled]

    @property
    def data_dir(self) -> Path:
        base = self.env.ma_data_dir or self.settings.data.dir
        return self._resolve(base)

    @property
    def duckdb_path(self) -> Path:
        return self.data_dir / self.settings.data.duckdb_file

    @property
    def parquet_dir(self) -> Path:
        return self.data_dir / self.settings.data.parquet_dir

    @property
    def lock_dir(self) -> Path:
        return self._resolve(self.settings.pipeline.lock_dir)

    @property
    def prompt_path(self) -> Path:
        return self._resolve(self.settings.llm.prompt_file)

    def rules_for(self, region: Region) -> list[Rule]:
        return [self.rules[rid] for rid in region.rules]

    def universe_dir(self, region: Region) -> Path:
        """Carpeta de universo de la región: ``<data>/universe/<region>/``."""
        return self.data_dir / "universe" / region.id

    def constituents_path(self, region: Region) -> Path:
        """Fichero de constituyentes: ``universe.constituents_file`` dentro de ``universe_dir``."""
        file = region.universe.constituents_file
        return file if file.is_absolute() else self.universe_dir(region) / file

    def _resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.root / path).resolve()

    # --- credenciales ------------------------------------------------------

    def required_env_vars(
        self, extra: Mapping[str, Sequence[str]] | None = None
    ) -> dict[str, list[str]]:
        """Variables de entorno que exige la configuración activa, por consumidor.

        ``extra`` permite añadir requisitos de otros paquetes (p. ej. los
        canales de delivery) sin que core.config los conozca.
        """
        required: dict[str, list[str]] = {}

        for region in self.enabled_regions():
            provider_ids = [
                region.providers.prices,
                region.providers.prices_fallback,
                region.providers.fundamentals,
                region.providers.macro,
            ]
            for pid in provider_ids:
                if pid is not None:
                    _add(required, f"provider:{pid}", self.sources.providers[pid].required_env)
            for sid in region.providers.news:
                _add(required, f"news:{sid}", self.sources.news_sources[sid].required_env)

        if self.settings.llm.enabled:
            _add(required, "llm", ["ANTHROPIC_API_KEY"])
        for consumer, names in (extra or {}).items():
            _add(required, consumer, list(names))

        return required

    def missing_env_vars(
        self, extra: Mapping[str, Sequence[str]] | None = None
    ) -> dict[str, list[str]]:
        """Subconjunto de ``required_env_vars`` que no está definido en el entorno."""
        return {
            consumer: missing
            for consumer, names in self.required_env_vars(extra).items()
            if (missing := [n for n in names if not self.env.has(n)])
        }


def _add(target: dict[str, list[str]], key: str, names: list[str]) -> None:
    if names:
        target.setdefault(key, []).extend(n for n in names if n not in target.get(key, []))


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------


def load_config(config_dir: Path | str | None = None, env: Env | None = None) -> AppConfig:
    """Carga y valida toda la configuración.

    ``config_dir`` por defecto es ``MA_CONFIG_DIR`` (o ``config``). La raíz del
    proyecto, contra la que se resuelven rutas relativas, es su directorio padre.
    """
    env = env if env is not None else Env()
    cfg_dir = Path(config_dir if config_dir is not None else env.ma_config_dir).resolve()
    if not cfg_dir.is_dir():
        raise ConfigError(f"directorio de configuración no encontrado: {cfg_dir}")
    root = cfg_dir.parent

    settings = _load_model(Settings, cfg_dir / "settings.yaml")
    sources = _load_model(Sources, cfg_dir / "sources.yaml")
    regions = _load_dir(Region, cfg_dir / "regions")
    rules = _load_dir(Rule, cfg_dir / "rules", with_hash=True)

    _validate_references(regions, rules, sources)

    return AppConfig(
        root=root,
        config_dir=cfg_dir,
        env=env,
        settings=settings,
        sources=sources,
        regions=regions,
        rules=rules,
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"fichero no encontrado: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: YAML inválido: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: se esperaba un mapeo en la raíz")
    return data


def _load_model[ModelT: BaseModel](
    model: type[ModelT], path: Path, extra: dict[str, Any] | None = None
) -> ModelT:
    data = _read_yaml(path)
    if extra:
        data.update(extra)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{path}:\n{exc}") from exc


def _load_dir[ModelT: BaseModel](
    model: type[ModelT], directory: Path, *, with_hash: bool = False
) -> dict[str, ModelT]:
    if not directory.is_dir():
        raise ConfigError(f"directorio no encontrado: {directory}")
    loaded: dict[str, ModelT] = {}
    for path in sorted(directory.glob("*.yaml")):
        extra = {"source_hash": _sha256(path)} if with_hash else None
        item = _load_model(model, path, extra)
        # Region y Rule tienen `id`; BaseModel no, de ahí getattr.
        item_id: str = getattr(item, "id")  # noqa: B009
        if item_id != path.stem:
            raise ConfigError(f"{path}: id={item_id!r} no coincide con el nombre del fichero")
        loaded[item_id] = item
    if not loaded:
        raise ConfigError(f"no hay ficheros .yaml en {directory}")
    return loaded


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_references(
    regions: dict[str, Region], rules: dict[str, Rule], sources: Sources
) -> None:
    known_calendars = set(xcals.get_calendar_names(include_aliases=True))
    problems: list[str] = []

    for region in regions.values():
        prefix = f"región {region.id!r}"

        for role, pid in (
            ("prices", region.providers.prices),
            ("prices", region.providers.prices_fallback),
            ("fundamentals", region.providers.fundamentals),
            ("macro", region.providers.macro),
        ):
            if pid is None:
                continue
            provider = sources.providers.get(pid)
            if provider is None:
                problems.append(f"{prefix}: proveedor de {role} desconocido {pid!r}")
            elif provider.kind != role:
                problems.append(
                    f"{prefix}: {pid!r} es de tipo {provider.kind!r}, usado como {role!r}"
                )

        for sid in region.providers.news:
            if sid not in sources.news_sources:
                problems.append(f"{prefix}: fuente de noticias desconocida {sid!r}")

        for rid in region.rules:
            if rid not in rules:
                problems.append(f"{prefix}: regla desconocida {rid!r}")

        for cal in sorted(region.calendars):
            if cal not in known_calendars:
                problems.append(f"{prefix}: calendario {cal!r} no existe en exchange_calendars")

    owners: dict[str, str] = {}
    for region in regions.values():
        if not region.enabled:
            continue
        for market in region.markets:
            owner = owners.setdefault(market.mic, region.id)
            if owner != region.id:
                problems.append(
                    f"región {region.id!r}: la bolsa {market.mic} ya la cubre la región "
                    f"activa {owner!r}; desactiva una de las dos"
                )

    if problems:
        raise ConfigError("referencias inválidas:\n  - " + "\n  - ".join(problems))
