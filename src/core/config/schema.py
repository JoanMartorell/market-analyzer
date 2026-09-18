"""Modelos Pydantic de los ficheros YAML de config/.

Todos los modelos usan ``extra="forbid"``: una clave mal escrita en el YAML es
un error de carga, no un valor ignorado en silencio. Y son inmutables: la
configuración se lee una vez y no se toca durante la ejecución.
"""

from __future__ import annotations

import ast
from datetime import time
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

WEIGHT_TOLERANCE = 1e-6


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# settings.yaml
# ---------------------------------------------------------------------------


class DataSettings(StrictModel):
    dir: Path = Path("./data")
    duckdb_file: str = "market.duckdb"
    parquet_dir: str = "parquet"
    staging_schema: str = "staging"
    prod_schema: str = "prod"


class QualitySettings(StrictModel):
    min_ticker_coverage: float = Field(0.98, ge=0.0, le=1.0)
    max_daily_jump: float = Field(0.40, gt=0.0)
    zero_volume_min_avg_volume: int = Field(1_000_000, ge=0)
    feed_date_must_be_last_session: bool = True


class IndicatorSettings(StrictModel):
    lookback_days: int = Field(250, ge=50)
    full_recompute_on_corporate_action: bool = True


class ScreenerSettings(StrictModel):
    max_candidates_per_region: int = Field(20, ge=1)
    min_score: float = Field(0.0, ge=0.0, le=1.0)


class DedupSettings(StrictModel):
    minhash_threshold: float = Field(0.85, ge=0.0, le=1.0)
    minhash_num_perm: int = Field(128, ge=16)
    shingle_size: int = Field(3, ge=1)
    embedding_model: str
    embedding_cosine_threshold: float = Field(0.85, ge=0.0, le=1.0)
    cluster_window_hours: int = Field(48, ge=1)


class RepresentativeWeights(StrictModel):
    source_reliability: float = Field(ge=0.0, le=1.0)
    is_oldest: float = Field(ge=0.0, le=1.0)
    normalized_length: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> RepresentativeWeights:
        total = self.source_reliability + self.is_oldest + self.normalized_length
        if abs(total - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(f"representative_weights deben sumar 1.0, suman {total:.4f}")
        return self


class NewsSettings(StrictModel):
    only_candidates_and_open_positions: bool = True
    lookback_hours: int = Field(48, ge=1)
    dedup: DedupSettings
    sentiment_model: str
    representative_weights: RepresentativeWeights


class LLMSettings(StrictModel):
    enabled: bool = True
    model: str = "claude-opus-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    max_tokens: int = Field(4096, ge=256)
    use_batch_api: bool = True
    prompt_caching: bool = True
    max_calls_per_cycle: int = Field(1, ge=0)
    monthly_budget_eur: float = Field(25.0, ge=0.0)
    prompt_file: Path


class DeliverySettings(StrictModel):
    # Nombres de canal del registro de delivery. Se validan allí, no aquí:
    # core.config no importa delivery.
    channels: list[str] = Field(default_factory=lambda: ["console"], min_length=1)
    notify_on_failure: bool = True
    notify_on_no_signals: bool = True


class PipelineSettings(StrictModel):
    lock_dir: Path
    fail_closed: bool = True
    max_runtime_minutes: int = Field(15, ge=1)


class SchedulerSettings(StrictModel):
    late_grace_minutes: int = Field(30, ge=0)  # un run_at ya pasado se lanza igual si cabe aquí
    heartbeat_seconds: int = Field(300, ge=5)  # cada cuánto se registra "esperando"


class Settings(StrictModel):
    version: Literal[1]
    base_currency: str = Field(min_length=3, max_length=3, pattern=r"^[A-Z]{3}$")
    local_timezone: str
    data: DataSettings
    quality: QualitySettings
    indicators: IndicatorSettings
    screener: ScreenerSettings
    news: NewsSettings
    llm: LLMSettings
    delivery: DeliverySettings
    pipeline: PipelineSettings
    scheduler: SchedulerSettings = SchedulerSettings()

    @field_validator("local_timezone")
    @classmethod
    def _timezone_exists(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"zona horaria desconocida: {value!r}") from exc
        return value


# ---------------------------------------------------------------------------
# regions/*.yaml
# ---------------------------------------------------------------------------


class Market(StrictModel):
    mic: str = Field(pattern=r"^[A-Z0-9]{4}$")  # ISO 10383
    calendar: str  # nombre en exchange_calendars
    currency: str = Field(pattern=r"^[A-Z]{3}$")


class ReevaluationGate(StrictModel):
    at: time
    reference: str
    cancel_if_abs_move_above: float = Field(gt=0.0, lt=1.0)


class Schedule(StrictModel):
    run_at: time
    execute_at: Literal["next_open"] = "next_open"
    reevaluation_gate: ReevaluationGate | None = None


class Universe(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    constituents_file: Path
    min_avg_volume_20: int = Field(ge=0)


class RegionProviders(StrictModel):
    prices: str
    fundamentals: str | None = None
    macro: str | None = None
    news: list[str] = Field(default_factory=list)


class RegionNews(StrictModel):
    language: str
    strategy: Literal["native", "translate", "technical_only"]


class Region(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str
    enabled: bool = False
    markets: list[Market] = Field(min_length=1)
    schedule: Schedule
    universe: Universe
    providers: RegionProviders
    news: RegionNews
    rules: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _news_strategy_needs_sources(self) -> Region:
        if self.news.strategy != "technical_only" and not self.providers.news:
            raise ValueError(
                f"región {self.id!r}: news.strategy={self.news.strategy!r} "
                "pero providers.news está vacío"
            )
        return self

    @property
    def currencies(self) -> frozenset[str]:
        return frozenset(m.currency for m in self.markets)

    @property
    def calendars(self) -> frozenset[str]:
        return frozenset(m.calendar for m in self.markets)


# ---------------------------------------------------------------------------
# rules/*.yaml
# ---------------------------------------------------------------------------


def _check_expression_syntax(expr: str) -> str:
    """Comprueba solo la sintaxis. La lista blanca de nodos la aplica rules/parser.py."""
    try:
        ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"expresión inválida {expr!r}: {exc.msg}") from exc
    return expr


class Condition(StrictModel):
    expr: str
    weight: float = Field(gt=0.0, le=1.0)
    required: bool = False

    _syntax = field_validator("expr")(_check_expression_syntax)


class RuleOutput(StrictModel):
    score_threshold: float = Field(ge=0.0, le=1.0)
    direction: Literal["long", "short"] = "long"


class Rule(StrictModel):
    version: Literal[3]
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = ""
    window: int = Field(ge=1)
    conditions: list[Condition] = Field(min_length=1)
    universe_filters: list[str] = Field(default_factory=list)
    output: RuleOutput
    source_hash: str = ""  # sha256 del fichero; lo rellena el cargador

    @field_validator("universe_filters")
    @classmethod
    def _filters_syntax(cls, filters: list[str]) -> list[str]:
        return [_check_expression_syntax(f) for f in filters]

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> Rule:
        total = sum(c.weight for c in self.conditions)
        if abs(total - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(f"regla {self.id!r}: los pesos deben sumar 1.0, suman {total:.4f}")
        return self


# ---------------------------------------------------------------------------
# sources.yaml
# ---------------------------------------------------------------------------


class Provider(StrictModel):
    kind: Literal["prices", "fundamentals", "macro"]
    base_url: str | None = None
    api_key_env: str | None = None
    user_agent_env: str | None = None
    rate_limit_per_second: float | None = Field(default=None, gt=0.0)
    monthly_cost_eur: float = Field(0.0, ge=0.0)
    notes: str = ""

    @property
    def required_env(self) -> list[str]:
        return [e for e in (self.api_key_env, self.user_agent_env) if e]


class NewsSource(StrictModel):
    kind: Literal["rss", "api"]
    url: str | None = None
    url_template: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    per_ticker: bool = False
    reliability: float = Field(ge=0.0, le=1.0)
    language: str = "en"
    rate_limit_per_minute: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _rss_needs_url(self) -> NewsSource:
        if self.kind == "rss":
            if self.per_ticker and not (self.url_template and "{ticker}" in self.url_template):
                raise ValueError("fuente RSS per_ticker necesita url_template con {ticker}")
            if not self.per_ticker and not self.url:
                raise ValueError("fuente RSS global necesita url")
        if self.kind == "api" and not self.base_url:
            raise ValueError("fuente api necesita base_url")
        return self

    @property
    def required_env(self) -> list[str]:
        return [self.api_key_env] if self.api_key_env else []


class Sources(StrictModel):
    version: Literal[1]
    providers: dict[str, Provider]
    news_sources: dict[str, NewsSource]
