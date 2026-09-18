"""Una llamada al modelo por ciclo, con su coste medido y guardada.

Este paso es la única parte del pipeline que cuesta dinero por ejecución, así
que se comporta como tal:

- **Una llamada y solo una.** Los candidatos del día van juntos en un mismo
  mensaje; no hay una llamada por valor.
- **Repetir el ciclo no vuelve a pagar.** Si ya hay una respuesta guardada
  para esa región y ese día con la misma entrada (mismo hash del payload) y
  el mismo modelo, se reutiliza.
- **Se valida lo que devuelve.** El esquema garantiza la forma; aquí se
  comprueba el fondo: un veredicto por candidato y ninguno inventado. Si no
  cuadra, el día falla y no se guarda nada: una respuesta que habla de otros
  valores no es mejor que no tener respuesta.
- **Se anota el coste.** Con los tokens y la tarifa de ``llm.pricing`` se
  estima el gasto del mes y se compara con ``monthly_budget_eur``. Es un
  aviso, no un freno: el freno es ``llm.enabled``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd
import structlog
from pydantic import ValidationError

from analyzer.steps.llm.client import LlmClient, TokenUsage
from analyzer.steps.llm.payload import build_payload, dumps, payload_hash
from analyzer.steps.llm.schema import VERDICTS, Analysis
from analyzer.storage import LlmCallStore
from core.config import LlmPricing, LLMSettings

log = structlog.get_logger(__name__)

MILLION = 1_000_000
CANDIDATE = ("candidato", "candidatos")
RAW_IN_LOG = 500  # de una respuesta inválida, lo justo para entender qué pasó


@dataclass(frozen=True)
class LlmReport:
    region_id: str
    as_of: date
    model: str
    candidates: int
    usage: TokenUsage
    cost_eur: float
    month_cost_eur: float
    budget_eur: float
    batch: bool
    reused: bool  # respondió la base de datos, no la API
    analysis: Analysis = field(repr=False, compare=False)

    @property
    def over_budget(self) -> bool:
        return self.month_cost_eur > self.budget_eur

    @property
    def tokens(self) -> int:
        usage = self.usage
        return (
            usage.input_tokens
            + usage.output_tokens
            + usage.cache_read_tokens
            + usage.cache_write_tokens
        )

    def summary(self) -> str:
        counts = ", ".join(
            f"{len(self.analysis.by_verdict(name))} {name}"
            for name in VERDICTS
            if self.analysis.by_verdict(name)
        )
        text = f"{_count(self.candidates, CANDIDATE)} -> {counts or 'sin veredictos'}"
        if self.reused:
            return f"{text}; respuesta reutilizada, sin coste"
        how = " en lote" if self.batch else ""
        text += f"; {self.tokens} tokens{how}, {self.cost_eur:.4f} EUR"
        text += f", {self.month_cost_eur:.2f} de {self.budget_eur:.2f} EUR este mes"
        if self.over_budget:
            text += " (PRESUPUESTO SUPERADO)"
        return text


def run_llm(
    candidates: pd.DataFrame,
    clusters: pd.DataFrame,
    as_of: date,
    *,
    region_id: str,
    currency: str,
    settings: LLMSettings,
    prompt: str,
    client: LlmClient,
    store: LlmCallStore,
    panel: pd.DataFrame | None = None,
) -> LlmReport:
    payload = build_payload(
        candidates,
        clusters,
        region_id=region_id,
        as_of=as_of,
        currency=currency,
        panel=panel,
    )
    digest = payload_hash(payload)

    saved = _reusable(store.get(region_id, as_of), digest, settings.model, candidates)
    if saved is not None:
        row, analysis = saved
        log.info("llm.reused", region=region_id, as_of=str(as_of), payload=digest[:12])
        return _report(
            region_id,
            as_of,
            settings,
            candidates=len(candidates),
            usage=TokenUsage(),
            cost=0.0,
            month_cost=store.month_cost(as_of),
            batch=bool(row["batch"]),
            reused=True,
            analysis=analysis,
        )

    reply = client.ask(prompt, dumps(payload))
    analysis = check_answer(parse_answer(reply.text), candidates)
    cost = estimate_cost(reply.usage, settings.pricing, batch=reply.batch)
    # Se guarda después de validar: una respuesta que no cuadra no puede
    # quedarse en la tabla, o el ciclo siguiente la reutilizaría.
    store.upsert(
        {
            "region": region_id,
            "date": as_of,
            "model": settings.model,
            "payload_hash": digest,
            "batch": reply.batch,
            "input_tokens": reply.usage.input_tokens,
            "output_tokens": reply.usage.output_tokens,
            "cache_read_tokens": reply.usage.cache_read_tokens,
            "cache_write_tokens": reply.usage.cache_write_tokens,
            "cost_eur": cost,
            "response": reply.text,
        }
    )
    report = _report(
        region_id,
        as_of,
        settings,
        candidates=len(candidates),
        usage=reply.usage,
        cost=cost,
        month_cost=store.month_cost(as_of),
        batch=reply.batch,
        reused=False,
        analysis=analysis,
    )
    if report.over_budget:
        log.warning("llm.over_budget", month=report.month_cost_eur, budget=report.budget_eur)
    log.info("llm.done", detail=report.summary())
    return report


def parse_answer(raw: str) -> Analysis:
    try:
        return Analysis.model_validate_json(raw)
    except ValidationError as exc:
        log.error("llm.invalid_answer", raw=raw[:RAW_IN_LOG])
        raise ValueError(
            f"la respuesta no encaja en el esquema: {exc.error_count()} errores"
        ) from exc


def check_answer(analysis: Analysis, candidates: pd.DataFrame) -> Analysis:
    """Un veredicto por candidato, ni uno menos ni uno de más."""
    expected = {str(t) for t in candidates["ticker"]}
    answered = set(analysis.tickers)
    invented = sorted(answered - expected)
    if invented:
        raise ValueError(f"el modelo responde por valores que no se enviaron: {_join(invented)}")
    forgotten = sorted(expected - answered)
    if forgotten:
        raise ValueError(f"el modelo no responde por {_join(forgotten)}")
    if len(analysis.verdicts) != len(expected):
        raise ValueError(f"{len(analysis.verdicts)} veredictos para {len(expected)} candidatos")
    return analysis


def estimate_cost(usage: TokenUsage, pricing: LlmPricing, *, batch: bool) -> float:
    """EUR de una llamada según la tarifa configurada. Estimación, no factura."""
    entry = pricing.input_per_mtok
    total = (
        usage.input_tokens * entry
        + usage.cache_write_tokens * entry * pricing.cache_write_multiplier
        + usage.cache_read_tokens * entry * pricing.cache_read_multiplier
        + usage.output_tokens * pricing.output_per_mtok
    ) / MILLION
    if batch:
        total *= pricing.batch_multiplier
    return round(total, 6)


def _reusable(
    saved: dict[str, Any] | None, digest: str, model: str, candidates: pd.DataFrame
) -> tuple[dict[str, Any], Analysis] | None:
    """La llamada guardada sirve si la entrada y el modelo son los mismos y aún se entiende.

    Una respuesta anterior a un cambio del esquema (un campo nuevo, por
    ejemplo) no encaja: se avisa y se vuelve a llamar en vez de fallar el día.
    """
    if saved is None or saved["payload_hash"] != digest or saved["model"] != model:
        return None
    try:
        return saved, check_answer(parse_answer(str(saved["response"])), candidates)
    except ValueError as exc:
        log.warning("llm.saved_answer_unusable", detail=str(exc))
        return None


def _report(
    region_id: str,
    as_of: date,
    settings: LLMSettings,
    *,
    candidates: int,
    usage: TokenUsage,
    cost: float,
    month_cost: float,
    batch: bool,
    reused: bool,
    analysis: Analysis,
) -> LlmReport:
    return LlmReport(
        region_id=region_id,
        as_of=as_of,
        model=settings.model,
        candidates=candidates,
        usage=usage,
        cost_eur=cost,
        month_cost_eur=round(month_cost, 6),
        budget_eur=settings.monthly_budget_eur,
        batch=batch,
        reused=reused,
        analysis=analysis,
    )


def _count(number: int, names: tuple[str, str]) -> str:
    singular, plural = names
    return f"{number} {singular if number == 1 else plural}"


def _join(tickers: list[str]) -> str:
    return ", ".join(tickers)
