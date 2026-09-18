"""Paso llm: payload, esquema de respuesta, cliente, servicio y paso."""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
import pytest
from anthropic.types import ErrorResponse, InvalidRequestError, Message, TextBlock, Usage
from anthropic.types.messages import MessageBatchErroredResult

from analyzer.engine import StepContext, StepError, StepStatus
from analyzer.steps.llm import (
    AnthropicClient,
    Llm,
    LlmError,
    Reply,
    TokenUsage,
    build_payload,
    estimate_cost,
    payload_hash,
    run_llm,
)
from analyzer.steps.llm.payload import MAX_TOPICS, SUMMARY_CHARS
from analyzer.steps.llm.schema import Analysis, json_schema
from analyzer.steps.llm.service import check_answer, parse_answer
from analyzer.steps.screener import CANDIDATE_COLUMNS
from analyzer.storage import CALL_COLUMNS, CLUSTER_COLUMNS, LlmCallStore, connect
from core.config import AppConfig, Env, LlmPricing, LLMSettings

AS_OF = date(2026, 9, 17)
NOW = datetime(2026, 9, 17, 20, 0, tzinfo=UTC)
PROMPT = "Eres el analista."


# --- datos de prueba --------------------------------------------------------------------


def _candidate(ticker: str = "AAA", **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ticker": ticker,
        "mic": "XNYS",
        "rule_id": "pullback",
        "direction": "long",
        "score": 0.8127,
        "conditions_met": "rsi_14 < 35; close > sma_200",
        "close": 101.23456,
        "sector": "Industrials",
        "rule_hash": "abc",
    }
    row.update(overrides)
    return row


def _candidates(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows) or [_candidate()], columns=[*CANDIDATE_COLUMNS])


def _cluster(ticker: str | None = "AAA", **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cluster_id": overrides.pop("cluster_id", "c1"),
        "ticker": ticker,
        "mic": "XNYS" if ticker else None,
        "representative_id": "n1",
        "size": 2,
        "method": "exacto",
        "sources": "finnhub, yahoo",
        "reliability": 0.7,
        "language": "en",
        "first_published_at": NOW - timedelta(hours=2),
        "last_published_at": NOW,
        "title": "AAA gana un contrato",
        "summary": "La compañía firma un contrato de diez años.",
        "url": "https://news.test/1",
        "sentiment_label": "positive",
        "sentiment_score": 0.9123,
    }
    row.update(overrides)
    return row


def _clusters(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=[*CLUSTER_COLUMNS])


def _panel(ticker: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame([{"ticker": ticker, "mic": "XNYS", "name": "Alpha Inc", "currency": "USD"}])


def _answer(*tickers: str, summary: str = "Día tranquilo.") -> str:
    return json.dumps(
        {
            "summary": summary,
            "verdicts": [
                {
                    "ticker": t,
                    "verdict": "confirmar",
                    "confidence": 0.6,
                    "headline": "Sin noticias en contra",
                    "rationale": "La señal técnica se sostiene.",
                    "risks": ["Resultados en dos semanas."],
                }
                for t in tickers
            ],
        }
    )


# --- dobles -----------------------------------------------------------------------------


@dataclass
class FakeClient:
    """Cliente que devuelve siempre lo mismo y cuenta cuántas veces se le pregunta."""

    text: str
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(1000, 500, 0, 0))
    batch: bool = False
    asked: list[tuple[str, str]] = field(default_factory=list)

    def ask(self, system: str, payload: str) -> Reply:
        self.asked.append((system, payload))
        return Reply(text=self.text, usage=self.usage, batch=self.batch)


def _message(text: str, *, stop_reason: str = "end_turn", **usage: int) -> Message:
    tokens = {"input_tokens": 100, "output_tokens": 50}
    tokens.update(usage)
    return Message(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-sonnet-5",
        content=[TextBlock(type="text", text=text)],
        stop_reason=stop_reason,
        usage=Usage(**tokens),
    )


@dataclass
class _Batch:
    id: str
    processing_status: str


@dataclass
class _Result:
    custom_id: str
    result: Any


@dataclass
class _Succeeded:
    message: Message
    type: str = "succeeded"


def _errored(message: str) -> MessageBatchErroredResult:
    return MessageBatchErroredResult(
        type="errored",
        error=ErrorResponse(
            type="error",
            error=InvalidRequestError(type="invalid_request_error", message=message),
        ),
    )


class FakeBatches:
    """Lotes que pasan por ``statuses`` en cada consulta hasta agotarlos."""

    def __init__(self, message: Message, statuses: Sequence[str], result: Any = None) -> None:
        self._message = message
        self._statuses = list(statuses)
        self._result = result
        self.created: list[Any] = []
        self.cancelled: list[str] = []

    def create(self, *, requests: Sequence[Any]) -> _Batch:
        self.created.extend(requests)
        return _Batch(id="batch_1", processing_status="in_progress")

    def retrieve(self, batch_id: str) -> _Batch:
        status = self._statuses.pop(0) if self._statuses else "ended"
        return _Batch(id=batch_id, processing_status=status)

    def cancel(self, batch_id: str) -> _Batch:
        self.cancelled.append(batch_id)
        return _Batch(id=batch_id, processing_status="canceling")

    def results(self, batch_id: str) -> Iterator[_Result]:
        outcome = self._result or _Succeeded(message=self._message)
        yield _Result(custom_id="analisis", result=outcome)


class FakeMessages:
    def __init__(self, message: Message, batches: FakeBatches) -> None:
        self._message = message
        self.batches = batches
        self.calls: list[dict[str, Any]] = []

    def create(self, **params: Any) -> Message:
        self.calls.append(params)
        return self._message


class FakeAnthropic:
    def __init__(self, message: Message, statuses: Sequence[str] = (), result: Any = None) -> None:
        self.messages = FakeMessages(message, FakeBatches(message, statuses, result))

    def with_options(self, **_: Any) -> "FakeAnthropic":
        return self


# --- fixtures ---------------------------------------------------------------------------


@pytest.fixture
def cfg_tmp(cfg: AppConfig, tmp_path: Any) -> AppConfig:
    return replace(cfg, env=Env(_env_file=None, ma_data_dir=tmp_path))


@pytest.fixture
def settings(cfg: AppConfig) -> LLMSettings:
    return cfg.settings.llm


@pytest.fixture
def store(cfg_tmp: AppConfig) -> Iterator[LlmCallStore]:
    with connect(cfg_tmp) as con:
        yield LlmCallStore(con, cfg_tmp.settings.data.prod_schema)


# --- payload ----------------------------------------------------------------------------


def test_payload_carries_the_candidate_with_its_topics() -> None:
    payload = build_payload(
        _candidates(),
        _clusters(_cluster()),
        region_id="americas",
        as_of=AS_OF,
        currency="EUR",
        panel=_panel(),
    )

    assert payload["region"] == "americas"
    assert payload["fecha"] == "2026-09-17"
    candidate = payload["candidatos"][0]
    assert candidate["ticker"] == "AAA"
    assert candidate["nombre"] == "Alpha Inc"
    assert candidate["divisa"] == "USD"
    assert candidate["cierre"] == 101.2346
    assert candidate["score"] == 0.813
    assert candidate["condiciones"] == ["rsi_14 < 35", "close > sma_200"]
    topic = candidate["temas"][0]
    assert topic["titular"] == "AAA gana un contrato"
    assert topic["sentimiento"] == 0.91
    assert topic["publicado"] == "2026-09-17T20:00Z"
    assert topic["articulos"] == 2


def test_payload_never_carries_urls_or_ids() -> None:
    payload = build_payload(
        _candidates(),
        _clusters(_cluster()),
        region_id="americas",
        as_of=AS_OF,
        currency="EUR",
    )

    text = json.dumps(payload)
    assert "https://news.test/1" not in text
    assert "cluster_id" not in text


def test_topics_without_ticker_go_apart() -> None:
    payload = build_payload(
        _candidates(),
        _clusters(_cluster(), _cluster(None, cluster_id="c2", title="La bolsa cierra plana")),
        region_id="americas",
        as_of=AS_OF,
        currency="EUR",
    )

    assert len(payload["candidatos"][0]["temas"]) == 1
    assert [t["titular"] for t in payload["temas_generales"]] == ["La bolsa cierra plana"]


def test_only_the_most_recent_topics_per_candidate() -> None:
    many = [
        _cluster(cluster_id=f"c{i}", last_published_at=NOW - timedelta(hours=i), title=f"T{i}")
        for i in range(MAX_TOPICS + 3)
    ]
    payload = build_payload(
        _candidates(),
        _clusters(*many),
        region_id="americas",
        as_of=AS_OF,
        currency="EUR",
    )

    titles = [t["titular"] for t in payload["candidatos"][0]["temas"]]
    assert titles == [f"T{i}" for i in range(MAX_TOPICS)]


def test_long_summaries_are_clipped() -> None:
    long_summary = "palabra " * 200
    payload = build_payload(
        _candidates(),
        _clusters(_cluster(summary=long_summary)),
        region_id="americas",
        as_of=AS_OF,
        currency="EUR",
    )

    clipped = payload["candidatos"][0]["temas"][0]["resumen"]
    assert len(clipped) <= SUMMARY_CHARS + 3
    assert clipped.endswith("...")


def test_missing_sentiment_is_null_not_nan() -> None:
    payload = build_payload(
        _candidates(),
        _clusters(_cluster(sentiment_score=float("nan"), sentiment_label=None)),
        region_id="americas",
        as_of=AS_OF,
        currency="EUR",
    )

    assert payload["candidatos"][0]["temas"][0]["sentimiento"] is None
    assert "NaN" not in json.dumps(payload)  # json.dumps escribiría NaN, que no es JSON válido


def test_the_hash_only_changes_when_the_input_changes() -> None:
    kwargs: dict[str, Any] = {"region_id": "americas", "as_of": AS_OF, "currency": "EUR"}
    first = payload_hash(build_payload(_candidates(), _clusters(_cluster()), **kwargs))
    same = payload_hash(build_payload(_candidates(), _clusters(_cluster()), **kwargs))
    other = payload_hash(
        build_payload(_candidates(), _clusters(_cluster(title="Otra cosa")), **kwargs)
    )

    assert first == same
    assert first != other


def test_a_candidate_without_news_keeps_an_empty_list() -> None:
    payload = build_payload(
        _candidates(),
        _clusters(),
        region_id="americas",
        as_of=AS_OF,
        currency="EUR",
    )

    assert payload["candidatos"][0]["temas"] == []
    assert payload["temas_generales"] == []


# --- respuesta --------------------------------------------------------------------------


def test_a_valid_answer_becomes_an_analysis() -> None:
    analysis = parse_answer(_answer("AAA", "BBB"))

    assert analysis.tickers == ["AAA", "BBB"]
    assert len(analysis.by_verdict("confirmar")) == 2
    assert analysis.by_verdict("descartar") == []


def test_an_answer_outside_the_schema_fails() -> None:
    with pytest.raises(ValueError, match="no encaja en el esquema"):
        parse_answer('{"summary": "x", "verdicts": [{"ticker": "AAA"}]}')


def test_an_unknown_verdict_fails() -> None:
    raw = _answer("AAA").replace("confirmar", "comprar")

    with pytest.raises(ValueError, match="no encaja en el esquema"):
        parse_answer(raw)


def test_an_invented_ticker_fails() -> None:
    with pytest.raises(ValueError, match="valores que no se enviaron: ZZZ"):
        check_answer(parse_answer(_answer("AAA", "ZZZ")), _candidates())


def test_a_forgotten_candidate_fails() -> None:
    candidates = _candidates(_candidate("AAA"), _candidate("BBB"))

    with pytest.raises(ValueError, match="no responde por BBB"):
        check_answer(parse_answer(_answer("AAA")), candidates)


# --- coste ------------------------------------------------------------------------------


def test_cost_adds_every_kind_of_token() -> None:
    # Tarifa fija: el test no depende de los multiplicadores que haya en settings.yaml.
    pricing = LlmPricing(
        input_per_mtok=3.0,
        output_per_mtok=15.0,
        cache_write_multiplier=1.25,
        cache_read_multiplier=0.1,
        batch_multiplier=0.5,
    )
    usage = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )

    direct = estimate_cost(usage, pricing, batch=False)

    # 3 (entrada) + 15 (salida) + 3*1.25 (escritura en caché) + 3*0.1 (lectura)
    assert direct == pytest.approx(22.05)
    assert estimate_cost(usage, pricing, batch=True) == pytest.approx(direct / 2)


# --- servicio ---------------------------------------------------------------------------


def _run(
    store: LlmCallStore,
    settings: LLMSettings,
    client: FakeClient,
    candidates: pd.DataFrame | None = None,
    clusters: pd.DataFrame | None = None,
) -> Any:
    return run_llm(
        candidates if candidates is not None else _candidates(),
        clusters if clusters is not None else _clusters(_cluster()),
        AS_OF,
        region_id="americas",
        currency="EUR",
        settings=settings,
        prompt=PROMPT,
        client=client,
        store=store,
    )


def test_the_service_saves_the_call_and_reports_it(
    store: LlmCallStore, settings: LLMSettings
) -> None:
    client = FakeClient(_answer("AAA"))

    report = _run(store, settings, client)

    assert report.reused is False
    assert report.candidates == 1
    assert report.tokens == 1500
    assert report.cost_eur > 0
    assert report.analysis.summary == "Día tranquilo."
    saved = store.get("americas", AS_OF)
    assert saved is not None
    assert saved["model"] == settings.model
    assert saved["input_tokens"] == 1000
    assert json.loads(saved["response"])["verdicts"][0]["ticker"] == "AAA"
    assert set(saved) == set(CALL_COLUMNS)


def test_repeating_the_cycle_does_not_pay_twice(store: LlmCallStore, settings: LLMSettings) -> None:
    client = FakeClient(_answer("AAA"))

    first = _run(store, settings, client)
    second = _run(store, settings, client)

    assert len(client.asked) == 1
    assert second.reused is True
    assert second.cost_eur == 0.0
    assert second.analysis == first.analysis
    assert "reutilizada" in second.summary()


def test_a_saved_answer_from_an_older_schema_is_not_reused(
    store: LlmCallStore, settings: LLMSettings
) -> None:
    client = FakeClient(_answer("AAA"))
    first = _run(store, settings, client)
    old = json.loads(_answer("AAA"))
    del old["verdicts"][0]["headline"]  # respuesta guardada antes de existir el campo
    saved = store.get("americas", AS_OF)
    assert saved is not None
    store.upsert({**saved, "response": json.dumps(old)})

    second = _run(store, settings, client)

    assert len(client.asked) == 2  # se vuelve a llamar en vez de fallar el día
    assert second.reused is False
    assert second.analysis == first.analysis


def test_a_different_input_replaces_the_call(store: LlmCallStore, settings: LLMSettings) -> None:
    client = FakeClient(_answer("AAA"))

    first = _run(store, settings, client)
    second = _run(store, settings, client, clusters=_clusters(_cluster(title="Profit warning")))

    assert len(client.asked) == 2
    assert second.reused is False
    # Una fila por región y día: el mes no acumula las dos llamadas del mismo día.
    assert store.month_cost(AS_OF) == pytest.approx(first.cost_eur)


def test_another_model_does_not_reuse_the_answer(
    store: LlmCallStore, settings: LLMSettings
) -> None:
    client = FakeClient(_answer("AAA"))

    _run(store, settings, client)
    _run(store, settings.model_copy(update={"model": "claude-haiku-4-5"}), client)

    assert len(client.asked) == 2


def test_an_answer_that_does_not_match_is_not_saved(
    store: LlmCallStore, settings: LLMSettings
) -> None:
    client = FakeClient(_answer("ZZZ"))

    with pytest.raises(ValueError, match="valores que no se enviaron"):
        _run(store, settings, client)

    assert store.get("americas", AS_OF) is None


def test_the_budget_warning_shows_in_the_summary(
    store: LlmCallStore, settings: LLMSettings
) -> None:
    broke = settings.model_copy(update={"monthly_budget_eur": 0.0})
    client = FakeClient(_answer("AAA"))

    report = _run(store, broke, client)

    assert report.over_budget is True
    assert "PRESUPUESTO SUPERADO" in report.summary()


def test_the_month_cost_adds_up_across_days(store: LlmCallStore, settings: LLMSettings) -> None:
    client = FakeClient(_answer("AAA"))

    first = _run(store, settings, client)
    second = run_llm(
        _candidates(),
        _clusters(_cluster()),
        date(2026, 9, 18),
        region_id="americas",
        currency="EUR",
        settings=settings,
        prompt=PROMPT,
        client=client,
        store=store,
    )

    assert second.month_cost_eur == pytest.approx(first.cost_eur * 2)


# --- cliente ----------------------------------------------------------------------------


def test_the_direct_call_sends_the_schema_and_the_cached_prompt(settings: LLMSettings) -> None:
    fake = FakeAnthropic(_message(_answer("AAA")))
    client = AnthropicClient(fake, settings.model_copy(update={"use_batch_api": False}))  # type: ignore[arg-type]

    reply = client.ask(PROMPT, '{"candidatos":[]}')

    params = fake.messages.calls[0]
    assert params["model"] == settings.model
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert params["output_config"]["effort"] == settings.effort
    assert params["output_config"]["format"]["schema"]["properties"]["verdicts"]
    assert reply.batch is False
    assert reply.usage == TokenUsage(input_tokens=100, output_tokens=50)


def test_without_caching_the_prompt_goes_plain(settings: LLMSettings) -> None:
    fake = FakeAnthropic(_message(_answer("AAA")))
    plain = settings.model_copy(update={"use_batch_api": False, "prompt_caching": False})

    AnthropicClient(fake, plain).ask(PROMPT, "{}")  # type: ignore[arg-type]

    assert "cache_control" not in fake.messages.calls[0]["system"][0]


def test_a_truncated_answer_is_an_error(settings: LLMSettings) -> None:
    fake = FakeAnthropic(_message('{"summary"', stop_reason="max_tokens"))
    client = AnthropicClient(fake, settings.model_copy(update={"use_batch_api": False}))  # type: ignore[arg-type]

    with pytest.raises(LlmError, match="max_tokens"):
        client.ask(PROMPT, "{}")


def test_the_batch_is_polled_until_it_ends(settings: LLMSettings) -> None:
    fake = FakeAnthropic(_message(_answer("AAA")), statuses=["in_progress", "in_progress"])
    slept: list[float] = []
    client = AnthropicClient(fake, settings, sleep=slept.append)  # type: ignore[arg-type]

    reply = client.ask(PROMPT, "{}")

    assert reply.batch is True
    assert len(slept) == 2
    assert fake.messages.batches.created[0]["custom_id"] == "analisis"
    assert fake.messages.calls == []  # no ha hecho falta la llamada directa


def test_a_slow_batch_is_cancelled_and_repeated_live(settings: LLMSettings) -> None:
    fake = FakeAnthropic(_message(_answer("AAA")), statuses=["in_progress"] * 50)
    patient = settings.model_copy(update={"batch_wait_minutes": 1})
    client = AnthropicClient(fake, patient, sleep=lambda _: None)  # type: ignore[arg-type]

    reply = client.ask(PROMPT, "{}")

    assert fake.messages.batches.cancelled == ["batch_1"]
    assert reply.batch is False
    assert len(fake.messages.calls) == 1


def test_a_failed_batch_says_why(settings: LLMSettings) -> None:
    """Sin el motivo de la API no hay forma de arreglar la petición."""
    detail = "output_config.format.schema: For 'number' type, properties maximum are not supported"
    fake = FakeAnthropic(_message(_answer("AAA")), result=_errored(detail))
    client = AnthropicClient(fake, settings)  # type: ignore[arg-type]

    with pytest.raises(LlmError, match="invalid_request_error"):
        client.ask(PROMPT, "{}")

    with pytest.raises(LlmError, match="properties maximum are not supported"):
        client.ask(PROMPT, "{}")


def test_the_schema_sent_has_no_numeric_bounds() -> None:
    """La API rechaza minimum/maximum; el rango lo sigue comprobando pydantic."""
    sent = json.dumps(json_schema())

    assert "minimum" not in sent
    assert "maximum" not in sent
    assert "De 0 a 1" in sent  # el rango se le dice al modelo por la descripción
    with pytest.raises(ValueError, match="no encaja en el esquema"):
        parse_answer(_answer("AAA").replace('"confidence": 0.6', '"confidence": 1.5'))


# --- paso -------------------------------------------------------------------------------


def _context(cfg: AppConfig, **data: Any) -> StepContext:
    ctx = StepContext(cfg=cfg, region=cfg.regions["americas"], as_of=AS_OF)
    ctx.data.update(data)
    return ctx


def test_the_step_needs_the_screener(cfg_tmp: AppConfig) -> None:
    with pytest.raises(StepError, match="paso screener debe ir antes"):
        Llm().run(_context(cfg_tmp))


def test_the_step_skips_when_disabled(cfg_tmp: AppConfig) -> None:
    off = replace(cfg_tmp, settings=_with_llm(cfg_tmp, enabled=False))

    outcome = Llm().run(_context(off, candidates=_candidates()))

    assert outcome.status is StepStatus.SKIPPED
    assert "llm.enabled" in outcome.message


def test_the_step_skips_without_candidates(cfg_tmp: AppConfig) -> None:
    outcome = Llm().run(_context(cfg_tmp, candidates=_candidates().iloc[0:0]))

    assert outcome.status is StepStatus.SKIPPED
    assert "nada que analizar" in outcome.message


def test_the_step_fails_without_the_api_key(cfg_tmp: AppConfig) -> None:
    with pytest.raises(StepError, match="ANTHROPIC_API_KEY"):
        Llm().run(_context(cfg_tmp, candidates=_candidates()))


def test_the_step_analyses_and_leaves_the_answer_in_the_context(
    cfg_tmp: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-de-prueba")
    with_key = replace(cfg_tmp, env=Env(_env_file=None, ma_data_dir=cfg_tmp.env.ma_data_dir))
    client = FakeClient(_answer("AAA"))
    monkeypatch.setattr("analyzer.steps.llm.step.build_client", lambda settings, api_key: client)
    ctx = _context(with_key, candidates=_candidates(), clusters=_clusters(_cluster()))

    outcome = Llm().run(ctx)

    assert outcome.status is StepStatus.OK
    assert "1 candidato -> 1 confirmar" in outcome.message
    assert isinstance(ctx.data["analysis"], Analysis)
    assert ctx.data["llm"].analysis.tickers == ["AAA"]
    assert PROMPT not in client.asked[0][0]  # el prompt real sale del fichero
    assert "Analista de mercado" in client.asked[0][0]


def _with_llm(cfg: AppConfig, **overrides: Any) -> Any:
    llm = cfg.settings.llm.model_copy(update=overrides)
    return cfg.settings.model_copy(update={"llm": llm})
