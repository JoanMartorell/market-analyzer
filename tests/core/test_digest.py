"""El mensaje del día: listas de comprar, vender y vigilar con un motivo corto."""

from datetime import date
from typing import Any

import pandas as pd

from analyzer.engine import PipelineResult, StepReport, StepStatus
from analyzer.steps.llm import Analysis, LlmReport, TokenUsage, Verdict
from analyzer.steps.persist import build_signals
from analyzer.steps.screener.service import CANDIDATE_COLUMNS
from core.digest import WHY_LIMIT, _brief, build_digest

AS_OF = date(2026, 9, 17)
NEXT = date(2026, 9, 18)
LONG_WHY = (
    "El backlog de 170.000 millones sostiene la tendencia y los temas no la contradicen. "
    "La compra de 11.750 millones no cierra hasta 2027."
)


def _candidate(ticker: str, direction: str = "long", score: float = 0.8) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "mic": "XNYS",
        "rule_id": "pullback",
        "direction": direction,
        "score": score,
        "conditions_met": "rsi_14 < 35; close > sma_200",
        "close": 101.25,
        "sector": "Industrials",
        "rule_hash": "regla-abc",
    }


def _signals(candidates: list[dict[str, Any]], analysis: Analysis | None) -> pd.DataFrame:
    frame = pd.DataFrame(candidates, columns=[*CANDIDATE_COLUMNS])
    panel = frame.loc[:, ["ticker", "mic"]].assign(currency="USD")
    return build_signals(
        frame,
        analysis,
        region_id="americas",
        as_of=AS_OF,
        snapshot_hash="a" * 64,
        execute_on={"XNYS": NEXT},
        model="claude-sonnet-5",
        panel=panel,
    )


def _verdict(
    ticker: str, name: str, confidence: float, why: str = "Nada en contra.", headline: str = ""
) -> Verdict:
    return Verdict(
        ticker=ticker,
        verdict=name,
        confidence=confidence,
        headline=headline,
        rationale=why,
        risks=[],
    )


def _result(*reports: StepReport) -> PipelineResult:
    return PipelineResult(region_id="americas", as_of=AS_OF, reports=list(reports))


def _llm(analysis: Analysis, *, reused: bool = False) -> LlmReport:
    return LlmReport(
        region_id="americas",
        as_of=AS_OF,
        model="claude-sonnet-5",
        candidates=4,
        usage=TokenUsage(),
        cost_eur=0.0783,
        month_cost_eur=0.08,
        budget_eur=25.0,
        batch=True,
        reused=reused,
        analysis=analysis,
    )


def test_the_digest_lists_buy_sell_watch_and_discarded() -> None:
    analysis = Analysis(
        summary="Día tranquilo.",
        verdicts=[
            _verdict(
                "PCAR", "confirmar", 0.8, LONG_WHY, headline="Backlog récord sostiene la tendencia"
            ),
            _verdict("RTX", "confirmar", 0.75),
            _verdict("SHRT", "confirmar", 0.6, "Rompe soporte."),
            _verdict("GE", "vigilar", 0.55, "Resultados el jueves."),
            _verdict("BAD", "descartar", 0.9, "Profit warning."),
        ],
    )
    candidates = [
        _candidate("RTX"),
        _candidate("PCAR"),
        _candidate("SHRT", direction="short"),
        _candidate("GE"),
        _candidate("BAD"),
    ]
    signals = _signals(candidates, analysis)
    result = _result(
        StepReport("persist", StepStatus.OK, "5 señales", 0.2),
        StepReport("reconcile", StepStatus.OK, "3 señales del 2026-09-16: 3 conciliadas", 0.1),
    )

    message = build_digest(
        result, {"signals": signals, "analysis": analysis, "llm": _llm(analysis)}
    )

    assert message.subject == "[americas] 2026-09-17: 2 comprar, 1 vender, 1 vigilar, 1 descartar"
    assert message.body.splitlines() == [
        "5 señales del 2026-09-17, a ejecutar el 2026-09-18",
        "",
        "COMPRAR",
        "• PCAR · 101.25 USD · 80% · Backlog récord sostiene la tendencia",
        "• RTX · 101.25 USD · 75% · Nada en contra.",  # sin headline, la primera frase del motivo
        "",
        "VENDER",
        "• SHRT · 101.25 USD · 60% · Rompe soporte.",
        "",
        "VIGILAR",
        "• GE (compra) · 101.25 USD · 55% · Resultados el jueves.",
        "",
        "Descartadas por el modelo: BAD",
        "",
        "Lectura del día: Día tranquilo.",
        "",
        "Ayer: 3 señales del 2026-09-16: 3 conciliadas",
        "ciclo 0 s · modelo 0.08 EUR (0.08 de 25 EUR este mes)",
    ]


def test_without_the_model_the_rules_speak() -> None:
    signals = _signals([_candidate("AAA", score=0.7), _candidate("BBB", score=0.9)], None)

    message = build_digest(_result(), {"signals": signals, "analysis": None})

    assert message.subject == "[americas] 2026-09-17: 2 candidatos sin veredicto"
    body = message.body
    assert "COMPRAR (candidatos de las reglas, sin veredicto del modelo)" in body
    assert body.index("• BBB") < body.index("• AAA")  # por puntuación
    assert "score 0.90 · rsi_14 < 35" in body  # la primera condición como motivo
    assert "Lectura del día" not in body


def test_a_ticker_flagged_by_two_rules_appears_once() -> None:
    analysis = Analysis(summary="", verdicts=[_verdict("AAA", "confirmar", 0.7)])
    two = [_candidate("AAA"), {**_candidate("AAA"), "rule_id": "breakout"}]
    signals = _signals(two, analysis)

    message = build_digest(_result(), {"signals": signals, "analysis": analysis})

    assert message.body.count("• AAA") == 1
    assert message.subject.endswith("1 comprar")


def test_no_signals_says_so() -> None:
    signals = _signals([], None).iloc[0:0]
    message = build_digest(_result(), {"signals": signals, "analysis": None})

    assert message.subject == "[americas] 2026-09-17: sin señales"
    assert message.body.startswith("Sin señales el 2026-09-17.")


def test_a_reused_answer_costs_nothing_in_the_footer() -> None:
    analysis = Analysis(summary="", verdicts=[_verdict("AAA", "confirmar", 0.7)])
    signals = _signals([_candidate("AAA")], analysis)
    llm = _llm(analysis, reused=True)

    message = build_digest(_result(), {"signals": signals, "analysis": analysis, "llm": llm})

    assert "modelo respuesta reutilizada (0.08 de 25 EUR este mes)" in message.body


def test_without_signals_in_the_context_the_step_table_is_sent() -> None:
    result = _result(StepReport("calendar_gate", StepStatus.OK, "sesión en XNYS", 0.1))
    message = build_digest(result, {})

    assert message.subject == "[americas] ciclo 2026-09-17"
    assert message.body == result.summary()


def test_brief_keeps_the_first_sentence_within_the_limit() -> None:
    assert _brief("Una frase. Otra frase.") == "Una frase."
    assert _brief("Sin punto final") == "Sin punto final"
    assert _brief("Primera parte; segunda parte.") == "Primera parte"
    long = "palabra " * 40
    cut = _brief(long)
    assert len(cut) <= WHY_LIMIT + 1 and cut.endswith("…")
