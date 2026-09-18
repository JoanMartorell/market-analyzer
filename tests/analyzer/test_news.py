"""Paso news: fuentes RSS y Finnhub, servicio, almacén y paso."""

import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import httpx
import pandas as pd
import pytest

from analyzer.engine import StepContext, StepError, StepStatus
from analyzer.steps.news import News, article_id, run_news
from analyzer.steps.news.sources import (
    Article,
    FinnhubSource,
    RssSource,
    build_source,
    clean_text,
    feed_symbol,
)
from analyzer.storage import NEWS_COLUMNS, NewsStore, connect
from core.config import AppConfig, Env
from core.config import NewsSource as NewsSourceConfig

AS_OF = date(2026, 9, 17)
NOW = datetime(2026, 9, 18, 6, 0, tzinfo=UTC)
SINCE = NOW - timedelta(hours=48)


def _rss(*items: tuple[str, str, str]) -> str:
    """Feed RSS con (título, enlace, fecha RFC 822)."""
    body = "".join(
        f"<item><title>{t}</title><link>{u}</link><pubDate>{d}</pubDate>"
        f"<description>&lt;p&gt;Resumen de {t}&lt;/p&gt;</description></item>"
        for t, u, d in items
    )
    return (
        f'<?xml version="1.0"?><rss version="2.0"><channel><title>x</title>{body}</channel></rss>'
    )


RECENT = "Thu, 17 Sep 2026 20:00:00 GMT"
OLD = "Mon, 01 Sep 2026 10:00:00 GMT"


def _client(handler: object) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def _rss_config(**overrides: object) -> NewsSourceConfig:
    data: dict[str, object] = {
        "kind": "rss",
        "url_template": "https://feeds.example/rss?s={ticker}",
        "per_ticker": True,
        "reliability": 0.6,
        "language": "en",
    }
    data.update(overrides)
    return NewsSourceConfig.model_validate(data)


def _api_config() -> NewsSourceConfig:
    return NewsSourceConfig.model_validate(
        {
            "kind": "api",
            "base_url": "https://finnhub.test/api/v1",
            "api_key_env": "FINNHUB_API_KEY",
            "per_ticker": True,
            "reliability": 0.7,
            "language": "en",
            "rate_limit_per_minute": 60,
        }
    )


# --- fuentes ----------------------------------------------------------------------


def test_clean_text_and_feed_symbol() -> None:
    assert clean_text("<p>Apple &amp; Co  <b>rises</b></p>\n") == "Apple & Co rises"
    assert clean_text(None) == ""
    assert feed_symbol("BRK.B") == "BRK-B"


def test_rss_per_ticker_filters_window_and_tolerates_one_failing_feed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        symbol = request.url.params["s"]
        if symbol == "BRK-B":
            return httpx.Response(404)
        return httpx.Response(
            200,
            text=_rss(
                (f"{symbol} sube", f"https://n.example/{symbol}/1", RECENT),
                (f"{symbol} antigua", f"https://n.example/{symbol}/0", OLD),
            ),
        )

    source = RssSource("yahoo", _rss_config())
    with _client(handler) as http:
        articles = source.fetch(http, ["AAPL", "BRK.B"], SINCE)

    assert [a.ticker for a in articles] == ["AAPL"]
    assert articles[0].title == "AAPL sube"
    assert articles[0].summary == "Resumen de AAPL sube"
    assert articles[0].published_at == datetime(2026, 9, 17, 20, 0, tzinfo=UTC)


def test_rss_raises_when_every_feed_fails() -> None:
    source = RssSource("yahoo", _rss_config())
    with _client(lambda request: httpx.Response(500)) as http:
        with pytest.raises(httpx.HTTPError, match="ningún feed respondió"):
            source.fetch(http, ["AAPL"], SINCE)


def test_rss_global_feed_has_no_ticker() -> None:
    config = _rss_config(url="https://cnbc.example/rss", url_template=None, per_ticker=False)
    source = RssSource("cnbc", config)
    feed = _rss(("Mercado abre al alza", "https://cnbc.example/1", RECENT))
    with _client(lambda request: httpx.Response(200, text=feed)) as http:
        articles = source.fetch(http, ["AAPL"], SINCE)

    assert len(articles) == 1
    assert articles[0].ticker is None


def test_finnhub_parses_payload_and_paces_requests() -> None:
    calls: list[str] = []
    pauses: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.params["symbol"])
        assert request.url.params["token"] == "secret"
        assert request.url.path.endswith("/company-news")
        payload = [
            {"headline": "Beat", "summary": "s", "url": "https://f/1", "datetime": 1789588800},
            {"headline": "Old", "summary": "", "url": "https://f/0", "datetime": 1700000000},
            {"headline": "", "url": "https://f/2", "datetime": 1789588800},
        ]
        return httpx.Response(200, text=json.dumps(payload))

    source = FinnhubSource("finnhub", _api_config(), "secret", sleep=pauses.append)
    with _client(handler) as http:
        articles = source.fetch(http, ["AAPL", "MSFT"], SINCE)

    assert calls == ["AAPL", "MSFT"]
    assert pauses == [1.0]  # 60 por minuto: un segundo entre peticiones
    assert [(a.ticker, a.title) for a in articles] == [("AAPL", "Beat"), ("MSFT", "Beat")]
    assert articles[0].published_at == datetime(2026, 9, 16, 20, 0, tzinfo=UTC)


def test_build_source_needs_api_key(empty_env: Env) -> None:
    with pytest.raises(ValueError, match="necesita FINNHUB_API_KEY"):
        build_source("finnhub", _api_config(), empty_env)
    assert isinstance(build_source("yahoo", _rss_config(), empty_env), RssSource)
    with pytest.raises(ValueError, match="no implementada"):
        build_source("bloomberg", _api_config(), empty_env)


# --- servicio y almacén -------------------------------------------------------------


class FakeSource:
    per_ticker = True

    def __init__(self, name: str, articles: list[Article], *, fail: bool = False) -> None:
        self.name = name
        self.reliability = 0.5
        self.language = "en"
        self._articles = articles
        self._fail = fail
        self.calls: list[list[str]] = []

    def fetch(self, http: httpx.Client, tickers: Sequence[str], since: datetime) -> list[Article]:
        self.calls.append(list(tickers))
        if self._fail:
            raise httpx.ConnectError("boom")
        return list(self._articles)


def _article(ticker: str | None, n: int, *, source: str = "a", hours_ago: float = 1) -> Article:
    return Article(
        source=source,
        ticker=ticker,
        title=f"Titular {n}",
        summary="",
        url=f"https://n/{ticker}/{n}",
        published_at=NOW - timedelta(hours=hours_ago),
    )


@pytest.fixture
def store() -> NewsStore:
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA prod")
    return NewsStore(con, "prod")


def _targets(*tickers: str) -> pd.DataFrame:
    return pd.DataFrame({"ticker": list(tickers), "mic": ["XNAS"] * len(tickers)})


def test_run_news_collects_dedups_and_counts_new(store: NewsStore) -> None:
    a = FakeSource("a", [_article("AAPL", 1), _article("AAPL", 1), _article("MSFT", 2)])
    b = FakeSource(
        "b",
        [_article(None, 3, source="b", hours_ago=2), _article("AAPL", 4, source="b", hours_ago=72)],
    )
    broken = FakeSource("c", [], fail=True)
    with httpx.Client() as http:
        first = run_news(
            _targets("MSFT", "AAPL"),
            AS_OF,
            sources=[a, b, broken],
            store=store,
            http=http,
            lookback_hours=48,
            now=NOW,
        )
        again = run_news(
            _targets("MSFT", "AAPL"),
            AS_OF,
            sources=[a, b, broken],
            store=store,
            http=http,
            lookback_hours=48,
            now=NOW,
        )

    assert a.calls == [["AAPL", "MSFT"], ["AAPL", "MSFT"]]
    assert first.fetched == 3 and first.new == 3
    assert first.summary() == "3 noticias de 2 valores, 3 nuevas (a 3, b 2; c sin respuesta)"
    assert first.sources[2].error == "boom"
    assert list(first.articles.columns) == list(NEWS_COLUMNS)
    by_ticker = dict(
        zip(first.articles["ticker"].fillna("-"), first.articles["mic"].fillna("-"), strict=True)
    )
    assert by_ticker == {"AAPL": "XNAS", "MSFT": "XNAS", "-": "-"}
    assert first.articles["reliability"].tolist() == [0.5, 0.5, 0.5]
    assert again.new == 0

    stored = store.load(since=SINCE, keys=_targets("AAPL"))
    assert stored["ticker"].fillna("-").tolist() == ["AAPL", "-"]  # la global entra
    assert stored["published_at"].dt.tz is not None
    assert store.load(keys=_targets("AAPL"), include_global=False)["ticker"].tolist() == ["AAPL"]
    assert len(store.existing_ids([article_id(_article("AAPL", 1)), "nope"])) == 1


def test_run_news_fails_when_every_source_fails(store: NewsStore) -> None:
    with httpx.Client() as http:
        with pytest.raises(ValueError, match="ninguna fuente de noticias respondió: a: boom"):
            run_news(
                _targets("AAPL"),
                AS_OF,
                sources=[FakeSource("a", [], fail=True)],
                store=store,
                http=http,
                lookback_hours=48,
                now=NOW,
            )


# --- paso -------------------------------------------------------------------------------


@pytest.fixture
def cfg_tmp(cfg: AppConfig, tmp_path: Path) -> AppConfig:
    return replace(cfg, env=Env(_env_file=None, ma_data_dir=tmp_path))


def _ctx(cfg_tmp: AppConfig, candidates: pd.DataFrame) -> StepContext:
    ctx = StepContext(cfg=cfg_tmp, region=cfg_tmp.regions["americas"], as_of=AS_OF)
    ctx.data["candidates"] = candidates
    return ctx


def test_step_skips_without_targets(cfg_tmp: AppConfig) -> None:
    ctx = _ctx(cfg_tmp, pd.DataFrame(columns=["ticker", "mic"]))

    outcome = News().run(ctx)

    assert outcome.status is StepStatus.SKIPPED
    assert ctx.data["articles"].empty


def test_step_fails_when_no_source_can_be_built(
    cfg_tmp: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(sid: str, config: object, env: object) -> object:
        raise ValueError(f"fuente de noticias {sid!r} necesita X en .env")

    monkeypatch.setattr("analyzer.steps.news.step.build_source", refuse)
    with pytest.raises(StepError, match=r"ninguna fuente de noticias disponible: .*finnhub"):
        News().run(_ctx(cfg_tmp, _targets("AAPL")))


def test_step_reports_source_without_key_and_continues(
    cfg_tmp: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = FakeSource("fake", [_article("AAPL", 1, source="fake")])

    def build(sid: str, config: object, env: object) -> object:
        if sid == "finnhub":
            raise ValueError("fuente de noticias 'finnhub' necesita FINNHUB_API_KEY en .env")
        return source

    monkeypatch.setattr("analyzer.steps.news.step.build_source", build)
    outcome = News().run(_ctx(cfg_tmp, _targets("AAPL")))

    assert outcome.status is StepStatus.OK
    assert outcome.message == "1 noticia de 1 valores, 1 nuevas (fake 1, fake 1; finnhub sin clave)"


def test_step_downloads_for_candidates_and_positions(
    cfg_tmp: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = FakeSource("fake", [_article("AAPL", 1, source="fake")])
    monkeypatch.setattr("analyzer.steps.news.step.build_source", lambda sid, c, e: source)
    ctx = _ctx(cfg_tmp, _targets("AAPL"))
    ctx.data["open_positions"] = _targets("NVDA")

    outcome = News().run(ctx)

    assert outcome.status is StepStatus.OK
    assert source.calls[0] == ["AAPL", "NVDA"]
    assert ctx.data["news"].fetched == 1
    assert ctx.data["articles"]["ticker"].tolist() == ["AAPL"]
    with connect(cfg_tmp) as con:
        assert len(NewsStore(con, "prod").load()) == 1
