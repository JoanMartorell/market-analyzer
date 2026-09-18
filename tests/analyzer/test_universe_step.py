"""Paso universe dentro del pipeline, con un fichero de constituyentes temporal."""

from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from analyzer.engine import StepContext, StepError
from analyzer.steps.universe import Universe
from analyzer.universe import write_constituents
from core.config import AppConfig, Env

BUILD_TARGET = "analyzer.steps.universe.step.build_universe"


@pytest.fixture
def cfg_tmp(cfg: AppConfig, tmp_path: Path) -> AppConfig:
    return replace(cfg, env=Env(_env_file=None, ma_data_dir=tmp_path))


def _write(cfg: AppConfig, region_id: str = "americas", meta: dict[str, Any] | None = None) -> Path:
    region = cfg.regions[region_id]
    table = pd.DataFrame(
        {
            "ticker": ["AAPL", "FB", "META", "MYSTERY"],
            "name": ["Apple", "Facebook", "Meta", None],
            "sector": ["IT", None, "Comms", None],
            "sub_industry": [None] * 4,
            "cik": pd.array([320193, None, 1326801, None], dtype="Int64"),
            "mic": ["XNAS", "XNAS", "XNAS", None],
            "currency": ["USD"] * 4,
            "start": pd.to_datetime(
                pd.Series(["1996-01-02", "2013-12-23", "2022-06-09", "2000-01-01"])
            ),
            "end": pd.to_datetime(pd.Series([None, "2022-06-09", None, None])),
            "source": ["test"] * 4,
        }
    )
    return write_constituents(table, cfg.constituents_path(region), meta=meta or {})


def _ctx(
    cfg: AppConfig, day: date, open_markets: list[str] | None, region_id: str = "americas"
) -> StepContext:
    ctx = StepContext(cfg=cfg, region=cfg.regions[region_id], as_of=day)
    if open_markets is not None:
        ctx.data["open_markets"] = open_markets
    return ctx


def test_universe_as_of_date(cfg_tmp: AppConfig) -> None:
    _write(cfg_tmp)
    ctx = _ctx(cfg_tmp, date(2020, 6, 1), ["XNYS", "XNAS"])

    outcome = Universe().run(ctx)

    assert set(ctx.data["universe"]["ticker"]) == {"AAPL", "FB", "MYSTERY"}
    assert "3 valores" in outcome.message


def test_universe_filters_by_open_markets_only_when_mic_known(cfg_tmp: AppConfig) -> None:
    _write(cfg_tmp)
    ctx = _ctx(cfg_tmp, date(2026, 9, 16), ["XNYS"])  # solo NYSE abrió

    Universe().run(ctx)

    # los Nasdaq se van; MYSTERY sin mic se conserva
    assert set(ctx.data["universe"]["ticker"]) == {"MYSTERY"}


def _boom(*_args: object, **_kwargs: object) -> None:
    raise httpx.ConnectError("sin red")


def test_universe_builds_file_when_missing(
    cfg_tmp: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_build(region: Any, path: Path, *, user_agent: str | None) -> None:
        calls.append(region.id)
        _write(cfg_tmp)

    monkeypatch.setattr(BUILD_TARGET, fake_build)
    ctx = _ctx(cfg_tmp, date(2026, 9, 16), None)

    outcome = Universe().run(ctx)

    assert calls == ["americas"]
    assert "construido ahora" in outcome.message


def test_universe_build_failure_fails_closed(
    cfg_tmp: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(BUILD_TARGET, _boom)

    with pytest.raises(StepError, match="no se pudo construir"):
        Universe().run(_ctx(cfg_tmp, date(2026, 9, 16), None))


def test_history_universe_is_not_rebuilt_when_present(
    cfg_tmp: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(cfg_tmp, meta={"built_at": "2020-01-01 00:00:00+00:00"})  # antiguo, da igual
    monkeypatch.setattr(BUILD_TARGET, _boom)  # si se llamara, fallaría

    outcome = Universe().run(_ctx(cfg_tmp, date(2026, 9, 16), None))

    assert "construido ahora" not in outcome.message


def test_snapshot_universe_is_refreshed_once_a_day(
    cfg_tmp: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_build(region: Any, path: Path, *, user_agent: str | None) -> None:
        calls.append(region.id)
        _write(cfg_tmp, "europe", meta={"built_at": str(datetime.now(UTC))})

    monkeypatch.setattr(BUILD_TARGET, fake_build)
    _write(cfg_tmp, "europe", meta={"built_at": "2026-01-01 00:00:00+00:00"})  # de ayer o antes

    Universe().run(_ctx(cfg_tmp, date(2026, 9, 17), None, "europe"))
    assert calls == ["europe"]

    Universe().run(_ctx(cfg_tmp, date(2026, 9, 17), None, "europe"))
    assert calls == ["europe"]  # ya construido hoy: no se repite


def test_universe_empty_fails_closed(cfg_tmp: AppConfig) -> None:
    _write(cfg_tmp)
    ctx = _ctx(cfg_tmp, date(1990, 1, 1), None)

    with pytest.raises(StepError, match="vacío"):
        Universe().run(ctx)
