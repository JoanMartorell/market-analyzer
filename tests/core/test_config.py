"""Tests del cargador de configuración contra los YAML reales del repositorio."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.config import ConfigError, Env, Rule, load_config
from core.config.schema import RegionProviders

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"


def test_repo_config_loads(empty_env: Env) -> None:
    cfg = load_config(CONFIG_DIR, env=empty_env)

    assert cfg.root == ROOT
    assert cfg.settings.base_currency == "EUR"
    assert set(cfg.regions) == {"americas", "europe", "apac"}
    assert "americas" in [r.id for r in cfg.enabled_regions()]
    assert set(cfg.rules) == {"oversold_uptrend"}
    assert cfg.data_dir == (ROOT / "data").resolve()


def test_rule_carries_source_hash(empty_env: Env) -> None:
    cfg = load_config(CONFIG_DIR, env=empty_env)
    rule = cfg.rules["oversold_uptrend"]

    assert len(rule.source_hash) == 64
    assert sum(c.weight for c in rule.conditions) == pytest.approx(1.0)


def test_missing_env_vars_only_for_enabled_regions(empty_env: Env) -> None:
    loaded = load_config(CONFIG_DIR, env=empty_env)
    # Solo Américas activa, independientemente de cómo esté el YAML en este momento.
    regions = {
        rid: region.model_copy(update={"enabled": rid == "americas"})
        for rid, region in loaded.regions.items()
    }
    cfg = replace(loaded, regions=regions)
    missing = cfg.missing_env_vars()

    assert missing["llm"] == ["ANTHROPIC_API_KEY"]
    assert missing["news:finnhub"] == ["FINNHUB_API_KEY"]
    assert missing["provider:sec_edgar"] == ["SEC_EDGAR_USER_AGENT"]
    # eodhd solo lo usan Europa y APAC, que aquí están apagadas
    assert "provider:eodhd" not in missing


def test_extra_requirements_are_merged(empty_env: Env) -> None:
    cfg = load_config(CONFIG_DIR, env=empty_env)
    missing = cfg.missing_env_vars(extra={"delivery:telegram": ["TELEGRAM_BOT_TOKEN"]})

    assert missing["delivery:telegram"] == ["TELEGRAM_BOT_TOKEN"]
    assert "delivery:telegram" not in cfg.missing_env_vars()


def test_env_get_treats_empty_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINNHUB_API_KEY", "")
    monkeypatch.setenv("FRED_API_KEY", "abc")
    env = Env(_env_file=None)

    assert env.get("FINNHUB_API_KEY") is None
    assert env.has("FRED_API_KEY")
    assert env.get("FRED_API_KEY") == "abc"


def _rule(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "version": 3,
        "id": "demo",
        "window": 5,
        "conditions": [
            {"expr": "rsi_14 < 30", "weight": 0.5},
            {"expr": "close > sma_200", "weight": 0.5},
        ],
        "output": {"score_threshold": 0.7},
    }
    base.update(overrides)
    return base


def test_rule_weights_must_sum_to_one() -> None:
    bad = _rule(conditions=[{"expr": "rsi_14 < 30", "weight": 0.4}])
    with pytest.raises(ValueError, match="pesos deben sumar"):
        Rule.model_validate(bad)


def test_rule_rejects_syntax_errors() -> None:
    bad = _rule(conditions=[{"expr": "rsi_14 <", "weight": 1.0}])
    with pytest.raises(ValueError, match="expresión inválida"):
        Rule.model_validate(bad)


def test_rule_rejects_unknown_keys() -> None:
    bad = _rule(ventana=5)
    with pytest.raises(ValueError, match="ventana"):
        Rule.model_validate(bad)


def test_missing_config_dir_is_config_error(empty_env: Env, tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no encontrado"):
        load_config(tmp_path / "nope", env=empty_env)


def test_universe_files_live_in_per_region_subfolders(empty_env: Env) -> None:
    cfg = load_config(CONFIG_DIR, env=empty_env)

    for region in cfg.regions.values():
        path = cfg.constituents_path(region)
        assert path.parent == cfg.data_dir / "universe" / region.id
        assert path.name == region.universe.constituents_file.name


def test_prices_fallback_must_be_a_different_provider() -> None:
    with pytest.raises(ValidationError, match="prices_fallback"):
        RegionProviders(prices="yfinance", prices_fallback="yfinance")
    assert RegionProviders(prices="yfinance", prices_fallback="eodhd").prices_fallback == "eodhd"


def test_fallback_provider_needs_its_credentials(empty_env: Env) -> None:
    cfg = load_config(CONFIG_DIR, env=empty_env)
    missing = cfg.missing_env_vars()

    # Europa y APAC rescatan con eodhd, así que su clave cuenta como requerida.
    assert missing["provider:eodhd"] == ["EODHD_API_KEY"]
