"""Carga y validación de la configuración: .env + config/*.yaml.

Uso:

    from core.config import load_config

    cfg = load_config()
    for region in cfg.enabled_regions():
        ...
"""

from core.config.env import Env
from core.config.loader import AppConfig, ConfigError, load_config
from core.config.schema import (
    Condition,
    DedupSettings,
    LlmPricing,
    LLMSettings,
    Market,
    NewsSettings,
    NewsSource,
    Provider,
    Region,
    RepresentativeWeights,
    Rule,
    Settings,
    Sources,
)

__all__ = [
    "AppConfig",
    "Condition",
    "ConfigError",
    "DedupSettings",
    "Env",
    "LLMSettings",
    "LlmPricing",
    "Market",
    "NewsSettings",
    "NewsSource",
    "Provider",
    "Region",
    "RepresentativeWeights",
    "Rule",
    "Settings",
    "Sources",
    "load_config",
]
