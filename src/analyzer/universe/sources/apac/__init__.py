"""Asia-Pacífico: unión de cuatro índices nacionales según Wikipedia.

Nikkei 225 (Tokio), Hang Seng (Hong Kong), S&P/ASX 200 (Sídney) y
KOSPI 200 (Corea). Sin histórico gratuito: ``history=False``.
"""

from analyzer.universe.base import UniverseSpec
from analyzer.universe.sources.apac.tables import ASX200, HSI, KOSPI200, NIKKEI225
from analyzer.universe.sources.wikipedia_table import WikipediaSnapshot

APAC = UniverseSpec(
    id="apac_large_cap",
    currency="USD",  # no hay divisa común; cada fila lleva la suya
    source=WikipediaSnapshot("wikipedia:apac", [NIKKEI225, HSI, ASX200, KOSPI200]),
    history=False,
)

__all__ = ["APAC", "ASX200", "HSI", "KOSPI200", "NIKKEI225"]
