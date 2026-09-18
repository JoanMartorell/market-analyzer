"""Europa: STOXX Europe 600 según la tabla de componentes de Wikipedia.

Sin histórico gratuito: ``history=False``, el fichero acumula cambios desde
el primer build. La bolsa y la divisa se derivan del país de la tabla.
"""

from analyzer.universe.base import UniverseSpec
from analyzer.universe.sources.europe.stoxx600 import STOXX600
from analyzer.universe.sources.wikipedia_table import WikipediaSnapshot

EUROPE = UniverseSpec(
    id="stoxx600",
    currency="EUR",
    source=WikipediaSnapshot("wikipedia:stoxx600", [STOXX600]),
    history=False,
)

__all__ = ["EUROPE", "STOXX600"]
