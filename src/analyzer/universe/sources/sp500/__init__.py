"""S&P 500 (Estados Unidos).

Historia desde 1996 vía el dataset de GitHub; nombre, sector y CIK desde la
lista actual de Wikipedia; bolsa (MIC) desde la SEC.
"""

from analyzer.universe.base import UniverseSpec
from analyzer.universe.sources.sp500.github import GithubIntervals
from analyzer.universe.sources.sp500.sec import SecExchanges
from analyzer.universe.sources.sp500.wikipedia import WikipediaCurrentList

SP500 = UniverseSpec(
    id="sp500",
    currency="USD",
    source=GithubIntervals(),
    enrichers=(WikipediaCurrentList(), SecExchanges()),
)

__all__ = ["SP500", "GithubIntervals", "SecExchanges", "WikipediaCurrentList"]
