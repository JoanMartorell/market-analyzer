"""Cascada de deduplicación: hash exacto, MinHash y embeddings.

Tres niveles de coste creciente, y cada uno solo mira lo que el anterior no
resolvió:

1. **hash** del texto normalizado. La misma nota republicada palabra por
   palabra. No cuesta nada y se lleva la mayor parte.
2. **MinHash + LSH** sobre shingles de palabras. La misma nota con el titular
   retocado o un párrafo de más. Sin modelos, solo aritmética.
3. **embeddings** coseno. Dos redacciones distintas del mismo hecho. Es el
   único nivel que necesita el extra ``nlp``; sin él la cascada se queda en
   los dos primeros y no falla.

Se agrupa por valor: la misma noticia que llega por el feed de dos valores
son dos temas, uno para cada uno, porque el análisis es por valor. Los
artículos sin valor (fuentes globales) forman su propio grupo. Dentro de un
grupo, dos artículos separados por más de la ventana son eventos distintos
aunque el texto coincida: una compañía repite noticia cada trimestre.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
from datasketch import MinHash, MinHashLSH

from analyzer.steps.dedup_sentiment.models import Embedder
from analyzer.steps.dedup_sentiment.text import article_text, shingles, text_hash

EXACT = "exacto"
MINHASH = "minhash"
EMBEDDING = "embeddings"
SINGLE = "unico"
LEVELS = (EXACT, MINHASH, EMBEDDING)
GLOBAL = "\x00global"  # grupo de los artículos sin valor, aparte de cualquier ticker
PAIR = 2  # mínimo de artículos sueltos para que merezca la pena embeber


@dataclass(frozen=True)
class Article:
    """Lo único que la cascada necesita de un artículo."""

    id: str
    group: str  # ticker, o GLOBAL si la fuente no habla de un valor
    text: str
    digest: str  # hash del texto normalizado
    published_at: datetime


@dataclass(frozen=True)
class Grouping:
    """Resultado de la cascada: a qué tema va cada artículo y por qué nivel entró."""

    labels: dict[str, int]  # id -> tema
    methods: dict[str, str]  # id -> nivel que lo unió al tema, ``unico`` si no se unió a nadie
    merges: dict[str, int]  # nivel -> artículos absorbidos

    @property
    def clusters(self) -> int:
        return len(set(self.labels.values()))

    def level_of(self, ids: Sequence[str]) -> str:
        """Nivel más profundo usado dentro de un tema."""
        used = {self.methods[i] for i in ids} - {SINGLE}
        return max(used, key=LEVELS.index) if used else SINGLE


Link = Callable[[str, str, str], None]


def cluster_articles(
    frame: pd.DataFrame,
    *,
    shingle_size: int,
    num_perm: int,
    minhash_threshold: float,
    window_hours: int,
    embedder: Embedder | None = None,
    cosine_threshold: float = 1.0,
) -> Grouping:
    """Agrupa en temas los artículos de ``frame`` (columnas de ``prod.news``)."""
    articles = to_articles(frame)
    ids = [a.id for a in articles]
    union = _Union(ids)
    merges: Counter[str] = Counter()
    methods: dict[str, str] = dict.fromkeys(ids, SINGLE)
    window = timedelta(hours=window_hours)

    def link(left: str, right: str, level: str) -> None:
        if union.union(left, right):
            merges[level] += 1
            methods[left] = _deeper(methods[left], level)
            methods[right] = _deeper(methods[right], level)

    for group in _by_group(articles):
        _merge_exact(group, window, link)
        _merge_minhash(
            group,
            window,
            link,
            shingle_size=shingle_size,
            num_perm=num_perm,
            threshold=minhash_threshold,
        )
        if embedder is not None:
            _merge_embeddings(group, window, link, union, embedder, cosine_threshold)

    roots: dict[str, int] = {}
    labels = {i: roots.setdefault(union.find(i), len(roots)) for i in ids}
    return Grouping(labels=labels, methods=methods, merges=dict(merges))


def to_articles(frame: pd.DataFrame) -> list[Article]:
    """Filas de ``prod.news`` como artículos comparables, ordenados por grupo y fecha."""
    articles = []
    for row in frame.to_dict("records"):
        text = article_text(row["title"], row["summary"])
        ticker = row["ticker"]
        articles.append(
            Article(
                id=str(row["id"]),
                group=GLOBAL if pd.isna(ticker) else str(ticker),
                text=text,
                digest=text_hash(text),
                published_at=pd.Timestamp(row["published_at"]).to_pydatetime(),
            )
        )
    return sorted(articles, key=lambda a: (a.group, a.published_at, a.id))


# --- niveles ---------------------------------------------------------------------------


def _merge_exact(group: Sequence[Article], window: timedelta, link: Link) -> None:
    """Nivel 0: mismo texto normalizado. Se encadenan por fecha dentro de la ventana."""
    by_digest: dict[str, Article] = {}
    for article in group:  # ya vienen ordenados por fecha
        previous = by_digest.get(article.digest)
        if previous is not None and article.published_at - previous.published_at <= window:
            link(previous.id, article.id, EXACT)
        by_digest[article.digest] = article


def _merge_minhash(
    group: Sequence[Article],
    window: timedelta,
    link: Link,
    *,
    shingle_size: int,
    num_perm: int,
    threshold: float,
) -> None:
    """Nivel 1: Jaccard de shingles. LSH descarta el grueso y el par se confirma exacto."""
    signatures: dict[str, MinHash] = {}
    times = {a.id: a.published_at for a in group}
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for article in group:
        pieces = shingles(article.text, shingle_size)
        if not pieces:
            continue
        signature = MinHash(num_perm=num_perm)
        for piece in sorted(pieces):
            signature.update(piece.encode("utf-8"))
        signatures[article.id] = signature
        lsh.insert(article.id, signature)

    for left, signature in signatures.items():
        for right in sorted(str(key) for key in lsh.query(signature)):
            if right <= left:  # cada par, una vez
                continue
            if signature.jaccard(signatures[right]) < threshold:  # el LSH es aproximado
                continue
            if abs(times[right] - times[left]) <= window:
                link(left, right, MINHASH)


def _merge_embeddings(
    group: Sequence[Article],
    window: timedelta,
    link: Link,
    union: _Union,
    embedder: Embedder,
    threshold: float,
) -> None:
    """Nivel 2: coseno entre embeddings, solo entre los que siguen sueltos."""
    pending = [a for a in group if union.size_of(a.id) == 1]
    if len(pending) < PAIR:
        return
    vectors = embedder.encode([a.text for a in pending])
    similarity = vectors @ vectors.T
    for i, left in enumerate(pending):
        for j in range(i + 1, len(pending)):
            right = pending[j]
            if similarity[i][j] < threshold:
                continue
            if abs(right.published_at - left.published_at) <= window:
                link(left.id, right.id, EMBEDDING)


# --- utilidades ------------------------------------------------------------------------


def _by_group(articles: Sequence[Article]) -> Iterator[list[Article]]:
    current: list[Article] = []
    for article in articles:  # ordenados por grupo
        if current and article.group != current[0].group:
            yield current
            current = []
        current.append(article)
    if current:
        yield current


def _deeper(current: str, level: str) -> str:
    return level if current == SINGLE else max(current, level, key=LEVELS.index)


class _Union:
    """Union-find de raíz determinista: manda el id menor, así el resultado no depende del orden."""

    def __init__(self, items: Sequence[str]) -> None:
        self._parent = {item: item for item in items}
        self._size = dict.fromkeys(items, 1)

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:  # compresión de caminos
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: str, right: str) -> bool:
        """Une dos temas; ``False`` si ya eran el mismo."""
        low, high = sorted((self.find(left), self.find(right)))
        if low == high:
            return False
        self._parent[high] = low
        self._size[low] += self._size[high]
        return True

    def size_of(self, item: str) -> int:
        return self._size[self.find(item)]
