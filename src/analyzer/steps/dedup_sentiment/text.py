"""Normalización del texto de un artículo: lo que ven el hash y los shingles.

Dos teletipos que publican la misma nota cambian mayúsculas, comillas y
espacios; el contenido es el mismo. Se normaliza a minúsculas sin acentos y
sin puntuación para que esas diferencias no cuenten como noticia distinta.
El texto original no se toca: esto solo alimenta la comparación.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_NON_WORD = re.compile(r"[^0-9a-z]+")


def article_text(title: object, summary: object) -> str:
    """Título y resumen en un solo texto, que es la unidad que se compara."""
    parts = [str(p).strip() for p in (title, summary) if isinstance(p, str) and p.strip()]
    return ". ".join(parts)


def normalize(text: str) -> str:
    """Minúsculas, sin acentos, sin puntuación y con un solo espacio entre palabras."""
    plain = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    return _NON_WORD.sub(" ", plain).strip()


def text_hash(text: str) -> str:
    """sha256 del texto normalizado: igualdad exacta, nivel 0 de la cascada."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def shingles(text: str, size: int) -> frozenset[str]:
    """Grupos de ``size`` palabras consecutivas del texto normalizado.

    Un texto más corto que la ventana es un único shingle: comparar dos
    titulares de tres palabras palabra a palabra da parecidos falsos.
    """
    words = normalize(text).split()
    if not words:
        return frozenset()
    if len(words) <= size:
        return frozenset({" ".join(words)})
    return frozenset(" ".join(words[i : i + size]) for i in range(len(words) - size + 1))
