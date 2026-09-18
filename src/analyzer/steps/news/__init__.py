"""Paso 7: descarga noticias solo de candidatos y posiciones abiertas."""

from analyzer.steps.news.service import NewsReport, SourceResult, article_id, run_news, to_frame
from analyzer.steps.news.step import News

__all__ = ["News", "NewsReport", "SourceResult", "article_id", "run_news", "to_frame"]
