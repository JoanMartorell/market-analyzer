"""Configuración de logging con structlog.

Salida legible en consola. Cuando el pipeline corra bajo Prefect se cambia
el renderer a JSON aquí y en ningún otro sitio.
"""

import logging
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    numeric = logging.getLevelNamesMapping()[level.upper()]
    logging.basicConfig(level=numeric, stream=sys.stderr, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # una línea por petición es ruido
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # sus fallos los cuenta ingest
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),  # hora local
            # Traceback plano: el de rich vuelca las variables locales (DataFrames
            # enteros) y convierte un error de una línea en 170 KB de salida.
            structlog.dev.ConsoleRenderer(exception_formatter=structlog.dev.plain_traceback),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),  # logs a stderr,
        cache_logger_on_first_use=True,  # el informe a stdout
    )
