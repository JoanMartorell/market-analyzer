"""Qué sesión procesar: la última que ya ha cerrado en toda la región.

Lanzar el pipeline sobre una barra diaria incompleta genera señales que en
el backtest nunca existieron. Por eso ``as_of`` nunca es "hoy" a secas:
es la última sesión cuyo cierre es anterior al instante de lanzamiento en
todos los mercados de la región que cotizaron ese día. Con Oslo cerrando a
las 16:20 y Fráncfort a las 17:30, a las 17:00 la sesión de hoy aún no vale.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import exchange_calendars as xcals
import pandas as pd

from core.config import Region

_WINDOW = timedelta(days=15)


def last_closed_session(region: Region, now: datetime) -> date | None:
    """Última sesión cerrada en todos los mercados de la región que la tuvieron.

    ``now`` debe llevar zona horaria. Devuelve ``None`` si ningún mercado tuvo
    sesión en la ventana de búsqueda, lo que solo pasa con un calendario roto.
    """
    if now.tzinfo is None:
        raise ValueError("now debe llevar zona horaria")

    now_ts = pd.Timestamp(now)
    start, end = (now - _WINDOW).date(), (now + _WINDOW).date()
    calendars = [
        xcals.get_calendar(market.calendar, start=str(start), end=str(end))
        for market in region.markets
    ]

    candidates: list[date] = sorted(
        {session.date() for cal in calendars for session in cal.sessions}, reverse=True
    )
    for day in candidates:
        session = pd.Timestamp(day)
        if all(
            not cal.is_session(session) or cal.session_close(session) <= now_ts for cal in calendars
        ):
            return day
    return None


def next_session(region: Region, after: date, mic: str | None = None) -> date | None:
    """Primera sesión posterior a ``after`` en el mercado ``mic`` de la región.

    Es cuando se ejecuta una señal del día ``after`` (``execute_at: next_open``).
    Sin ``mic``, o con uno que no es de la región (universo sin MIC), vale la
    primera sesión de cualquiera de sus mercados. ``None`` solo con un
    calendario roto: no hay bolsa que cierre quince días seguidos.
    """
    markets = [m for m in region.markets if m.mic == mic] or list(region.markets)
    start, end = after + timedelta(days=1), after + _WINDOW
    found: list[date] = []
    for market in markets:
        calendar = xcals.get_calendar(market.calendar, start=str(start), end=str(end))
        if len(calendar.sessions):
            found.append(calendar.sessions[0].date())
    return min(found, default=None)
