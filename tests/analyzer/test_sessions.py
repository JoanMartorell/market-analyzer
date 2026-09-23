"""Última sesión cerrada por región, con calendarios reales."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from analyzer.sessions import last_closed_session, next_session
from core.config import AppConfig

MADRID = ZoneInfo("Europe/Madrid")


def _at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=MADRID)


def test_americas_before_us_close_is_previous_session(cfg: AppConfig) -> None:
    # jueves 18:00 Madrid: Wall Street cierra a las 22:00, la última cerrada es el miércoles
    assert last_closed_session(cfg.regions["americas"], _at(date(2026, 9, 17), 18)) == date(
        2026, 9, 16
    )


def test_americas_after_us_close_is_today(cfg: AppConfig) -> None:
    assert last_closed_session(cfg.regions["americas"], _at(date(2026, 9, 17), 23)) == date(
        2026, 9, 17
    )


def test_monday_morning_is_friday(cfg: AppConfig) -> None:
    assert last_closed_session(cfg.regions["americas"], _at(date(2026, 9, 21), 6, 30)) == date(
        2026, 9, 18
    )


def test_europe_waits_for_every_market_to_close(cfg: AppConfig) -> None:
    europe = cfg.regions["europe"]
    # 17:00: Oslo (16:20) y Varsovia (17:00) ya cerraron, Fráncfort (17:30) no
    assert last_closed_session(europe, _at(date(2026, 9, 17), 17)) == date(2026, 9, 16)
    assert last_closed_session(europe, _at(date(2026, 9, 17), 17, 25)) == date(2026, 9, 16)
    assert last_closed_session(europe, _at(date(2026, 9, 17), 18)) == date(2026, 9, 17)


def test_naive_datetime_is_rejected(cfg: AppConfig) -> None:
    with pytest.raises(ValueError, match="zona horaria"):
        last_closed_session(cfg.regions["americas"], datetime(2026, 9, 17, 18))


# --- siguiente sesión ---------------------------------------------------------------


def test_next_session_is_the_following_trading_day(cfg: AppConfig) -> None:
    americas = cfg.regions["americas"]

    assert next_session(americas, date(2026, 9, 17), "XNYS") == date(2026, 9, 18)
    assert next_session(americas, date(2026, 9, 18), "XNYS") == date(2026, 9, 21)  # fin de semana
    assert next_session(americas, date(2026, 12, 24), "XNAS") == date(2026, 12, 28)  # Navidad


def test_next_session_without_a_known_market_takes_the_earliest(cfg: AppConfig) -> None:
    americas = cfg.regions["americas"]

    assert next_session(americas, date(2026, 9, 18)) == date(2026, 9, 21)
    assert next_session(americas, date(2026, 9, 18), "UNKNOWN") == date(2026, 9, 21)


def test_latam_waits_for_mexico_to_close(cfg: AppConfig) -> None:
    latam = cfg.regions["latam"]
    # jueves 22:30 Madrid: B3 ya cerró (22:00), la BMV cierra a las 23:00
    assert last_closed_session(latam, _at(date(2026, 9, 17), 22, 30)) == date(2026, 9, 16)
    # viernes a las 07:00, la hora del ciclo: la sesión del jueves ya vale
    assert last_closed_session(latam, _at(date(2026, 9, 18), 7)) == date(2026, 9, 17)
