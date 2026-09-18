"""Variables de entorno: un comentario en línea tras un valor vacío no es un valor."""

from pathlib import Path

from core.config import Env


def test_env_ignores_inline_comment_after_empty_value(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "FINNHUB_API_KEY=   # noticias por ticker\nFRED_API_KEY=abc123   # macro\nFMP_API_KEY=\n",
        encoding="utf-8",
    )

    env = Env(_env_file=env_file)

    assert env.get("FINNHUB_API_KEY") is None
    assert not env.has("FINNHUB_API_KEY")
    assert env.get("FRED_API_KEY") == "abc123"
    assert env.get("FMP_API_KEY") is None
