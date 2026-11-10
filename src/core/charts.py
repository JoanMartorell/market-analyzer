"""El gráfico de una operación: las velas, la figura encontrada y sus niveles.

Es la imagen que acompaña al mensaje de cada operación recomendada. Enseña
lo que el detector vio y nada más: los pivotes que forman el patrón, la
línea que el cierre ha roto y los tres niveles (entrada, stop, objetivo).
Sin patrón (una regla de indicadores) quedan las velas solas.

Decisiones de dibujo:

- Las velas van en gris neutro, claras las alcistas y oscuras las bajistas.
  El color se reserva para lo que hay que mirar: la figura en azul, el stop
  en rojo y el objetivo en verde, cada nivel con su nombre y su precio
  escritos al lado para que no dependa de distinguir colores.
- Los precios son ajustados y reescalados para que la última vela esté a su
  precio real: un split o un dividendo no rompe la figura, y los niveles
  coinciden con los del mensaje.
- El eje horizontal son sesiones, no días de calendario: sin huecos de fin
  de semana que deformen la figura.

Se dibuja con la API de objetos de matplotlib (``Figure`` + Agg), sin
``pyplot``: no hay estado global ni hace falta pantalla.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from analyzer.patterns import PatternMatch, adjusted_ohlc

SESSIONS = 110  # velas que se enseñan si la figura no pide más
LEFT_MARGIN = 12  # velas a la izquierda del primer pivote
RIGHT_MARGIN = 16  # hueco a la derecha para rotular los niveles
SIZE_INCHES = (8.0, 4.5)
DPI = 150  # 1200 x 675 px: nítido en el móvil sin pesar

SURFACE = "#1a1a19"
INK = "#f0efec"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"
CANDLE_UP = "#c3c2b7"
CANDLE_DOWN = "#6f6e69"
FIGURE = "#3987e5"  # la figura y su línea de ruptura
STOP = "#e66767"
TARGET = "#0ca30c"


def render_chart(
    bars: pd.DataFrame,
    *,
    title: str,
    subtitle: str = "",
    match: PatternMatch | None = None,
    sessions: int = SESSIONS,
) -> bytes:
    """PNG con las últimas velas de ``bars`` (un valor, ordenado por fecha) y la figura.

    ``match`` es un hallazgo de ``find_patterns`` sobre esas mismas velas:
    sus índices son posiciones dentro de ``bars``.
    """
    if bars.empty:
        raise ValueError("no hay velas que dibujar")
    ohlc = adjusted_ohlc(bars.reset_index(drop=True))
    scale = 1.0 / float(ohlc["factor"].iloc[-1])
    if not np.isfinite(scale):
        raise ValueError("la última vela no tiene cierre ajustado")
    prices = ohlc.loc[:, ["open", "high", "low", "close"]] * scale
    dates = pd.to_datetime(bars["date"]).reset_index(drop=True)
    figure_match = match.scaled(scale) if match is not None else None

    last = len(prices) - 1
    first = max(0, last - sessions + 1)
    if figure_match is not None:
        first = max(0, min(first, figure_match.pivots[0].index - LEFT_MARGIN))

    figure = Figure(figsize=SIZE_INCHES, dpi=DPI, facecolor=SURFACE)
    FigureCanvasAgg(figure)
    ax = figure.add_axes((0.035, 0.09, 0.885, 0.76), facecolor=SURFACE)
    _candles(ax, prices.iloc[first:], first)
    if figure_match is not None:
        _pattern(ax, figure_match, last)
    _frame(ax, dates, first, last)
    figure.text(0.035, 0.94, title, color=INK, fontsize=13, fontweight="bold", va="center")
    if subtitle:
        figure.text(0.035, 0.885, subtitle, color=INK_SECONDARY, fontsize=9.5, va="center")

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", facecolor=SURFACE)
    return buffer.getvalue()


def _candles(ax: Axes, prices: pd.DataFrame, offset: int) -> None:
    x = np.arange(offset, offset + len(prices))
    opens, closes = prices["open"].to_numpy(), prices["close"].to_numpy()
    highs, lows = prices["high"].to_numpy(), prices["low"].to_numpy()
    rising = closes >= opens
    for mask, color in ((rising, CANDLE_UP), (~rising, CANDLE_DOWN)):
        ax.vlines(x[mask], lows[mask], highs[mask], color=color, linewidth=0.8, zorder=2)
        body = np.abs(closes[mask] - opens[mask])
        floor = np.minimum(opens[mask], closes[mask])
        # Un cuerpo de altura cero no se pinta: se le da un mínimo visible.
        body = np.maximum(body, (np.nanmax(highs) - np.nanmin(lows)) * 0.002)
        ax.bar(x[mask], body, bottom=floor, width=0.62, color=color, linewidth=0, zorder=3)


def _pattern(ax: Axes, match: PatternMatch, last: int) -> None:
    """La figura por sus pivotes, la línea rota y los niveles rotulados a la derecha."""
    xs = [p.index for p in match.pivots] + [match.bar]
    ys = [p.price for p in match.pivots] + [match.entry]
    ax.plot(xs, ys, color=FIGURE, linewidth=2, solid_joinstyle="round", zorder=5)
    ax.scatter(xs[:-1], ys[:-1], s=42, color=FIGURE, edgecolors=SURFACE, linewidths=1.5, zorder=6)
    (x1, y1), (x2, y2) = match.neckline
    ax.plot([x1, x2], [y1, y2], color=FIGURE, linewidth=1.2, linestyle=(0, (4, 3)), zorder=4)

    end = last + RIGHT_MARGIN
    levels = (
        ("Entrada", match.entry, INK_SECONDARY),
        ("Stop", match.stop, STOP),
        ("Objetivo", match.target, TARGET),
    )
    for name, level, color in levels:
        ax.hlines(level, match.bar, end, color=color, linewidth=1.4, zorder=4)
        ax.annotate(
            f"{name} {_price(level)}",
            xy=(end, level),
            xytext=(-2, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color=INK,
            fontsize=8.5,
            zorder=7,
        )


def _frame(ax: Axes, dates: pd.Series, first: int, last: int) -> None:
    """Ejes discretos: fechas en unas pocas sesiones, precios a la derecha, rejilla tenue."""
    ax.set_xlim(first - 1, last + RIGHT_MARGIN + 0.5)
    ticks = np.linspace(first, last, 6).round().astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{dates.iloc[t]:%d/%m/%y}" for t in ticks])
    ax.yaxis.tick_right()
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0, pad=6)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.margins(y=0.08)


def _price(value: float) -> str:
    """Dos decimales salvo en precios pequeños, donde harían falta más para distinguir niveles."""
    return f"{value:.2f}" if abs(value) >= 10 else f"{value:.4g}"
