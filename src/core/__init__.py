"""core: configuración, arranque y ejecución.

Es el único paquete que conoce a los otros dos. Carga la configuración,
monta el pipeline de ``analyzer`` para cada región activa y entrega el
resultado a través de ``delivery``.
"""

__version__ = "0.1.0"
