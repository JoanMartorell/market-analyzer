"""analyzer: el motor que ejecuta los pasos del pipeline y los pasos mismos.

``engine`` define el contrato de un paso y cómo se encadenan. Cada paso vive
en su propia subcarpeta de ``steps`` y no importa a ningún otro paso: se
comunican solo a través de ``StepContext.data``.
"""
