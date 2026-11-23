"""ip-sketch-bench: how well do compression schemes preserve inner products?"""

from .methods import (
    METHODS,
    AdaptiveStochasticQuant,
    CountSketchIP,
    GaussianJL,
    ProductQuant,
    QJL,
    RotatedQuant,
    UniformQuant,
)

__all__ = [
    "METHODS",
    "AdaptiveStochasticQuant",
    "CountSketchIP",
    "GaussianJL",
    "ProductQuant",
    "QJL",
    "RotatedQuant",
    "UniformQuant",
]
