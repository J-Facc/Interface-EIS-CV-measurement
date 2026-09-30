"""Circuits équivalents définis par l'utilisateur (Python restreint, liste blanche AST).

Voir docs/CIRCUIT_UTILISATEUR.md et circuit/parser.py (modèle de sécurité).
"""

from circuit.parser import (
    CircuitError,
    CircuitEvaluationError,
    CircuitSyntaxError,
    UnsafeExpressionError,
    parse_circuit,
)

__all__ = [
    "CircuitError",
    "CircuitEvaluationError",
    "CircuitSyntaxError",
    "UnsafeExpressionError",
    "parse_circuit",
]
