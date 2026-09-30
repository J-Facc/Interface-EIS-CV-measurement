"""Évaluateur SÉCURISÉ d'un circuit équivalent écrit en texte par l'utilisateur.

Exemple : ``Re + parallel(R(Rct), Q(Qdl, alpha)) + ZD_bounded(w, R_D, tau_d)``

Modèle de sécurité
------------------
Le texte n'est JAMAIS passé à ``eval``/``exec``/``compile`` : aucun bytecode Python
n'est produit à partir de lui. La chaîne de traitement est :

1. contrôles textuels (``_precheck_text``) : type ``str``, longueur bornée,
   caractères ASCII imprimables uniquement (pas d'homoglyphe Unicode, pas d'octet
   nul), aucune séquence ``__`` ;
2. ``ast.parse(expr, mode="eval")`` : analyse syntaxique pure, n'exécute rien ;
3. liste blanche des TYPES de nœuds (``_check_node_types``) : tout nœud absent de
   ``_ALLOWED_NODE_TYPES`` est rejeté — ``Attribute``, ``Subscript``, ``Lambda``,
   compréhensions, chaînes, f-strings, ``:=``, comparaisons… ne passent pas ; le
   nombre de nœuds est borné ;
4. validation STRUCTURELLE et compilation (``_Compiler``) : un appel n'est permis
   que sur un ``ast.Name`` de ``ALLOWED_FUNCTIONS``, sans argument nommé, avec
   l'arité attendue ; un nom libre doit être ``w`` ou un nom de paramètre valide
   (ni builtin Python, ni mot-clé, ni fonction, ni ``_…``) ; une constante doit
   être un nombre fini (pas de ``bool``, ``str``, ``None``…) ; la profondeur est
   bornée ;
5. l'arbre validé est traduit en un arbre de FERMETURES Python qui n'appellent
   que des fonctions de ``circuit/elements.py`` (résolues à la compilation depuis
   un dictionnaire explicite) et des opérateurs arithmétiques de ``operator``.
   Les valeurs manipulées sont des flottants/complexes numpy : aucune arithmétique
   sur des entiers Python non bornés (``10**10**10`` rend ``inf``, pas un calcul
   interminable).

Les étapes 1 et 3 sont redondantes avec l'étape 4 par construction (défense en
profondeur) : chacune suffit à elle seule à rejeter les attaques connues, et les
tests vérifient chaque couche indépendamment.

API publique : ``parse_circuit(expr) -> (Z, param_names)``.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import keyword
import numbers
import operator
import re
import warnings
from typing import Callable

import numpy as np

from circuit import elements
from circuit.registry_elements import ELEMENTS as _REGISTRY

__all__ = [
    "ALLOWED_FUNCTIONS",
    "CircuitError",
    "CircuitSyntaxError",
    "UnsafeExpressionError",
    "CircuitEvaluationError",
    "CompiledCircuit",
    "parse_circuit",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CircuitError(ValueError):
    """Erreur de base pour tout circuit utilisateur invalide."""


class CircuitSyntaxError(CircuitError):
    """Expression mal formée ou incorrecte (syntaxe, arité, nom de paramètre)."""


class UnsafeExpressionError(CircuitError):
    """Construction interdite par la liste blanche (rejet de sécurité)."""


class CircuitEvaluationError(CircuitError):
    """Appel du circuit compilé avec des paramètres manquants, inconnus ou invalides."""


# ---------------------------------------------------------------------------
# Listes blanches — explicites, jamais dérivées du texte utilisateur
# ---------------------------------------------------------------------------

#: Seules fonctions appelables depuis une expression. Liste EXPLICITE : ajouter un
#: élément demande de l'ajouter ici, dans elements.py ET dans registry_elements.py
#: (les tests vérifient la cohérence des trois).
_FUNCTIONS: dict[str, Callable] = {
    "R": elements.R,
    "C": elements.C,
    "L": elements.L,
    "Q": elements.Q,
    "W": elements.W,
    "Wo": elements.Wo,
    "Ws": elements.Ws,
    "ZD_bounded": elements.ZD_bounded,
    "parallel": elements.parallel,
}
ALLOWED_FUNCTIONS: frozenset[str] = frozenset(_FUNCTIONS)
_COMBINATORS: frozenset[str] = frozenset({"parallel"})

#: Nom réservé à la pulsation (rad/s).
OMEGA_NAME = "w"

#: Types de nœuds AST autorisés (comparaison par type EXACT, pas isinstance).
_ALLOWED_NODE_TYPES: frozenset[type] = frozenset({
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.USub,
})

_BINOPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

#: Messages explicites pour les constructions dangereuses les plus courantes.
_FORBIDDEN_MESSAGES: dict[str, str] = {
    "Attribute": "l'accès à un attribut (« objet.nom ») est interdit",
    "Subscript": "l'indexation (« x[...] ») est interdite",
    "Lambda": "« lambda » est interdit",
    "ListComp": "les compréhensions de liste sont interdites",
    "SetComp": "les compréhensions d'ensemble sont interdites",
    "DictComp": "les compréhensions de dictionnaire sont interdites",
    "GeneratorExp": "les expressions génératrices sont interdites",
    "comprehension": "les compréhensions sont interdites",
    "NamedExpr": "l'opérateur « := » est interdit",
    "JoinedStr": "les f-strings sont interdites",
    "FormattedValue": "les f-strings sont interdites",
    "Starred": "le dépaquetage « *x » est interdit",
    "keyword": "les arguments nommés (« f(x=...) ») sont interdits",
    "IfExp": "l'expression conditionnelle « a if c else b » est interdite",
    "Compare": "les comparaisons sont interdites",
    "BoolOp": "les opérateurs booléens (and/or) sont interdits",
    "Tuple": "les tuples sont interdits",
    "List": "les listes sont interdites",
    "Dict": "les dictionnaires sont interdits",
    "Set": "les ensembles sont interdits",
    "Slice": "les tranches sont interdites",
    "Await": "« await » est interdit",
    "Yield": "« yield » est interdit",
    "YieldFrom": "« yield from » est interdit",
}

# Limites anti-déni de service (texte, arbre, paramètres).
MAX_EXPRESSION_LENGTH = 2000
MAX_NODES = 1000
MAX_DEPTH = 60
MAX_PARAMS = 64
MAX_PARAM_NAME_LENGTH = 64

_PARAM_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_ALLOWED_WHITESPACE = frozenset("\t\n\r")

#: Noms interdits comme paramètres : tous les builtins Python (exec, eval, open,
#: getattr, __import__, print…) et les mots-clés. Ces noms ne seraient de toute
#: façon jamais résolus (l'évaluateur n'a accès à aucun builtin), mais les
#: refuser rend l'intention illisible pour un attaquant et explicite l'erreur.
_RESERVED_NAMES: frozenset[str] = frozenset(
    set(dir(builtins)) | set(keyword.kwlist) | set(getattr(keyword, "softkwlist", []))
)

#: Noms qui deviendraient silencieusement un paramètre alors que l'utilisateur
#: pensait sans doute à autre chose.
_CONFUSING_NAMES: dict[str, str] = {
    "j": "« j » n'est pas l'unité imaginaire : écrivez 1j (ex. 1j*w*C).",
    "omega": "la pulsation s'écrit « w », pas « omega ».",
}


# ---------------------------------------------------------------------------
# Étape 1 — contrôles textuels
# ---------------------------------------------------------------------------

def _precheck_text(expr: object) -> str:
    """Contrôles purement textuels, AVANT toute analyse syntaxique."""
    if not isinstance(expr, str):
        raise CircuitSyntaxError(
            f"le circuit doit être une chaîne de caractères, reçu {type(expr).__name__}."
        )
    # Copie en str EXACT : une sous-classe de str ne peut pas surcharger les
    # méthodes utilisées ci-dessous (str.__str__ non lié ignore les surcharges).
    expr = str.__str__(expr)
    if not expr.strip():
        raise CircuitSyntaxError("le circuit est vide.")
    if len(expr) > MAX_EXPRESSION_LENGTH:
        raise UnsafeExpressionError(
            f"expression trop longue ({len(expr)} caractères, maximum "
            f"{MAX_EXPRESSION_LENGTH})."
        )
    for pos, ch in enumerate(expr):
        if not (" " <= ch <= "~" or ch in _ALLOWED_WHITESPACE):
            raise UnsafeExpressionError(
                f"caractère interdit {ch!r} (U+{ord(ch):04X}) en position {pos} : "
                "seuls les caractères ASCII imprimables sont acceptés."
            )
    if "__" in expr:
        raise UnsafeExpressionError(
            "la séquence « __ » (double tiret bas) est interdite."
        )
    return expr


# ---------------------------------------------------------------------------
# Étape 2 — analyse syntaxique
# ---------------------------------------------------------------------------

def _parse(expr: str) -> ast.Expression:
    try:
        # « 1if w else 2 » etc. émettent un SyntaxWarning sur stderr : sans intérêt
        # ici (la construction est de toute façon rejetée plus loin). « ignore »
        # plutôt que « error » : le filtre de warnings est global au processus
        # (Streamlit est multi-thread), et « error » pourrait lever ailleurs.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        if re.search(r"\bimport\b", expr):
            raise UnsafeExpressionError("« import » est interdit.") from None
        raise CircuitSyntaxError(
            f"syntaxe invalide : {exc.msg} (colonne {exc.offset})."
        ) from None
    except (RecursionError, MemoryError, ValueError):
        raise UnsafeExpressionError("expression trop complexe ou invalide.") from None
    if type(tree) is not ast.Expression:  # défense : mode="eval" le garantit
        raise UnsafeExpressionError("seule une expression unique est acceptée.")
    return tree


# ---------------------------------------------------------------------------
# Étape 3 — liste blanche des types de nœuds
# ---------------------------------------------------------------------------

def _check_node_types(tree: ast.AST) -> None:
    """Rejette tout nœud dont le type n'est pas dans la liste blanche.

    ``ast.walk`` est itératif (file) : pas de récursion, donc pas de
    ``RecursionError`` quelle que soit la profondeur.
    """
    count = 0
    for node in ast.walk(tree):
        count += 1
        if count > MAX_NODES:
            raise UnsafeExpressionError(
                f"expression trop complexe (plus de {MAX_NODES} nœuds)."
            )
        if type(node) not in _ALLOWED_NODE_TYPES:
            name = type(node).__name__
            detail = _FORBIDDEN_MESSAGES.get(name, f"construction « {name} » non autorisée")
            raise UnsafeExpressionError(f"{detail}.")


# ---------------------------------------------------------------------------
# Étape 4/5 — validation structurelle et compilation en fermetures
# ---------------------------------------------------------------------------

def _element_arity(name: str) -> int:
    n = _REGISTRY[name]["n_params"]
    if not isinstance(n, int):
        raise RuntimeError(f"registre incohérent : arité de {name!r} = {n!r}")
    return n


def _check_signatures() -> None:
    """Vérifie à l'import que liste blanche, registre et signatures concordent."""
    if set(_FUNCTIONS) != set(_REGISTRY):
        raise RuntimeError(
            "circuit/parser.py et circuit/registry_elements.py divergent : "
            f"{sorted(set(_FUNCTIONS) ^ set(_REGISTRY))}"
        )
    for name, func in _FUNCTIONS.items():
        if name in _COMBINATORS:
            continue
        params = list(inspect.signature(func).parameters.values())
        if not params or params[0].name != OMEGA_NAME:
            raise RuntimeError(f"elements.{name} doit prendre « w » en premier argument.")
        if len(params) - 1 != _element_arity(name):
            raise RuntimeError(
                f"elements.{name} : {len(params) - 1} paramètres, le registre en "
                f"annonce {_element_arity(name)}."
            )


_check_signatures()


def _validate_param_name(name: str) -> None:
    if name in _CONFUSING_NAMES:
        raise CircuitSyntaxError(_CONFUSING_NAMES[name])
    if name.startswith("_") or "__" in name:
        raise UnsafeExpressionError(
            f"nom « {name} » interdit : un paramètre ne peut pas commencer par « _ »."
        )
    if name in _RESERVED_NAMES:
        raise UnsafeExpressionError(
            f"« {name} » est un nom réservé de Python et ne peut pas servir de paramètre."
        )
    if not _PARAM_NAME_RE.match(name) or len(name) > MAX_PARAM_NAME_LENGTH:
        raise CircuitSyntaxError(
            f"nom de paramètre invalide « {name} » : lettres ASCII, chiffres et « _ », "
            f"commençant par une lettre, {MAX_PARAM_NAME_LENGTH} caractères au plus."
        )


# Une fermeture compilée prend l'environnement (omega, params) et rend une valeur.
_Env = tuple  # (np.ndarray, dict[str, np.float64])
_Node = Callable[[_Env], object]


class _Compiler:
    """Valide l'arbre (déjà filtré par type) et le traduit en fermetures."""

    def __init__(self) -> None:
        self.param_names: list[str] = []  # ordre de première apparition

    # -- utilitaires -------------------------------------------------------

    def _register_param(self, name: str) -> _Node:
        _validate_param_name(name)
        if name not in self.param_names:
            if len(self.param_names) >= MAX_PARAMS:
                raise UnsafeExpressionError(f"trop de paramètres (maximum {MAX_PARAMS}).")
            self.param_names.append(name)

        def param(env: _Env, _name: str = name):
            return env[1][_name]

        return param

    @staticmethod
    def _constant(node: ast.Constant, allow_complex: bool = True) -> _Node:
        value = node.value
        # bool est une sous-classe de int : à exclure explicitement.
        if type(value) is bool or not isinstance(value, (int, float, complex)):
            raise UnsafeExpressionError(
                f"constante non numérique interdite : {type(value).__name__}."
            )
        if isinstance(value, complex) and not allow_complex:
            raise CircuitSyntaxError(
                f"un paramètre d'élément doit être réel, reçu {value!r}."
            )
        # Conversion en flottant numpy : jamais d'arithmétique sur entier Python
        # (10**10**10 sur des int ne terminerait pas ; en float64 il vaut inf).
        try:
            const = (np.complex128(value) if isinstance(value, complex)
                     else np.float64(value))
        except OverflowError:
            raise CircuitSyntaxError("constante trop grande.") from None
        if not np.isfinite(const):
            raise CircuitSyntaxError(f"constante non finie : {value!r}.")

        def constant(env: _Env, _c=const):
            return _c

        return constant

    # -- visite ------------------------------------------------------------

    def compile(self, node: ast.AST, depth: int = 0) -> _Node:
        if depth > MAX_DEPTH:
            raise UnsafeExpressionError(
                f"expression trop imbriquée (profondeur maximale {MAX_DEPTH})."
            )
        kind = type(node)

        if kind is ast.Expression:
            return self.compile(node.body, depth + 1)

        if kind is ast.Constant:
            return self._constant(node)

        if kind is ast.Name:
            if node.id == OMEGA_NAME:
                return lambda env: env[0]
            if node.id in ALLOWED_FUNCTIONS:
                raise CircuitSyntaxError(
                    f"« {node.id} » est un élément : il doit être appelé, ex. {node.id}(...)."
                )
            return self._register_param(node.id)

        if kind is ast.UnaryOp:
            if type(node.op) is not ast.USub:
                raise UnsafeExpressionError(
                    f"opérateur unaire « {type(node.op).__name__} » non autorisé."
                )
            operand = self.compile(node.operand, depth + 1)
            return lambda env: operator.neg(operand(env))

        if kind is ast.BinOp:
            op = _BINOPS.get(type(node.op))
            if op is None:
                raise UnsafeExpressionError(
                    f"opérateur « {type(node.op).__name__} » non autorisé "
                    "(seuls + - * / ** sont permis)."
                )
            left = self.compile(node.left, depth + 1)
            right = self.compile(node.right, depth + 1)
            return lambda env: op(left(env), right(env))

        if kind is ast.Call:
            return self._call(node, depth)

        # Inatteignable si _check_node_types a été appliqué ; défense en profondeur.
        name = kind.__name__
        detail = _FORBIDDEN_MESSAGES.get(name, f"construction « {name} » non autorisée")
        raise UnsafeExpressionError(f"{detail}.")

    def _call(self, node: ast.Call, depth: int) -> _Node:
        func = node.func
        if type(func) is not ast.Name:
            raise UnsafeExpressionError(
                "seul l'appel direct d'un élément autorisé est permis "
                f"({', '.join(sorted(ALLOWED_FUNCTIONS))})."
            )
        name = func.id
        if name not in ALLOWED_FUNCTIONS:
            raise UnsafeExpressionError(
                f"fonction « {name} » non autorisée. Éléments disponibles : "
                f"{', '.join(sorted(ALLOWED_FUNCTIONS))}."
            )
        if node.keywords:
            raise UnsafeExpressionError(_FORBIDDEN_MESSAGES["keyword"] + ".")
        if any(type(a) is ast.Starred for a in node.args):
            raise UnsafeExpressionError(_FORBIDDEN_MESSAGES["Starred"] + ".")
        target = _FUNCTIONS[name]

        if name in _COMBINATORS:
            if len(node.args) < 2:
                raise CircuitSyntaxError(f"{name}() exige au moins deux impédances.")
            branches = tuple(self.compile(a, depth + 1) for a in node.args)
            return lambda env: target(*(b(env) for b in branches))

        arity = _element_arity(name)
        args = list(node.args)
        # « w » explicite en premier argument : toléré, puis retiré (injecté).
        if (len(args) == arity + 1 and type(args[0]) is ast.Name
                and args[0].id == OMEGA_NAME):
            args = args[1:]
        if len(args) != arity:
            roles = ", ".join(_REGISTRY[name]["params"])
            raise CircuitSyntaxError(
                f"{name}({roles}) attend {arity} paramètre(s), {len(args)} fourni(s)."
            )
        compiled = tuple(self._element_arg(name, a, depth + 1) for a in args)
        return lambda env: target(env[0], *(a(env) for a in compiled))

    def _element_arg(self, element: str, node: ast.AST, depth: int) -> _Node:
        """Argument d'un élément : nom de paramètre ou nombre littéral (±)."""
        kind = type(node)
        if kind is ast.Constant:
            return self._constant(node, allow_complex=False)
        if (kind is ast.UnaryOp and type(node.op) is ast.USub
                and type(node.operand) is ast.Constant):
            inner = self._constant(node.operand, allow_complex=False)
            value = operator.neg(inner(None))
            return lambda env, _v=value: _v
        if kind is ast.Name:
            if node.id == OMEGA_NAME:
                raise CircuitSyntaxError(
                    f"« w » (pulsation) ne peut être passé à {element}() qu'en "
                    "premier argument."
                )
            if node.id in ALLOWED_FUNCTIONS:
                raise CircuitSyntaxError(
                    f"« {node.id} » est un élément, pas un paramètre de {element}()."
                )
            return self._register_param(node.id)
        # Argument refusé : on valide d'abord le sous-arbre avec un compilateur
        # jetable, pour qu'une construction interdite (ex. R(open(...))) soit
        # signalée comme un rejet de sécurité et non comme une simple erreur.
        _Compiler().compile(node, depth + 1)
        raise CircuitSyntaxError(
            f"les arguments de {element}() doivent être des noms de paramètres ou des "
            "nombres littéraux (pas d'expression ni d'appel imbriqué)."
        )


# ---------------------------------------------------------------------------
# Circuit compilé
# ---------------------------------------------------------------------------

class CompiledCircuit:
    """Fonction d'impédance ``Z(w, **params)`` issue d'une expression validée.

    Les paramètres attendus sont exactement ``param_names`` (ni plus, ni moins) ;
    chaque valeur doit être un nombre réel. Le résultat est un tableau complexe de
    même forme que ``w``.
    """

    __slots__ = ("_root", "expression", "param_names")

    def __init__(self, root: _Node, expression: str, param_names: list[str]) -> None:
        self._root = root
        self.expression = expression
        self.param_names = tuple(param_names)

    def __call__(self, w, **params) -> np.ndarray:
        expected = set(self.param_names)
        missing = [p for p in self.param_names if p not in params]
        unknown = sorted(set(params) - expected)
        if missing or unknown:
            parts = []
            if missing:
                parts.append(f"manquant(s) : {', '.join(missing)}")
            if unknown:
                parts.append(f"inconnu(s) : {', '.join(unknown)}")
            raise CircuitEvaluationError("paramètres " + " ; ".join(parts) + ".")
        values: dict[str, np.float64] = {}
        for name in self.param_names:
            value = params[name]
            if type(value) is bool or not isinstance(value, numbers.Real):
                raise CircuitEvaluationError(
                    f"le paramètre « {name} » doit être un nombre réel, "
                    f"reçu {type(value).__name__}."
                )
            values[name] = np.float64(value)
        try:
            omega = np.asarray(w, dtype=float)
        except (TypeError, ValueError) as exc:
            raise CircuitEvaluationError(f"pulsation « w » invalide : {exc}") from None
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            result = self._root((omega, values))
        return np.broadcast_to(np.asarray(result, dtype=complex), omega.shape).copy()

    def __repr__(self) -> str:
        return f"CompiledCircuit({self.expression!r}, params={list(self.param_names)})"


def parse_circuit(expr: str) -> tuple[Callable[..., np.ndarray], list[str]]:
    """Valide et compile une expression de circuit.

    Args:
        expr: Expression textuelle, ex. ``"Re + parallel(R(Rct), Q(Qdl, alpha))"``.

    Returns:
        ``(Z, param_names)`` : ``Z(w, **params)`` rend l'impédance complexe (Ω) ;
        ``param_names`` liste les paramètres libres dans l'ordre de première
        apparition dans l'expression (un nom répété est UN SEUL paramètre).

    Raises:
        UnsafeExpressionError: construction interdite (rejet de sécurité).
        CircuitSyntaxError: expression mal formée (syntaxe, arité, nom invalide).
    """
    text = _precheck_text(expr)
    tree = _parse(text)
    _check_node_types(tree)
    compiler = _Compiler()
    root = compiler.compile(tree)
    return CompiledCircuit(root, text, compiler.param_names), list(compiler.param_names)
