"""Tests de circuit/parser.py : sécurité (attaques) puis fonctionnement.

Les tests d'attaque vérifient trois choses :

1. ``parse_circuit`` LÈVE une exception de sécurité (``UnsafeExpressionError``)
   — une attaque qui passerait silencieusement ferait échouer le test ;
2. chaque couche de défense rejette l'attaque À ELLE SEULE (liste blanche des
   types de nœuds, puis validation structurelle), sans compter sur les autres ;
3. aucun effet de bord n'a lieu (fichier témoin jamais créé, ``eval``/``exec``
   jamais appelés).
"""

import ast
import builtins
import time

import numpy as np
import pytest

from circuit import elements
from circuit.parser import (
    ALLOWED_FUNCTIONS,
    MAX_DEPTH,
    MAX_EXPRESSION_LENGTH,
    CircuitError,
    CircuitEvaluationError,
    CircuitSyntaxError,
    UnsafeExpressionError,
    _check_node_types,
    _Compiler,
    parse_circuit,
)
from fits.physics import Z_randles_full

W = 2.0 * np.pi * np.logspace(-3, 6, 60)


# ===========================================================================
# 1. Attaques — toutes doivent être REJETÉES
# ===========================================================================

# Les attaques demandées explicitement, puis des variantes.
ATTACKS = [
    # --- demandées explicitement ---
    "__import__('os').system('echo pwned')",
    "exec('import os')",
    "(lambda: None)()",
    "[x for x in ().__class__.__bases__[0].__subclasses__()]",
    "w.__class__",
    "print('pwned')",
    "open('/etc/passwd')",
    # --- import / exec / eval / getattr ---
    "import os",
    "__import__",
    "eval('1+1')",
    "exec",
    "Re + eval",
    "getattr(w, 'real')",
    "getattr",
    "compile('1', '', 'eval')",
    "globals()",
    "locals()",
    "vars()",
    "dir()",
    "breakpoint()",
    "input()",
    "help()",
    "type(w)",
    "setattr(w, 'x', 1)",
    "delattr(w, 'x')",
    "memoryview(b'x')",
    # --- attributs / dunder ---
    "w.real",
    "R.__globals__",
    "R(1).real",
    "w.__class__.__mro__[1].__subclasses__()",
    "parallel.__code__",
    "().__class__",
    "''.__class__",
    # --- indexation ---
    "w[0]",
    "W(1)[0]",
    # --- lambda / compréhensions ---
    "lambda: 0",
    "(lambda x: x)(w)",
    "[x for x in w]",
    "{x for x in w}",
    "{x: 1 for x in w}",
    "sum(x for x in w)",
    "parallel(*[R(1) for _ in (1, 2)])",
    # --- appels non whitelistés ou indirects ---
    "R(1)(2)",
    "sum([1, 2])",
    "abs(w)",
    "np.exp(w)",
    "os.system('ls')",
    # --- arguments nommés / dépaquetage ---
    "R(r=1)",
    "R(*[1])",
    "parallel(**{})",
    # --- autres constructions ---
    "(x := 1)",
    "f'{w}'",
    "1 if w else 2",
    "w < 1",
    "w and 1",
    "not w",
    "(1, 2)",
    "[1, 2]",
    "{'a': 1}",
    "{1}",
    "w % 2",
    "w // 2",
    "w @ w",
    "~w",
    "+w",
    "w << 1",
    "w | 1",
    "...",
    "None",
    "True",
    "'abc'",
    "b'abc'",
    "R('100')",
    "R(True)",
    "R(None)",
    # --- noms réservés comme paramètres ---
    "Re + open",
    "Re + print",
    "R(exec)",
    "Q(getattr, alpha)",
    "R(_secret)",
    # --- contournements textuels ---
    "ｅｘｅｃ('1')",   # « exec » en pleine chasse (NFKC → exec)
    "R(1)\x00",
    "R(1) + ​Re",                  # espace de largeur nulle
    "eхec('1')",                   # homoglyphe cyrillique
    "__" + "import__('os')",       # « __ » reconstitué
]


@pytest.mark.parametrize("attack", ATTACKS)
def test_attack_is_rejected(attack):
    """Chaque attaque lève UnsafeExpressionError — jamais d'exécution silencieuse."""
    with pytest.raises(UnsafeExpressionError):
        parse_circuit(attack)


def test_attacks_never_return():
    """Contre-vérification explicite : aucun appel ne rend quoi que ce soit."""
    silently_accepted = []
    for attack in ATTACKS:
        try:
            parse_circuit(attack)
        except UnsafeExpressionError:
            continue
        silently_accepted.append(attack)
    assert silently_accepted == []


def test_attack_side_effects_never_happen(tmp_path):
    """Des attaques qui créeraient un fichier témoin ne le créent pas."""
    canary = tmp_path / "canary"
    payloads = [
        f"__import__('os').system('touch {canary}')",
        f"open('{canary}', 'w')",
        f"exec(\"open('{canary}', 'w')\")",
        f"eval(\"open('{canary}', 'w')\")",
        f"(lambda: open('{canary}', 'w'))()",
        f"[open('{canary}', 'w') for x in (1,)]",
        f"R(open('{canary}', 'w'))",
        f"parallel(R(1), open('{canary}', 'w'))",
    ]
    for payload in payloads:
        with pytest.raises(UnsafeExpressionError):
            parse_circuit(payload)
        assert not canary.exists(), f"effet de bord pour : {payload}"


def test_eval_and_exec_are_never_called(monkeypatch):
    """Ni le parsing ni l'évaluation n'appellent eval/exec (sur quoi que ce soit)."""
    calls = []

    def forbidden(name):
        def _f(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} appelé")
        return _f

    monkeypatch.setattr(builtins, "eval", forbidden("eval"))
    monkeypatch.setattr(builtins, "exec", forbidden("exec"))

    Z, names = parse_circuit("Re + parallel(R(Rct), Q(Qdl, alpha)) + ZD_bounded(w, D, delta)")
    Z(W, **{n: 1.0 for n in names})
    for attack in ATTACKS:
        with pytest.raises(UnsafeExpressionError):
            parse_circuit(attack)
    assert calls == []


# --- défense en profondeur : chaque couche seule ---------------------------

def _parses(expr):
    try:
        return ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError):
        return None


# Attaques qui franchissent ast.parse ET dont la structure est hors liste blanche
# de TYPES (la couche 3 seule doit les arrêter).
NODE_TYPE_ATTACKS = [
    "w.__class__",
    "w.real",
    "().__class__",
    "[x for x in ().__class__.__bases__[0].__subclasses__()]",
    "(lambda: None)()",
    "w[0]",
    "(x := 1)",
    "f'{w}'",
    "R(r=1)",
    "R(*[1])",
    "1 if w else 2",
    "w % 2",
    "+w",
    "not w",
    "(1, 2)",
]


@pytest.mark.parametrize("attack", NODE_TYPE_ATTACKS)
def test_node_type_whitelist_alone_rejects(attack):
    """Couche 3 (types de nœuds), sans les contrôles textuels de la couche 1."""
    tree = _parses(attack)
    assert tree is not None
    with pytest.raises(UnsafeExpressionError):
        _check_node_types(tree)


# Attaques que la couche 4 (validation structurelle) doit arrêter SEULE, même si
# les couches 1 et 3 étaient contournées.
STRUCTURAL_ATTACKS = NODE_TYPE_ATTACKS + [
    "__import__('os').system('echo pwned')",
    "exec('import os')",
    "eval('1')",
    "print('x')",
    "open('/etc/passwd')",
    "getattr(w, 'x')",
    "R(1)(2)",
    "Re + exec",
    "__import__",
    "R(_x)",
    "R('1')",
    "R(True)",
    "None",
]


@pytest.mark.parametrize("attack", STRUCTURAL_ATTACKS)
def test_structural_validation_alone_rejects(attack):
    """Couche 4 (validation + compilation), sans les couches 1 et 3."""
    tree = _parses(attack)
    assert tree is not None
    with pytest.raises(UnsafeExpressionError):
        _Compiler().compile(tree)


# --- déni de service ---------------------------------------------------------

def test_too_long_expression_rejected():
    expr = " + ".join(["R(1)"] * (MAX_EXPRESSION_LENGTH // 4))
    assert len(expr) > MAX_EXPRESSION_LENGTH
    with pytest.raises(UnsafeExpressionError):
        parse_circuit(expr)


def test_deep_nesting_rejected():
    expr = "w"
    for _ in range(MAX_DEPTH + 5):
        expr = f"parallel({expr}, w)"
    assert len(expr) <= MAX_EXPRESSION_LENGTH
    with pytest.raises(UnsafeExpressionError):
        parse_circuit(expr)


def test_many_unary_minus_rejected_without_recursion_error():
    with pytest.raises(CircuitError):
        parse_circuit("-" * 1500 + "w")


def test_many_parentheses_rejected():
    with pytest.raises(CircuitError):
        parse_circuit("(" * 900 + "w" + ")" * 900)


def test_huge_power_is_fast_and_does_not_hang():
    """10**10**10 sur des int Python ne terminerait pas ; ici tout est float64."""
    start = time.perf_counter()
    Z, names = parse_circuit("10**10**10 + 9**9**9**9")
    out = Z(W)
    assert time.perf_counter() - start < 1.0
    assert names == []
    assert np.all(np.isinf(out.real))


def test_huge_integer_literal_rejected():
    with pytest.raises(CircuitSyntaxError):
        parse_circuit("R(1" + "0" * 400 + ")")


def test_random_token_fuzz_never_leaks_other_exceptions():
    """Fuzz déterministe : soit CircuitError, soit un circuit sûr et évaluable.

    Toute chaîne ACCEPTÉE ne doit contenir aucun jeton dangereux hors d'un
    identifiant (ex. « execRe » est un simple nom de paramètre, pas un appel).
    """
    import random
    import re
    import warnings

    tokens = ["R", "C", "Q", "Wo", "ZD_bounded", "parallel", "w", "Re", "(", ")",
              ",", "+", "-", "*", "/", "**", "1", "2.5", "1j", ".", "[", "]", "_",
              "__", "lambda", ":", "'a'", "if", "else", "for", "in", "=", "%", "@",
              "~", "not", "exec", "open", "import", ";", "\n", "#"]
    rng = random.Random(1234)
    accepted = 0
    for _ in range(20000):
        expr = "".join(rng.choice(tokens) + rng.choice(["", " "])
                       for _ in range(rng.randint(1, 12)))
        try:
            Z, names = parse_circuit(expr)
        except CircuitError:
            continue
        accepted += 1
        code = expr.split("#", 1)[0]
        without_floats = re.sub(r"\d*\.\d+|\d+\.", "", code)  # « 1. », « 2.5 »
        assert not re.search(r"[.\[\]:'=%@~;]", without_floats), expr
        assert not re.search(r"\b(lambda|exec|open|import|if|for|not)\b", code), expr
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = Z(W[:5], **{n: rng.choice([0.0, 1.0, -1.0]) for n in names})
        assert out.shape == (5,)
    assert accepted > 100  # le fuzz exerce bien aussi le chemin d'acceptation


# ===========================================================================
# 2. Circuits valides
# ===========================================================================

def test_constant_resistor_has_no_parameter():
    Z, names = parse_circuit("R(1000)")
    assert names == []
    np.testing.assert_allclose(Z(W), 1000.0 + 0j)
    assert Z(W).shape == W.shape and Z(W).dtype == complex


def test_named_parameters_detected_in_order():
    Z, names = parse_circuit("Re + R(Rct)")
    assert names == ["Re", "Rct"]
    np.testing.assert_allclose(Z(W, Re=10.0, Rct=90.0), 100.0 + 0j)


def test_user_example_from_specification():
    Z, names = parse_circuit("Re + parallel(R(Rct), Q(Qdl, alpha)) + ZD_bounded(w, D, delta)")
    assert names == ["Re", "Rct", "Qdl", "alpha", "D", "delta"]
    p = dict(Re=100.0, Rct=1e3, Qdl=1e-6, alpha=0.9, D=200.0, delta=1.0)
    expected = (p["Re"]
                + elements.parallel(elements.R(W, p["Rct"]), elements.Q(W, p["Qdl"], p["alpha"]))
                + elements.ZD_bounded(W, p["D"], p["delta"]))
    np.testing.assert_allclose(Z(W, **p), expected, rtol=1e-14)


def test_full_randles_reconstruction_matches_fits_physics():
    """Le Randles complet de fits/physics.py se réécrit avec les éléments."""
    expr = ("Re + parallel(Re_prime + parallel(R(Rct) + ZD_bounded(R_D, tau_d), "
            "Q(Qdl, alpha)), C(Cb))")
    Z, names = parse_circuit(expr)
    assert sorted(names) == sorted(
        ["Re", "Re_prime", "Cb", "Rct", "Qdl", "alpha", "R_D", "tau_d"])
    p = dict(Re=100.0, Re_prime=5.0, Cb=1e-9, Rct=2e3, Qdl=1e-6, alpha=0.85,
             R_D=400.0, tau_d=2.0)
    np.testing.assert_allclose(Z(W, **p), Z_randles_full(W, **p), rtol=1e-12)


def test_simple_randles_with_semi_infinite_warburg():
    Z, names = parse_circuit("Rs + parallel(R(Rct) + W(sigma), C(Cdl))")
    assert names == ["Rs", "Rct", "sigma", "Cdl"]
    p = dict(Rs=50.0, Rct=500.0, sigma=30.0, Cdl=2e-5)
    zf = p["Rct"] + p["sigma"] * (1 - 1j) / np.sqrt(W)
    expected = p["Rs"] + 1.0 / (1.0 / zf + 1j * W * p["Cdl"])
    np.testing.assert_allclose(Z(W, **p), expected, rtol=1e-12)


def test_rc_parallel_matches_analytic():
    Z, _ = parse_circuit("parallel(R(Rp), C(Cp))")
    expected = 1e3 / (1.0 + 1j * W * 1e3 * 1e-6)
    np.testing.assert_allclose(Z(W, Rp=1e3, Cp=1e-6), expected, rtol=1e-12)


def test_three_branch_parallel_and_all_elements():
    Z, names = parse_circuit(
        "L(Lw) + parallel(R(R1), C(C1), Q(Q1, a1)) + Wo(Rw, tw) + Ws(Rs, ts)")
    assert names == ["Lw", "R1", "C1", "Q1", "a1", "Rw", "tw", "Rs", "ts"]
    out = Z(W, Lw=1e-7, R1=1e3, C1=1e-9, Q1=1e-6, a1=0.8, Rw=100.0, tw=0.5,
            Rs=50.0, ts=0.1)
    assert out.shape == W.shape
    assert np.all(np.isfinite(out))


def test_repeated_name_is_a_single_parameter():
    Z, names = parse_circuit("R(Ra) + R(Ra) + Ra")
    assert names == ["Ra"]
    np.testing.assert_allclose(Z(W, Ra=5.0), 15.0 + 0j)


def test_explicit_w_first_argument_is_equivalent():
    Z1, n1 = parse_circuit("ZD_bounded(w, R_D, tau_d) + Q(w, q, a)")
    Z2, n2 = parse_circuit("ZD_bounded(R_D, tau_d) + Q(q, a)")
    assert n1 == n2 == ["R_D", "tau_d", "q", "a"]
    p = dict(R_D=10.0, tau_d=1.0, q=1e-5, a=0.7)
    np.testing.assert_array_equal(Z1(W, **p), Z2(W, **p))


def test_w_usable_in_arithmetic():
    Z, names = parse_circuit("Re + 1/(1j*w*Cdl)")
    assert names == ["Re", "Cdl"]
    np.testing.assert_allclose(Z(W, Re=10.0, Cdl=1e-6),
                               10.0 + elements.C(W, 1e-6), rtol=1e-14)


def test_arithmetic_operators_and_unary_minus():
    Z, names = parse_circuit("(R(a) * 2 - b / 4) ** 1 + -c")
    assert names == ["a", "b", "c"]
    np.testing.assert_allclose(Z(W, a=3.0, b=8.0, c=1.0), 3.0 + 0j)


def test_negative_literal_element_argument():
    Z, names = parse_circuit("R(10) + L(-1e-6)")
    assert names == []
    np.testing.assert_allclose(Z(W), 10.0 - 1j * W * 1e-6)


def test_bare_number_and_bare_parameter():
    Z, names = parse_circuit("42")
    assert names == [] and np.all(Z(W) == 42.0)
    Z, names = parse_circuit("Rx")
    assert names == ["Rx"]
    np.testing.assert_allclose(Z(W, Rx=3.0), 3.0 + 0j)


def test_multiline_and_comment_are_accepted():
    Z, names = parse_circuit("Re + parallel(\n    R(Rct),\n    C(Cdl)\n)  # Randles simple")
    assert names == ["Re", "Rct", "Cdl"]


def test_returned_names_are_a_copy():
    Z, names = parse_circuit("Re + R(Rct)")
    names.append("hack")
    assert list(Z.param_names) == ["Re", "Rct"]


def test_division_by_zero_gives_inf_not_exception():
    Z, _ = parse_circuit("1 / Rx")
    assert np.all(np.isinf(Z(W, Rx=0.0).real))


def test_scalar_omega():
    Z, _ = parse_circuit("parallel(R(Rp), C(Cp))")
    out = Z(1.0, Rp=1.0, Cp=1.0)
    assert out.shape == ()
    assert out == pytest.approx(1.0 / (1.0 + 1j))


# ===========================================================================
# 3. Erreurs utilisateur (non sécuritaires) — messages clairs
# ===========================================================================

@pytest.mark.parametrize("expr", [
    "R()", "R(1, 2)", "Q(1)", "Q(1, 2, 3)", "ZD_bounded(w, 1)",
    "parallel(R(1))", "parallel()",
])
def test_wrong_arity(expr):
    with pytest.raises(CircuitSyntaxError):
        parse_circuit(expr)


@pytest.mark.parametrize("expr", [
    "R(2*Rct)",           # expression dans un argument d'élément
    "R(R(1))",            # appel imbriqué dans un argument d'élément
    "Q(Qdl, w)",          # w ailleurs qu'en premier argument
    "R(w, w)",
    "R + 1",              # élément utilisé comme valeur
    "R(C)",               # élément passé comme paramètre
    "R(1j)",              # paramètre d'élément complexe
    "Re + 1/(j*w*C1)",    # j n'est pas l'unité imaginaire
    "R(omega)",           # la pulsation s'écrit w
    "R(1e999)",           # constante non finie
    "R(1",                # syntaxe
    "Re +",
    "",
    "   ",
])
def test_user_errors_raise_syntax_error(expr):
    with pytest.raises(CircuitSyntaxError):
        parse_circuit(expr)


@pytest.mark.parametrize("value", [None, 123, b"R(1)", ["R(1)"]])
def test_non_string_input_rejected(value):
    with pytest.raises(CircuitSyntaxError):
        parse_circuit(value)


def test_str_subclass_cannot_override_checks():
    class Evil(str):
        def __contains__(self, item):
            return False

        def __len__(self):
            return 1

    with pytest.raises(UnsafeExpressionError):
        parse_circuit(Evil("__import__('os')"))


def test_evaluation_parameter_checks():
    Z, _ = parse_circuit("Re + R(Rct)")
    with pytest.raises(CircuitEvaluationError, match="manquant"):
        Z(W, Re=1.0)
    with pytest.raises(CircuitEvaluationError, match="inconnu"):
        Z(W, Re=1.0, Rct=1.0, Rtypo=2.0)
    for bad in ["100", None, True, 1 + 2j, [1.0]]:
        with pytest.raises(CircuitEvaluationError):
            Z(W, Re=bad, Rct=1.0)
    with pytest.raises(CircuitEvaluationError):
        Z("abc", Re=1.0, Rct=1.0)


def test_error_hierarchy():
    assert issubclass(UnsafeExpressionError, CircuitError)
    assert issubclass(CircuitSyntaxError, CircuitError)
    assert issubclass(CircuitEvaluationError, CircuitError)
    assert issubclass(CircuitError, ValueError)


def test_allowed_functions_is_exactly_the_documented_set():
    assert ALLOWED_FUNCTIONS == frozenset(
        {"R", "C", "L", "Q", "W", "Wo", "Ws", "ZD_bounded", "parallel"})
