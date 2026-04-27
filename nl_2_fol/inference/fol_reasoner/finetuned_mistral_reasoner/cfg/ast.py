"""
Code written by https://github.com/fvossel.

@misc{vossel2025advancingnaturallanguageformalization,
    title={Advancing Natural Language Formalization to First Order Logic with Fine-tuned LLMs},
    author={Felix Vossel and Till Mossakowski and Bj"orn Gehrke},
    year={2025},
    eprint={2509.22338},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2509.22338},
}

AST node definitions for a first-order logic (FOL) language.

Each node class represents a syntactic construct — terms (Variable, Constant,
Number, Function) or formulas (Atom, Not, And, Or, Xor, Implies, Iff,
Quantifier) — and provides serialisation to dict, Z3, Prover9, and TPTP.
FOLTransformer bridges the Lark parse tree to these AST nodes.
"""

from dataclasses import dataclass
from typing import Dict, List, Union

import z3
from lark import Transformer

_SORT = z3.DeclareSort("S")


# =========================
# Z3 Environment
# =========================


class Z3Env:
    """Tracks declared Z3 symbols. Single sort for all terms."""

    def __init__(self):
        """Initialise empty symbol, function, and predicate tables."""
        self.symbols: Dict[str, z3.ExprRef] = {}
        self.funcs: Dict[str, z3.FuncDeclRef] = {}
        self.preds: Dict[str, z3.FuncDeclRef] = {}

    def get_symbol(self, name: str) -> z3.ExprRef:
        """Get or create a Z3 constant (used for both variables and constants)."""
        if name not in self.symbols:
            self.symbols[name] = z3.Const(name, _SORT)
        return self.symbols[name]

    def get_func(self, name: str, arity: int) -> z3.FuncDeclRef:
        """Get or create an uninterpreted Z3 function of the given arity mapping S^arity -> S."""
        if name not in self.funcs:
            self.funcs[name] = z3.Function(name, *([_SORT] * arity), _SORT)
        return self.funcs[name]

    def get_pred(self, name: str, arity: int) -> z3.FuncDeclRef:
        """Get or create an uninterpreted Z3 predicate of the given arity mapping S^arity -> Bool."""
        if name not in self.preds:
            self.preds[name] = z3.Function(name, *([_SORT] * arity), z3.BoolSort())
        return self.preds[name]


# =========================
# Base Node
# =========================


class Node:
    """Base class for all AST nodes."""

    def to_dict(self) -> dict:
        """Serialise this node to a JSON-compatible dictionary."""
        raise NotImplementedError

    def to_z3(self, env: Z3Env = None) -> z3.ExprRef:
        """Translate this node into a Z3 expression using the given environment."""
        raise NotImplementedError

    def to_prover9(self) -> str:
        """Render this node as a Prover9-syntax string."""
        raise NotImplementedError

    def to_tptp(self) -> str:
        """Render this node as a TPTP-syntax string."""
        raise NotImplementedError

    @staticmethod
    def from_dict(d: dict) -> "Node":
        """Deserialise a node from a dictionary produced by to_dict."""
        t = d["_type"]
        if t not in NODE_CLASSES:
            raise ValueError(f"Unknown type: {t}")
        return NODE_CLASSES[t].from_dict(d)


# =========================
# Term Nodes
# =========================


@dataclass
class Variable(Node):
    """A logical variable, represented by a single lowercase letter in the grammar."""

    name: str

    def to_dict(self):
        """Serialise to dict with type tag and variable name."""
        return {"_type": "Variable", "name": self.name}

    @staticmethod
    def from_dict(d):
        """Deserialise a Variable from a dict produced by to_dict."""
        return Variable(d["name"])

    def to_z3(self, env: Z3Env = None):
        """Translate to a Z3 constant in the uninterpreted sort S."""
        return (env or Z3Env()).get_symbol(self.name)

    def to_prover9(self) -> str:
        """Render the variable name as-is; Prover9 treats uppercase as variables."""
        return self.name

    def to_tptp(self) -> str:
        """Render variable in TPTP syntax. TPTP requires variables to be uppercase; single lowercase letters are capitalized."""
        return self.name.upper()


@dataclass
class Constant(Node):
    """A ground constant or numeric literal, produced by the NAME terminal in the grammar."""

    name: str

    def to_dict(self):
        """Serialise to dict with type tag and constant name."""
        return {"_type": "Constant", "name": self.name}

    @staticmethod
    def from_dict(d):
        """Deserialise a Constant from a dict produced by to_dict."""
        return Constant(d["name"])

    def to_z3(self, env: Z3Env = None):
        """Translate to a Z3 constant in the uninterpreted sort S."""
        return (env or Z3Env()).get_symbol(self.name)

    def to_prover9(self) -> str:
        """Render the constant name as-is."""
        return self.name

    def to_tptp(self) -> str:
        """Render constant in TPTP syntax. TPTP requires constants to start with a lowercase letter."""
        return self.name.lower()


@dataclass
class Number(Node):
    """A numeric literal node, only reachable via from_dict deserialization."""

    value: Union[int, float]

    def to_dict(self):
        """Serialise to dict with type tag and numeric value."""
        return {"_type": "Number", "value": self.value}

    @staticmethod
    def from_dict(d):
        """Deserialise a Number from a dict produced by to_dict."""
        return Number(d["value"])

    def to_z3(self, env: Z3Env = None):
        """Encode the number as a named constant in the uninterpreted sort S."""
        return (env or Z3Env()).get_symbol(str(self.value))

    def to_prover9(self) -> str:
        """Render the numeric value as a plain string."""
        return str(self.value)

    def to_tptp(self) -> str:
        """Render number in TPTP syntax as an integer or rational literal."""
        return str(self.value)


@dataclass
class Function(Node):
    """A function application node, covering both named functions and arithmetic operators."""

    name: str
    args: List[Node]

    INFIX_OPS = {"+", "-", "*", "/"}

    def to_dict(self):
        """Serialise to dict with type tag, function name, and recursively serialised arguments."""
        return {
            "_type": "Function",
            "name": self.name,
            "args": [a.to_dict() for a in self.args],
        }

    @staticmethod
    def from_dict(d):
        """Deserialise a Function from a dict produced by to_dict."""
        return Function(d["name"], [Node.from_dict(a) for a in d["args"]])

    def to_z3(self, env: Z3Env = None):
        """Translate to an uninterpreted Z3 function application in sort S."""
        env = env or Z3Env()
        z3_args = [a.to_z3(env) for a in self.args]
        func = env.get_func(self.name, len(self.args))
        return func(*z3_args)

    def to_prover9(self) -> str:
        """Render in Prover9 syntax, using infix notation for arithmetic operators."""
        if self.name in self.INFIX_OPS and len(self.args) == 2:
            left = self.args[0].to_prover9()
            right = self.args[1].to_prover9()
            return f"({left} {self.name} {right})"

        args_str = ", ".join(a.to_prover9() for a in self.args)
        return f"{self.name}({args_str})"

    TPTP_ARITH_OPS = {
        "+": "$sum",
        "-": "$difference",
        "*": "$product",
        "/": "$quotient",
    }

    def to_tptp(self) -> str:
        """Render function application in TPTP syntax.

        Arithmetic operators (+, -, *, /) are mapped to their TPTP dollar-word
        equivalents ($sum, $difference, $product, $quotient) and emitted in
        prefix notation. All other functions are emitted as lowercase
        identifiers with a parenthesised argument list.
        """
        args_str = ",".join(a.to_tptp() for a in self.args)
        tptp_name = self.TPTP_ARITH_OPS.get(self.name, self.name.lower())
        return f"{tptp_name}({args_str})"


# =========================
# Formula Nodes
# =========================


@dataclass
class Atom(Node):
    """An atomic formula: either a named predicate application or an infix comparison."""

    predicate: str
    args: List[Node]

    INFIX_PREDS_P9 = {
        "=": "=",
        "<": "<",
        ">": ">",
        "≤": "<=",
        "≥": ">=",
        "≠": "!=",
    }

    def to_dict(self):
        """Serialise to dict with type tag, predicate name, and recursively serialised arguments."""
        return {
            "_type": "Atom",
            "predicate": self.predicate,
            "args": [a.to_dict() for a in self.args],
        }

    @staticmethod
    def from_dict(d):
        """Deserialise an Atom from a dict produced by to_dict."""
        return Atom(d["predicate"], [Node.from_dict(a) for a in d["args"]])

    def to_z3(self, env: Z3Env = None):
        """Translate to a Z3 boolean expression.

        Equality and disequality map to native Z3 operators; all other
        predicates become uninterpreted Z3 functions returning Bool.
        """
        env = env or Z3Env()
        z3_args = [a.to_z3(env) for a in self.args]

        if self.predicate == "=" and len(self.args) == 2:
            return z3_args[0] == z3_args[1]
        if self.predicate == "≠" and len(self.args) == 2:
            return z3_args[0] != z3_args[1]

        pred = env.get_pred(self.predicate, len(self.args))
        return pred(*z3_args)

    def to_prover9(self) -> str:
        """Render in Prover9 syntax, using infix notation for comparison predicates."""
        if self.predicate in self.INFIX_PREDS_P9 and len(self.args) == 2:
            left = self.args[0].to_prover9()
            right = self.args[1].to_prover9()
            op = self.INFIX_PREDS_P9[self.predicate]
            return f"({left} {op} {right})"

        if not self.args:
            return f"{self.predicate}"

        args_str = ", ".join(a.to_prover9() for a in self.args)
        return f"{self.predicate}({args_str})"

    INFIX_PREDS_TPTP = {
        "=": "=",
        "≠": "!=",
        "<": "$less",
        ">": "$greater",
        "≤": "$lesseq",
        "≥": "$greatereq",
    }

    def to_tptp(self) -> str:
        """Render an atom in TPTP syntax.

        All infix predicates (=, !=, <, >, ≤, ≥) are kept as infix expressions,
        mirroring the Prover9 approach. Arithmetic comparison predicates use
        their TPTP dollar-word symbols. All other predicates are emitted as
        lowercase identifiers with a parenthesised argument list.
        """
        if self.predicate in self.INFIX_PREDS_TPTP and len(self.args) == 2:
            left = self.args[0].to_tptp()
            right = self.args[1].to_tptp()
            op = self.INFIX_PREDS_TPTP[self.predicate]
            return f"({left} {op} {right})"

        if not self.args:
            return f"{self.predicate.lower()}"

        args_str = ",".join(a.to_tptp() for a in self.args)
        return f"{self.predicate.lower()}({args_str})"


@dataclass
class Not(Node):
    """Logical negation of a formula."""

    formula: Node

    def to_dict(self):
        """Serialise to dict with type tag and recursively serialised subformula."""
        return {"_type": "Not", "formula": self.formula.to_dict()}

    @staticmethod
    def from_dict(d):
        """Deserialise a Not from a dict produced by to_dict."""
        return Not(Node.from_dict(d["formula"]))

    def to_z3(self, env: Z3Env = None):
        """Translate to a Z3 Not expression."""
        return z3.Not(self.formula.to_z3(env or Z3Env()))

    def to_prover9(self) -> str:
        """Render negation in Prover9 syntax using the dash operator."""
        return f"-({self.formula.to_prover9()})"

    def to_tptp(self) -> str:
        """Render negation in TPTP syntax using the tilde operator."""
        return f"~({self.formula.to_tptp()})"


@dataclass
class And(Node):
    """Conjunction of two formulas."""

    left: Node
    right: Node

    def to_dict(self):
        """Serialise to dict with type tag and recursively serialised operands."""
        return {
            "_type": "And",
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @staticmethod
    def from_dict(d):
        """Deserialise an And from a dict produced by to_dict."""
        return And(Node.from_dict(d["left"]), Node.from_dict(d["right"]))

    def to_z3(self, env: Z3Env = None):
        """Translate to a Z3 And expression."""
        env = env or Z3Env()
        return z3.And(self.left.to_z3(env), self.right.to_z3(env))

    def to_prover9(self) -> str:
        """Render conjunction in Prover9 syntax using the ampersand operator."""
        return f"({self.left.to_prover9()} & {self.right.to_prover9()})"

    def to_tptp(self) -> str:
        """Render conjunction in TPTP syntax using the ampersand operator."""
        return f"({self.left.to_tptp()} & {self.right.to_tptp()})"


@dataclass
class Or(Node):
    """Disjunction of two formulas."""

    left: Node
    right: Node

    def to_dict(self):
        """Serialise to dict with type tag and recursively serialised operands."""
        return {
            "_type": "Or",
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @staticmethod
    def from_dict(d):
        """Deserialise an Or from a dict produced by to_dict."""
        return Or(Node.from_dict(d["left"]), Node.from_dict(d["right"]))

    def to_z3(self, env: Z3Env = None):
        """Translate to a Z3 Or expression."""
        env = env or Z3Env()
        return z3.Or(self.left.to_z3(env), self.right.to_z3(env))

    def to_prover9(self) -> str:
        """Render disjunction in Prover9 syntax using the pipe operator."""
        return f"({self.left.to_prover9()} | {self.right.to_prover9()})"

    def to_tptp(self) -> str:
        """Render disjunction in TPTP syntax using the pipe operator."""
        return f"({self.left.to_tptp()} | {self.right.to_tptp()})"


@dataclass
class Xor(Node):
    """Exclusive disjunction of two formulas."""

    left: Node
    right: Node

    def to_dict(self):
        """Serialise to dict with type tag and recursively serialised operands."""
        return {
            "_type": "Xor",
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @staticmethod
    def from_dict(d):
        """Deserialise an Xor from a dict produced by to_dict."""
        return Xor(Node.from_dict(d["left"]), Node.from_dict(d["right"]))

    def to_z3(self, env: Z3Env = None):
        """Translate to a Z3 Xor expression."""
        env = env or Z3Env()
        return z3.Xor(self.left.to_z3(env), self.right.to_z3(env))

    def to_prover9(self) -> str:
        """Render exclusive or in Prover9 syntax by expanding to (l | r) & -(l & r)."""
        left = self.left.to_prover9()
        right = self.right.to_prover9()
        return f"(({left} | {right}) & -(({left}) & ({right})))"

    def to_tptp(self) -> str:
        """Render exclusive or in TPTP syntax using the XOR operator (~|)."""
        return f"({self.left.to_tptp()} ~| {self.right.to_tptp()})"


@dataclass
class Implies(Node):
    """Material implication from left to right."""

    left: Node
    right: Node

    def to_dict(self):
        """Serialise to dict with type tag and recursively serialised operands."""
        return {
            "_type": "Implies",
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @staticmethod
    def from_dict(d):
        """Deserialise an Implies from a dict produced by to_dict."""
        return Implies(Node.from_dict(d["left"]), Node.from_dict(d["right"]))

    def to_z3(self, env: Z3Env = None):
        """Translate to a Z3 Implies expression."""
        env = env or Z3Env()
        return z3.Implies(self.left.to_z3(env), self.right.to_z3(env))

    def to_prover9(self) -> str:
        """Render implication in Prover9 syntax using the -> operator."""
        return f"({self.left.to_prover9()} -> {self.right.to_prover9()})"

    def to_tptp(self) -> str:
        """Render implication in TPTP syntax using the => operator."""
        return f"({self.left.to_tptp()} => {self.right.to_tptp()})"


@dataclass
class Iff(Node):
    """Biconditional (if and only if) between two formulas."""

    left: Node
    right: Node

    def to_dict(self):
        """Serialise to dict with type tag and recursively serialised operands."""
        return {
            "_type": "Iff",
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @staticmethod
    def from_dict(d):
        """Deserialise an Iff from a dict produced by to_dict."""
        return Iff(Node.from_dict(d["left"]), Node.from_dict(d["right"]))

    def to_z3(self, env: Z3Env = None):
        """Translate to Z3 equality of the two boolean subexpressions."""
        env = env or Z3Env()
        return self.left.to_z3(env) == self.right.to_z3(env)

    def to_prover9(self) -> str:
        """Render biconditional in Prover9 syntax using the <-> operator."""
        return f"({self.left.to_prover9()} <-> {self.right.to_prover9()})"

    def to_tptp(self) -> str:
        """Render biconditional in TPTP syntax using the <=> operator."""
        return f"({self.left.to_tptp()} <=> {self.right.to_tptp()})"


@dataclass
class Quantifier(Node):
    """A universally or existentially quantified formula over a single variable."""

    type: str
    variable: Variable
    formula: Node

    def to_dict(self):
        """Serialise to dict with type tag, quantifier type, variable, and recursively serialised body."""
        return {
            "_type": "Quantifier",
            "type": self.type,
            "variable": self.variable.to_dict(),
            "formula": self.formula.to_dict(),
        }

    @staticmethod
    def from_dict(d):
        """Deserialise a Quantifier from a dict produced by to_dict."""
        return Quantifier(
            d["type"], Node.from_dict(d["variable"]), Node.from_dict(d["formula"])
        )

    def to_z3(self, env: Z3Env = None):
        """Translate to a Z3 ForAll or Exists expression over the bound variable."""
        env = env or Z3Env()
        z3_var = self.variable.to_z3(env)
        body = self.formula.to_z3(env)

        if self.type in ("forall", "∀"):
            return z3.ForAll([z3_var], body)
        elif self.type in ("exists", "∃"):
            return z3.Exists([z3_var], body)
        raise ValueError(f"Unknown quantifier: {self.type}")

    def to_prover9(self) -> str:
        """Render the quantified formula in Prover9 syntax using all/exists keywords."""
        var = self.variable.to_prover9()
        body = self.formula.to_prover9()

        if self.type in ("forall", "∀"):
            return f"(all {var} {body})"
        elif self.type in ("exists", "∃"):
            return f"(exists {var} {body})"
        raise ValueError(f"Unknown quantifier: {self.type}")

    def to_tptp(self) -> str:
        """Render a quantified formula in TPTP syntax.

        Universal quantification uses ! and existential uses ?,
        with the bound variable listed in brackets: ![X]: body or ?[X]: body.
        """
        var = self.variable.to_tptp()
        body = self.formula.to_tptp()

        if self.type in ("forall", "∀"):
            return f"(![{var}]: {body})"
        elif self.type in ("exists", "∃"):
            return f"(?[{var}]: {body})"
        raise ValueError(f"Unknown quantifier: {self.type}")


# =========================
# Registry
# =========================

NODE_CLASSES = {
    "Variable": Variable,
    "Constant": Constant,
    "Number": Number,
    "Function": Function,
    "Atom": Atom,
    "Not": Not,
    "And": And,
    "Or": Or,
    "Xor": Xor,
    "Implies": Implies,
    "Iff": Iff,
    "Quantifier": Quantifier,
}


# =========================
# Transformer
# =========================
class FOLTransformer(Transformer):
    """Transforms parsed tokens from Lark parser into AST nodes."""

    @staticmethod
    def _fold_binary(items, node_cls):
        """Left-fold a variable-length item list into nested binary nodes."""
        node = items[0]
        for item in items[1:]:
            node = node_cls(node, item)
        return node

    def VARIABLE(self, items):
        """Transform variable token into Variable node."""
        return Variable(str(items))

    def NAME(self, items):
        """Transform name token into Constant node."""
        return Constant(str(items))

    def function_(self, items):
        """Transform function application into Function node."""
        name = str(items[0])
        args = items[1:]
        if args and isinstance(args[0], list):
            args = args[0]
        return Function(name, args)

    def add_(self, items):
        """Transform addition into Function node."""
        left, right = items
        return Function("+", [left, right])

    def sub_(self, items):
        """Transform subtraction into Function node."""
        left, right = items
        return Function("-", [left, right])

    def mul_(self, items):
        """Transform multiplication into Function node."""
        left, right = items
        return Function("*", [left, right])

    def div_(self, items):
        """Transform division into Function node."""
        left, right = items
        return Function("/", [left, right])

    def atom_term(self, items):
        """Pass through atom term."""
        return items[0]

    def term(self, items):
        """Pass through term."""
        return items[0]

    def sum(self, items):
        """Pass through sum expression."""
        return items[0]

    def product(self, items):
        """Pass through product expression."""
        return items[0]

    def termlist(self, items):
        """Transform term list."""
        return items

    def infix_predicate(self, items):
        """Pass through infix predicate."""
        return items[0]

    def atom(self, items):
        """Pass through atom."""
        return items[0]

    def atom_(self, items):
        """Transform predicate application into Atom node."""
        pred = str(items[0])
        if not isinstance(items[1], list):
            args = [items[1]]
        else:
            args = items[1]
        return Atom(pred, args)

    def atom0_(self, items):
        """Transform bare predicate symbol into a zero-arity Atom node."""
        pred = str(items[0])
        return Atom(pred, [])

    def lt_(self, items):
        """Transform less-than comparison into Atom node."""
        left, right = items
        return Atom("<", [left, right])

    def gt_(self, items):
        """Transform greater-than comparison into Atom node."""
        left, right = items
        return Atom(">", [left, right])

    def eq_(self, items):
        """Transform equality comparison into Atom node."""
        left, right = items
        return Atom("=", [left, right])

    def le_(self, items):
        """Transform less-than-or-equal comparison into Atom node."""
        left, right = items
        return Atom("≤", [left, right])

    def ge_(self, items):
        """Transform greater-than-or-equal comparison into Atom node."""
        left, right = items
        return Atom("≥", [left, right])

    def ne_(self, items):
        """Transform not-equal comparison into Atom node."""
        left, right = items
        return Atom("≠", [left, right])

    def not_(self, items):
        """Transform negation into Not node."""
        return Not(items[0])

    def and_(self, items):
        """Transform conjunction into And node."""
        return self._fold_binary(items, And)

    def or_(self, items):
        """Transform disjunction into Or node."""
        return self._fold_binary(items, Or)

    def xor_(self, items):
        """Transform exclusive or into Xor node."""
        return self._fold_binary(items, Xor)

    def implies_(self, items):
        """Transform implication into Implies node."""
        return Implies(items[0], items[1])

    def iff_(self, items):
        """Transform biconditional into Iff node."""
        return Iff(items[0], items[1])

    def quantifier_(self, items):
        """Transform quantifier expression into Quantifier node."""
        quant = items[0]
        var = items[1]
        formula = items[2]
        return Quantifier(str(quant), var, formula)
