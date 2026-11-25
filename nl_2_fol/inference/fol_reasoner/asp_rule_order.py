"""Reorder the body literals of ASP rules so that clingo grounds them efficiently.

gringo grounds rule bodies roughly from left to right. Rules written as
``p(M) :- has_atom(M, A1), has_atom(M, A2), ..., has_bond_to(A1, A2), ...``
therefore enumerate all combinations of atoms of a molecule before the bonds
restrict them, which makes large molecules infeasible. Reordering the literals
(``has_atom(M, A1), has_bond_to(A1, A2), has_atom(M, A2), ...``) does not change
the meaning of a rule, only the grounding effort.
"""

import clingo.ast as ast

# The first argument of these predicates is a molecule, i.e. binding a new variable
# through it enumerates all atoms of the molecule
_MOLECULE_PREDICATES = {"has_atom"}


def _variables(node) -> set[str]:
    found: set[str] = set()

    def walk(n):
        if isinstance(n, ast.AST):
            if n.ast_type == ast.ASTType.Variable:
                if n.name != "_":
                    found.add(n.name)
                return
            for key in n.child_keys:
                walk(getattr(n, key))
        elif isinstance(n, (list, tuple, ast.ASTSequence)):
            for child in n:
                walk(child)

    walk(node)
    return found


def _is_positive_atom(literal) -> bool:
    return (
        literal.ast_type == ast.ASTType.Literal
        and literal.sign == ast.Sign.NoSign
        and literal.atom.ast_type == ast.ASTType.SymbolicAtom
    )


def _is_filter(literal) -> bool:
    """Negated atoms and comparisons only test variables once they are bound."""
    if literal.ast_type != ast.ASTType.Literal:
        return False
    if literal.atom.ast_type == ast.ASTType.Comparison:
        return True
    return literal.atom.ast_type == ast.ASTType.SymbolicAtom and (
        literal.sign != ast.Sign.NoSign
    )


def _molecule_variables(body) -> set[str]:
    result = set()
    for literal in body:
        if _is_positive_atom(literal):
            symbol = literal.atom.symbol
            if (
                symbol.ast_type == ast.ASTType.Function
                and symbol.name in _MOLECULE_PREDICATES
                and symbol.arguments
                and symbol.arguments[0].ast_type == ast.ASTType.Variable
            ):
                result.add(symbol.arguments[0].name)
    return result


def _reorder_body(body) -> list:
    molecule_vars = _molecule_variables(body)
    positives = [
        (i, lit, _variables(lit))
        for i, lit in enumerate(body)
        if _is_positive_atom(lit)
    ]
    filters = [
        (i, lit, _variables(lit)) for i, lit in enumerate(body) if _is_filter(lit)
    ]
    # aggregates, conditional literals etc. are placed at the end
    rest = [lit for lit in body if not _is_positive_atom(lit) and not _is_filter(lit)]

    ordered: list = []
    bound: set[str] = set()

    def place_ready_filters():
        for item in list(filters):
            if item[2] <= bound:
                ordered.append(item[1])
                filters.remove(item)

    place_ready_filters()
    while positives:

        def rank(item):
            index, _, variables = item
            new = len(variables - bound)
            shared = variables & bound
            if shared and new == 0:
                group = 0  # only tests bound variables
            elif shared - molecule_vars:
                group = 1  # connected to a bound atom, e.g. via a bond
            elif shared:
                group = 2  # connected only via the molecule: enumerates its atoms
            else:
                group = 3  # not connected (start of the rule)
            return (group, new if group < 3 else 0, index)

        best = min(positives, key=rank)
        positives.remove(best)
        ordered.append(best[1])
        bound |= best[2]
        place_ready_filters()

    ordered.extend(item[1] for item in sorted(filters, key=lambda x: x[0]))
    ordered.extend(rest)
    return ordered


def reorder_body_literals(program: str) -> str:
    """Return ``program`` with the body literals of each rule reordered for grounding.

    If the program cannot be parsed, it is returned unchanged (clingo reports the error).
    So are programs with ``#program`` parts other than ``base``, which would otherwise
    end up in ``base``.
    """
    statements: list = []
    try:
        ast.parse_string(program, statements.append, logger=lambda *_: None)
    except RuntimeError:
        return program
    if any(s.ast_type == ast.ASTType.Program and s.name != "base" for s in statements):
        return program
    lines = []
    for statement in statements:
        if statement.ast_type == ast.ASTType.Program:
            continue
        if statement.ast_type == ast.ASTType.Rule and len(statement.body) > 1:
            statement = statement.update(body=_reorder_body(statement.body))
        lines.append(str(statement))
    return "\n".join(lines)
