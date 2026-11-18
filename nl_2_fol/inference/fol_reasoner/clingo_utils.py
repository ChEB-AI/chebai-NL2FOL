"""Clingo-based evaluation of ASP class definitions on molecules.

Adapted from the ``chebILP`` package (``chebILP.utils.split_prolog_literals``,
``chebILP.ilp_problem_builder.build_background_chemlog`` restricted to the
atom-level predicate set, and ``chebILP.evaluation.clingo_eval``) so that this
package does not depend on chebILP.

Differences to the chebILP version:
- Timeouts cover grounding *and* solving. For the stratified programs generated
  here, clingo does nearly all work during grounding, which ``Control.solve``
  timeouts do not cover. Grounding therefore runs in a forked subprocess that
  is killed when the time limit is exceeded.
- A timeout raises :class:`ClingoTimeoutError` instead of silently returning
  partial (usually empty) results.
"""

import multiprocessing
import multiprocessing.connection
import sys
import time
import zlib

import pandas as pd

# Predicates returned by chebi_utils that are not used as background knowledge
_SKIPPED_PREDICATES = {"EQ", "atom", "*", "r", "r#"}

# Target predicates of up to this arity are returned by `ground_extensions`
_MAX_TARGET_ARITY = 4


class ClingoTimeoutError(TimeoutError):
    """Raised if grounding / solving a program exceeds the time limit."""


def split_prolog_literals(body: str) -> list[str]:
    """Split a rule body into literals, respecting parenthesis and brace depth."""
    literals, current, depth, curly_depth = [], [], 0, 0
    for char in body:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "{":
            curly_depth += 1
        elif char == "}":
            curly_depth -= 1
        elif char == "," and depth == 0 and curly_depth == 0:
            literals.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        literals.append("".join(current).strip())
    return literals


def get_atom_id(atom_idx: int, molecule_id) -> str:
    return f"a{molecule_id}_{atom_idx + 1}"


def _molecule_facts(mol, mol_id, allowed_predicates: set[str] | None) -> list[str]:
    from chebi_utils.extract_properties import mol_to_fol_atoms

    def _allowed(predicate: str) -> bool:
        if predicate in _SKIPPED_PREDICATES:
            return False
        return allowed_predicates is None or predicate in allowed_predicates

    lines = [
        f"has_atom({mol_id},{get_atom_id(atom.GetIdx(), mol_id)})."
        for atom in mol.GetAtoms()
    ]
    atom_extensions, mol_extensions = mol_to_fol_atoms(
        mol, with_rings=False, with_steroids=False
    )

    for predicate, indices in atom_extensions.items():
        if predicate.startswith("cip_code_"):
            predicate = "cip_code_" + predicate[-1].upper()
        if not indices or not _allowed(predicate):
            continue
        for args in indices:
            if not isinstance(args, tuple):
                args = (args,)
            arg_str = ",".join(get_atom_id(a, mol_id) for a in args)
            lines.append(f"{predicate}({arg_str}).")

    lines.extend(
        f"{predicate}({mol_id})." for predicate in mol_extensions if _allowed(predicate)
    )
    return lines


# Facts of each molecule are the same in every evaluation (only the molecule ID differs),
# so they can be computed once: SMILES -> zlib-compressed facts with a placeholder for
# the molecule ID, valid for the allowed predicates in `_FACTS_CACHE_PREDICATES`.
_MOL_ID_PLACEHOLDER = "@M@"
_FACTS_CACHE: dict[str, bytes] = {}
_FACTS_CACHE_PREDICATES: frozenset[str] | None = None


def _facts_template(item) -> tuple[str, bytes | str]:
    """SMILES and compressed facts, or SMILES and error message if the facts failed."""
    smiles, mol, allowed_predicates = item
    try:
        facts = _molecule_facts(mol, _MOL_ID_PLACEHOLDER, allowed_predicates)
    except Exception as e:  # noqa: BLE001 - any failure drops the molecule
        return smiles, f"{type(e).__name__}: {e}"
    return smiles, zlib.compress("\n".join(facts).encode())


def cache_molecule_facts(
    molecules: dict[str, object], allowed_predicates: set[str], n_jobs: int = 1
) -> set[str]:
    """Compute the facts of ``molecules`` (SMILES -> RDKit molecule) in parallel and keep
    them in this process, so that later evaluations (also in forked processes) reuse them.

    Returns:
        The SMILES of molecules whose facts could not be computed (not cached).
    """
    global _FACTS_CACHE, _FACTS_CACHE_PREDICATES
    if _FACTS_CACHE_PREDICATES != frozenset(allowed_predicates):
        _FACTS_CACHE, _FACTS_CACHE_PREDICATES = {}, frozenset(allowed_predicates)
    items = [
        (smiles, mol, allowed_predicates)
        for smiles, mol in molecules.items()
        if smiles not in _FACTS_CACHE
    ]
    if not items:
        return set()
    start = time.monotonic()
    if n_jobs > 1 and "fork" in multiprocessing.get_all_start_methods():
        with multiprocessing.get_context("fork").Pool(n_jobs) as pool:
            results = list(pool.imap_unordered(_facts_template, items, chunksize=200))
    else:
        results = list(map(_facts_template, items))
    failed: set[str] = set()
    for smiles, facts in results:
        if isinstance(facts, str):
            print(f"Failed to compute ASP facts for {smiles}, dropping it: {facts}")
            failed.add(smiles)
        else:
            _FACTS_CACHE[smiles] = facts
    size = sum(len(v) for v in _FACTS_CACHE.values()) / 2**20
    print(
        f"Computed ASP facts of {len(items) - len(failed)} molecules in "
        f"{time.monotonic() - start:.0f}s ({size:.0f} MB compressed), "
        f"{len(failed)} failed"
    )
    return failed


def molecules_to_asp_facts(
    molecules_df: pd.DataFrame, allowed_predicates: set[str] | None = None
) -> list[str]:
    """Convert molecules into ASP facts (atom-level representation).

    Rings and steroid positions are not part of the representation.

    Args:
        molecules_df: DataFrame with a ``mol`` column of RDKit molecules. The index
            is used as molecule identifier and has to be a valid clingo term
            (e.g. an integer). With a ``smiles`` column, facts that have been computed
            by :func:`cache_molecule_facts` are reused.
        allowed_predicates: If given, only facts for these predicates are returned
            (``has_atom`` is always included). This keeps the representation
            consistent with the predicates that are described in the prompt.

    Returns:
        A list of facts, e.g. ``has_atom(1,a1_1).``, ``c(a1_1).``,
        ``bSINGLE(a1_1,a1_2).``, ``net_charge_neutral(1).``
    """
    use_cache = (
        "smiles" in molecules_df.columns
        and allowed_predicates is not None
        and _FACTS_CACHE_PREDICATES == frozenset(allowed_predicates)
    )
    lines = []
    for row in molecules_df.itertuples():
        cached = _FACTS_CACHE.get(row.smiles) if use_cache else None
        if cached is None:
            lines.extend(_molecule_facts(row.mol, row.Index, allowed_predicates))
        else:
            facts = zlib.decompress(cached).decode()
            lines.extend(facts.replace(_MOL_ID_PLACEHOLDER, str(row.Index)).split("\n"))
    return lines


def _ground_extensions(
    rules: list[str], background_facts: list[str], target_labels: list[str]
) -> dict[str, set[tuple[str, ...]]]:
    """Ground and solve, return the derived argument tuples of each target predicate.

    Raises:
        RuntimeError: if clingo reports errors (e.g. syntax errors or unsafe variables);
            the message contains clingo's error messages.
    """
    import clingo

    from nl_2_fol.inference.fol_reasoner.asp_rule_order import reorder_body_literals

    messages: list[str] = []

    def _log(code, message: str) -> None:
        # collect clingo's messages (instead of printing them to stderr), so that they
        # can be passed on - e.g. as feedback to the LLM
        messages.append(message.replace("[#inc_base];", "").strip())

    ctl = clingo.Control(["--warn=none"], logger=_log, message_limit=10)
    try:
        ctl.add("base", [], "\n".join(background_facts))
        # same meaning, but a literal order that gringo can ground efficiently
        ctl.add("base", [], reorder_body_literals("\n".join(rules)))
        # Only show the target predicates: iterating over all atoms of the model (including
        # all background facts) in Python is far more expensive than grounding itself.
        ctl.add(
            "base",
            [],
            "\n".join(
                f"#show {label}/{arity}."
                for label in target_labels
                for arity in range(_MAX_TARGET_ARITY + 1)
            ),
        )
        ctl.ground([("base", [])])
    except RuntimeError as e:
        details = "\n".join(messages)
        raise RuntimeError(f"{e}\n{details}" if details else str(e)) from None

    wanted = set(target_labels)
    extensions: dict[str, set[tuple[str, ...]]] = {}

    def _collect(model):
        for atom in model.symbols(shown=True):
            if atom.name in wanted:
                extensions.setdefault(atom.name, set()).add(
                    tuple(str(a) for a in atom.arguments)
                )

    ctl.solve(on_model=_collect)
    return extensions


def _ground_in_child(conn, rules, background_facts, target_labels):
    try:
        conn.send(("ok", _ground_extensions(rules, background_facts, target_labels)))
    except BaseException as e:  # noqa: BLE001 - forward every error to the parent
        conn.send(("error", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def ground_extensions(
    rules: list[str],
    background_facts: list[str],
    target_labels: list[str],
    timeout: float | None = None,
) -> dict[str, set[tuple[str, ...]]]:
    """Like :func:`_ground_extensions`, but enforces ``timeout`` (in seconds).

    Raises:
        ClingoTimeoutError: if grounding and solving take longer than ``timeout``.
        RuntimeError: if clingo fails (e.g. a syntax error in ``rules``).
    """
    if timeout is None or "fork" not in multiprocessing.get_all_start_methods():
        if timeout is not None:
            print(
                f"Warning: clingo timeouts require the 'fork' start method, which is "
                f"not available on {sys.platform}. Running without timeout."
            )
        return _ground_extensions(rules, background_facts, target_labels)

    ctx = multiprocessing.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_ground_in_child,
        args=(child_conn, rules, background_facts, target_labels),
        daemon=True,
    )
    proc.start()
    child_conn.close()
    try:
        if not parent_conn.poll(timeout):
            raise ClingoTimeoutError(
                f"Grounding / solving with clingo exceeded the time limit of {timeout}s"
            )
        try:
            status, payload = parent_conn.recv()
        except EOFError:
            raise RuntimeError(
                f"Clingo subprocess terminated unexpectedly (exit code {proc.exitcode})"
            )
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join()
        parent_conn.close()
    if status == "error":
        raise RuntimeError(payload)
    return payload


def evaluate_with_clingo(
    rules: list[str],
    background_facts: list[str],
    target_labels: list[str],
    examples: list,
    timeout: float | None = None,
) -> dict[str, list]:
    """For each target predicate, return the examples ``e`` for which ``target(e)`` is derived.

    Only unary extensions are considered. Targets without any positive example are omitted.
    """
    extensions = ground_extensions(
        rules, background_facts, target_labels, timeout=timeout
    )
    positives = {}
    for target_label in target_labels:
        derived = {
            args[0] for args in extensions.get(target_label, ()) if len(args) == 1
        }
        hits = [example for example in examples if str(example) in derived]
        if hits:
            positives[target_label] = hits
    return positives


def _match_chunk(
    rules: list[str],
    chunk: pd.DataFrame,
    target_label: str,
    allowed_predicates: set[str] | None,
) -> list:
    """IDs (index values) of the molecules in ``chunk`` for which ``target_label`` holds."""
    facts = molecules_to_asp_facts(chunk, allowed_predicates=allowed_predicates)
    positives = evaluate_with_clingo(rules, facts, [target_label], list(chunk.index))
    return positives.get(target_label, [])


def _match_chunk_in_child(conn, rules, chunk, target_label, allowed_predicates):
    try:
        conn.send(("ok", _match_chunk(rules, chunk, target_label, allowed_predicates)))
    except BaseException as e:  # noqa: BLE001 - forward every error to the parent
        conn.send(("error", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def match_chunks_with_clingo(
    rules: list[str],
    chunks: list[pd.DataFrame],
    target_label: str,
    allowed_predicates: set[str] | None = None,
    timeout: float | None = None,
    n_jobs: int = 1,
) -> list[list | ClingoTimeoutError]:
    """Evaluate ``rules`` on several chunks of molecules, in parallel.

    Each chunk is evaluated in a separate forked process (fact generation, grounding
    and solving), with at most ``n_jobs`` processes running at the same time. A chunk
    that exceeds ``timeout`` seconds is killed.

    Args:
        chunks: DataFrames with a ``mol`` column, indexed by molecule IDs (valid clingo
            terms, unique across chunks).

    Returns:
        For each chunk, either the list of IDs of the molecules for which
        ``target_label`` holds, or a :class:`ClingoTimeoutError`.

    Raises:
        RuntimeError: if clingo fails (e.g. a syntax error in ``rules``).
    """
    if "fork" not in multiprocessing.get_all_start_methods():
        if timeout is not None:
            print(
                f"Warning: clingo timeouts require the 'fork' start method, which is "
                f"not available on {sys.platform}. Running without timeout."
            )
        return [
            _match_chunk(rules, chunk, target_label, allowed_predicates)
            for chunk in chunks
        ]

    ctx = multiprocessing.get_context("fork")
    results: list = [None] * len(chunks)
    pending = list(range(len(chunks)))
    running: dict = {}  # chunk index -> (process, connection, deadline)
    try:
        while pending or running:
            while pending and len(running) < max(1, n_jobs):
                idx = pending.pop(0)
                parent_conn, child_conn = ctx.Pipe(duplex=False)
                proc = ctx.Process(
                    target=_match_chunk_in_child,
                    args=(
                        child_conn,
                        rules,
                        chunks[idx],
                        target_label,
                        allowed_predicates,
                    ),
                    daemon=True,
                )
                proc.start()
                child_conn.close()
                deadline = time.monotonic() + timeout if timeout is not None else None
                running[idx] = (proc, parent_conn, deadline)

            ready = multiprocessing.connection.wait(
                [conn for _, conn, _ in running.values()], timeout=0.5
            )
            now = time.monotonic()
            for idx, (proc, conn, deadline) in list(running.items()):
                if conn in ready:
                    try:
                        status, payload = conn.recv()
                    except EOFError:
                        status, payload = (
                            "error",
                            f"Clingo subprocess terminated unexpectedly (exit code {proc.exitcode})",
                        )
                    if status == "error":
                        raise RuntimeError(payload)
                    results[idx] = payload
                elif deadline is not None and now > deadline:
                    results[idx] = ClingoTimeoutError(
                        f"Grounding / solving with clingo exceeded the time limit of {timeout}s"
                    )
                else:
                    continue
                proc.kill()
                proc.join()
                conn.close()
                del running[idx]
    finally:
        for proc, conn, _ in running.values():
            proc.kill()
            proc.join()
            conn.close()
    return results
