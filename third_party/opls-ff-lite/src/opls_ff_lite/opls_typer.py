"""OPLS-AA atom typing from SMARTS rules in the Foyer dialect, without Foyer.

This reproduces Foyer 1.2.0's typing algorithm (``foyer.atomtyper``,
``foyer.smarts_graph``, ``foyer.smarts``; MIT License, mosdef-hub/foyer)
on top of RDKit, so that the project's rule data (``data/opls_rules.json``,
exported from Foyer's ``oplsaa.xml``, plus the atom-type archive) types
exactly as before. Reproduced behaviour, including Foyer's quirks:

- SMARTS dialect (Foyer's lark grammar): elements by atomic number (no
  aromatic lower case), ``#n``, ``X`` (neighbour count), ``r`` (ring size),
  ``R`` (ring count), ``%type`` (atom already whitelisted for that type),
  ``!`` on a primitive, ``&`` > ``,`` > ``;`` precedence; ring closures.
- Pattern graph construction as in ``SMARTSGraph._add_edges``: an unspecified
  bond matches any bond, an explicit bond order persists along a chain, and a
  branch takes the first bond symbol that lark's ``find_data`` returns for it.
- Matching is node-induced subgraph isomorphism (networkx VF2): pattern atoms
  that are not bonded in the pattern must not be bonded in the molecule.
- Rings: Foyer's ``_find_chordless_cycles`` (max size 8), ported line by line.
- Iteration: rules in table order, repeated until no whitelist changes (at
  most 10 passes); each new whitelist entry adds the rule's overrides to the
  atom's blacklist. The type is whitelist minus blacklist; more than one
  candidate is an error, none leaves the atom untyped.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from functools import lru_cache

from rdkit import Chem

MAX_ITERATIONS = 10
MAX_CYCLE_SIZE = 8
_PERIODIC_TABLE = Chem.GetPeriodicTable()
# Foyer's SYMBOL terminal (canonical elements only).
_SYMBOL = re.compile(
    r"C[laroudsemf]?|Os?|N[eaibdpos]?|S[icernbmg]?|P[drmtboau]?|H[eofgas]?|A[lrsgutcm]|"
    r"B[eraik]?|Dy|E[urs]|F[erm]?|G[aed]|I[nr]?|Kr?|L[iaur]|M[gnodt]|R[buhenaf]|"
    r"T[icebmalh]|U|V|W|Xe|Yb?|Z[nr]"
)
_BOND_ORDERS = {"~": 0.0, "-": 1.0, "=": 2.0, "#": 3.0, ":": 1.5}
_BOND_NAMES = {"~": "wildcard_bond", "-": "single_bond", "=": "double_bond",
               "#": "triple_bond", ":": "aromatic_bond"}
_ORDER_OF_NAME = {name: _BOND_ORDERS[symbol] for symbol, name in _BOND_NAMES.items()}


class TypingError(ValueError):
    """Invalid SMARTS, or an atom with more than one candidate type."""


# ---------------------------------------------------------------- parse tree --
class Tree:
    """Minimal stand-in for lark's Tree (same traversal orders)."""

    __slots__ = ("data", "children")

    def __init__(self, data: str, children: list):
        self.data = data
        self.children = children

    def iter_subtrees(self):
        """lark ``Tree.iter_subtrees``: reversed breadth-first order."""
        queue = [self]
        seen = {}
        for subtree in queue:
            seen[id(subtree)] = subtree
            queue += [c for c in reversed(subtree.children)
                      if isinstance(c, Tree) and id(c) not in seen]
        return reversed(list(seen.values()))

    def iter_subtrees_topdown(self):
        """lark ``Tree.iter_subtrees_topdown``: depth-first pre-order."""
        stack = [self]
        while stack:
            node = stack.pop()
            if not isinstance(node, Tree):
                continue
            yield node
            stack.extend(reversed(node.children))

    def find_data(self, data: str):
        return (t for t in self.iter_subtrees() if t.data == data)


class _Parser:
    """Recursive-descent parser for Foyer's SMARTS grammar (lark LALR rules,
    with ``_rule`` inlined and ``?rule`` collapsed when it has one child)."""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def error(self, message: str) -> TypingError:
        return TypingError(f"SMARTS {self.text!r}, position {self.pos}: {message}")

    def peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def take(self, expected: str) -> None:
        if not self.text.startswith(expected, self.pos):
            raise self.error(f"expected {expected!r}")
        self.pos += len(expected)

    def number(self) -> str:
        match = re.compile(r"\d+").match(self.text, self.pos)
        if match is None:
            raise self.error("expected a number")
        self.pos = match.end()
        return match.group()

    def parse(self) -> Tree:
        children = self.string()
        if self.pos != len(self.text):
            raise self.error("unexpected character")
        return Tree("start", children)

    def string(self) -> list:
        # _string: _chain _nonlastbranch* _lastbranch?
        items = self.chain()
        while self.peek() == "(":
            self.take("(")
            items.append(Tree("branch", self.branch_body()))
            self.take(")")
        if self.peek() and self.peek() != ")":
            items.append(Tree("branch", self.branch_body()))
        return items

    def branch_body(self) -> list:
        # branch: bond_symbol? _string
        items = []
        if self.peek() in _BOND_ORDERS:
            items.append(self.bond())
        return items + self.string()

    def chain(self) -> list:
        # _chain: atom (bond_symbol? atom)*
        items = [self.atom()]
        while True:
            if self.peek() in _BOND_ORDERS:
                items.append(self.bond())
                items.append(self.atom())
            elif self.peek() == "[" or self.peek() == "*" or _SYMBOL.match(self.text, self.pos):
                items.append(self.atom())
            else:
                return items

    def bond(self) -> Tree:
        symbol = self.peek()
        self.pos += 1
        return Tree("bond_symbol", [Tree(_BOND_NAMES[symbol], [])])

    def atom(self) -> Tree:
        # atom: ("[" weak_and_expression "]" | atom_symbol) atom_label?
        if self.peek() == "[":
            self.take("[")
            expression = self.weak_and()
            self.take("]")
        else:
            expression = self.symbol()
        children = [expression]
        if self.peek().isdigit():
            children.append(Tree("atom_label", [self.number()]))
        return Tree("atom", children)

    def symbol(self) -> Tree:
        if self.peek() == "*":
            self.pos += 1
            return Tree("atom_symbol", ["*"])
        match = _SYMBOL.match(self.text, self.pos)
        if match is None:
            raise self.error("expected an element symbol")
        self.pos = match.end()
        return Tree("atom_symbol", [match.group()])

    def weak_and(self) -> Tree:
        node = self.or_()
        while self.peek() == ";":
            self.pos += 1
            node = Tree("weak_and_expression", [node, self.or_()])
        return node

    def or_(self) -> Tree:
        node = self.and_()
        while self.peek() == ",":
            self.pos += 1
            node = Tree("or_expression", [node, self.and_()])
        return node

    def and_(self) -> Tree:
        node = self.unary()
        while self.peek() == "&":
            self.pos += 1
            node = Tree("and_expression", [node, self.unary()])
        return node

    def unary(self) -> Tree:
        if self.peek() == "!":
            self.pos += 1
            return Tree("not_expression", [self.atom_id()])
        return self.atom_id()

    def atom_id(self) -> Tree:
        text, pos = self.text, self.pos
        if text.startswith("$(", pos):
            raise self.error("recursive SMARTS $(...) is not supported (as in Foyer)")
        if self.peek() == "#":
            self.pos += 1
            return Tree("atom_id", [Tree("atomic_num", [self.number()])])
        if self.peek() == "%":
            match = re.compile(r"%[A-Za-z_0-9]+").match(text, pos)
            self.pos = match.end()
            return Tree("atom_id", [Tree("has_label", [match.group()])])
        if self.peek() == "X":
            self.pos += 1
            return Tree("atom_id", [Tree("neighbor_count", [self.number()])])
        if self.peek() == "r":
            self.pos += 1
            return Tree("atom_id", [Tree("ring_size", [self.number()])])
        if self.peek() == "R" and pos + 1 < len(text) and text[pos + 1].isdigit():
            self.pos += 1
            return Tree("atom_id", [Tree("ring_count", [self.number()])])
        return Tree("atom_id", [self.symbol()])


# ------------------------------------------------------------ pattern graph --
@dataclass
class Pattern:
    """A SMARTS rule as a graph: node expressions and edges with bond orders
    (0 = any bond), built exactly as ``foyer.smarts_graph.SMARTSGraph``."""

    name: str
    smarts: str
    overrides: frozenset
    nodes: list = field(default_factory=list)  # expression trees
    edges: dict = field(default_factory=dict)  # (i, j), i < j -> bond order
    prefilter: str | None = None  # Foyer's element filter of node 0 (see find_atomtypes)
    query: Chem.Mol | None = None

    @property
    def uses_rings(self) -> bool:
        return any(t.data in ("ring_size", "ring_count")
                   for node in self.nodes for t in node.iter_subtrees())


def _add_edge(edges: dict, a: int, b: int, order: float | None) -> None:
    key = (min(a, b), max(a, b))
    if order is None:  # ring-closure edge: no bond order unless already bonded
        edges.setdefault(key, 0.0)
    else:
        edges[key] = order


def build_pattern(name: str, smarts: str, overrides=()) -> Pattern:
    ast = _Parser(smarts).parse()
    atoms = [t for t in ast.iter_subtrees_topdown() if t.data == "atom"]
    index = {id(atom): n for n, atom in enumerate(atoms)}
    pattern = Pattern(name, smarts, frozenset(overrides), [a.children[0] for a in atoms])

    def add_edges(node: Tree, trunk: Tree | None = None, bond_order: float = 0.0) -> None:
        # Literal port of SMARTSGraph._add_edges.
        for child in node.children:
            if child.data == "atom":
                if trunk is not None:
                    bond_data = list(child.find_data("bond_symbol"))
                    if bond_data:
                        bond_order = _ORDER_OF_NAME[bond_data[0].children[0].data]
                    _add_edge(pattern.edges, index[id(child)], index[id(trunk)], bond_order)
                trunk = child
            elif child.data == "branch":
                bond_data = list(child.find_data("bond_symbol"))
                if bond_data:
                    bond_order = _ORDER_OF_NAME[bond_data[0].children[0].data]
                add_edges(child, trunk, bond_order=bond_order)
                bond_order = 0.0
            elif child.data == "bond_symbol":
                bond_order = _ORDER_OF_NAME[child.children[0].data]

    add_edges(ast)
    labels: dict[str, list[int]] = {}
    for n, atom in enumerate(atoms):
        for label in atom.find_data("atom_label"):
            for digit in label.children[0]:
                labels.setdefault(digit, []).append(n)
    for digit, members in labels.items():
        if len(members) != 2:
            raise TypingError(f"SMARTS {smarts!r}: ring closure {digit} is not a pair")
        _add_edge(pattern.edges, members[0], members[1], None)

    pattern.prefilter = _prefilter_element(atoms[0])
    pattern.query = _query(pattern)
    return pattern


def _prefilter_element(atom: Tree) -> str | None:
    """Foyer's rule filter (``find_atomtypes``): when the first atom has exactly
    one element symbol and no negation, the rule only runs on molecules that
    contain that element; a lone ``*`` therefore never runs."""
    symbols = list(atom.find_data("atom_symbol"))
    if len(symbols) == 1 and not list(atom.find_data("not_expression")):
        return symbols[0].children[0]
    return None


def _conjuncts(expression: Tree):
    if expression.data in ("weak_and_expression", "and_expression"):
        for child in expression.children:
            yield from _conjuncts(child)
    else:
        yield expression


def _fixed_element(expression: Tree) -> int | None:
    """Atomic number every match must have (a positive element conjunct)."""
    for term in _conjuncts(expression):
        child = term.children[0] if term.data == "atom_id" else term
        if child.data == "atom_symbol" and child.children[0] != "*":
            return _PERIODIC_TABLE.GetAtomicNumber(child.children[0])
        if child.data == "atomic_num":
            return int(child.children[0])
    return None


def _query(pattern: Pattern) -> Chem.Mol:
    """RDKit query of the pattern skeleton: fixed elements and bond orders.

    Everything else (and the induced-subgraph condition) is checked on each
    match by ``_match_is_valid``."""
    query = Chem.RWMol()
    for expression in pattern.nodes:
        element = _fixed_element(expression)
        query.AddAtom(Chem.AtomFromSmarts(f"[#{element}]" if element is not None else "*"))
    symbols = {0.0: "~", 1.0: "-", 1.5: ":", 2.0: "=", 3.0: "#"}
    for (a, b), order in pattern.edges.items():
        query.AddBond(a, b, Chem.BondType.SINGLE)
        query.ReplaceBond(query.GetBondBetweenAtoms(a, b).GetIdx(),
                          Chem.BondFromSmarts(symbols[order]))
    return query.GetMol()


# --------------------------------------------------------------- the molecule --
class _Host:
    """Molecule data used by the matcher (atoms, neighbours in bond order,
    Foyer's chordless cycles, white- and blacklists)."""

    def __init__(self, molecule: Chem.Mol):
        self.molecule = molecule
        n = molecule.GetNumAtoms()
        self.atomic_numbers = [a.GetAtomicNum() for a in molecule.GetAtoms()]
        self.neighbors: list[list[int]] = [[] for _ in range(n)]
        for bond in molecule.GetBonds():  # insertion order, as networkx neighbours
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            self.neighbors[a].append(b)
            self.neighbors[b].append(a)
        self.bonded = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                       for b in molecule.GetBonds()}
        self._cycles: list[set] | None = None
        self.whitelist: list[set[str]] = [set() for _ in range(n)]
        self.blacklist: list[set[str]] = [set() for _ in range(n)]

    @property
    def cycles(self) -> list[set]:
        if self._cycles is None:
            self._cycles = [set(map(tuple, c)) for c in _find_chordless_cycles(self.neighbors)]
        return self._cycles


def _find_chordless_cycles(neighbors: list[list[int]]) -> list[list[list[int]]]:
    """Port of foyer.smarts_graph._find_chordless_cycles (max_cycle_size 8)."""

    def has_edge(a: int, b: int) -> bool:
        return b in neighbors[a]

    cycles: list[list[list[int]]] = [[] for _ in neighbors]
    for node in range(len(neighbors)):
        for pair in itertools.combinations(neighbors[node], 2):
            connected = False
            last_node = pair[0]
            ring = [last_node, node, pair[1]]
            possible_rings = [ring]
            if has_edge(last_node, pair[1]):
                cycles[node].append(ring)
                connected = True
            while not connected:
                new_possible_rings = []
                for possible_ring in possible_rings:
                    for next_neighbor in neighbors[possible_ring[-1]]:
                        if next_neighbor != possible_ring[-2]:
                            new_possible_rings.append(possible_ring + [next_neighbor])
                possible_rings = new_possible_rings
                for possible_ring in possible_rings:
                    if has_edge(possible_ring[-1], last_node):
                        if not any(has_edge(possible_ring[-1], internal)
                                   for internal in possible_ring[1:-2]):
                            cycles[node].append(possible_ring)
                            connected = True
                if not possible_rings or len(possible_rings[0]) == MAX_CYCLE_SIZE:
                    break
    return cycles


def _expression_matches(expression: Tree, host: _Host, atom: int) -> bool:
    data = expression.data
    if data == "not_expression":
        return not _expression_matches(expression.children[0], host, atom)
    if data in ("and_expression", "weak_and_expression"):
        return (_expression_matches(expression.children[0], host, atom)
                and _expression_matches(expression.children[1], host, atom))
    if data == "or_expression":
        return (_expression_matches(expression.children[0], host, atom)
                or _expression_matches(expression.children[1], host, atom))
    primitive = expression.children[0] if data == "atom_id" else expression
    kind, value = primitive.data, primitive.children[0]
    if kind == "atom_symbol":
        return value == "*" or host.atomic_numbers[atom] == _PERIODIC_TABLE.GetAtomicNumber(value)
    if kind == "atomic_num":
        return host.atomic_numbers[atom] == int(value)
    if kind == "has_label":
        return value[1:] in host.whitelist[atom]
    if kind == "neighbor_count":
        return len(host.neighbors[atom]) == int(value)
    if kind == "ring_size":
        return any(len(cycle) == int(value) for cycle in host.cycles[atom])
    if kind == "ring_count":
        return len(host.cycles[atom]) == int(value)
    raise TypingError(f"Unsupported SMARTS primitive {kind!r}")


def _match_is_valid(pattern: Pattern, host: _Host, match: tuple[int, ...]) -> bool:
    n = len(match)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in pattern.edges and frozenset((match[i], match[j])) in host.bonded:
                return False  # node-induced subgraph isomorphism, as networkx VF2
    return all(_expression_matches(expression, host, match[i])
               for i, expression in enumerate(pattern.nodes))


def _matched_atoms(pattern: Pattern, host: _Host) -> list[int]:
    found: list[int] = []
    for match in host.molecule.GetSubstructMatches(
        pattern.query, uniquify=False, maxMatches=10_000_000
    ):
        if match[0] not in found and _match_is_valid(pattern, host, match):
            found.append(match[0])
    return found


# ------------------------------------------------------------------- typing --
@lru_cache(maxsize=4096)
def compiled(name: str, smarts: str, overrides: frozenset) -> Pattern:
    return build_pattern(name, smarts, overrides)


def find_atomtypes(
    molecule: Chem.Mol, definitions: dict[str, str], overrides: dict[str, set[str]]
) -> list[dict]:
    """Foyer's ``find_atomtypes`` + iteration: per atom the whitelist and
    blacklist of type names. ``definitions`` is ordered (type -> SMARTS)."""
    host = _Host(molecule)
    elements = {_PERIODIC_TABLE.GetElementSymbol(n) for n in host.atomic_numbers}
    rules = []
    for name, smarts in definitions.items():
        if not smarts:
            continue
        pattern = compiled(name, smarts, frozenset(overrides.get(name, ())))
        if pattern.prefilter is None or pattern.prefilter in elements:
            rules.append(pattern)
    for _ in range(MAX_ITERATIONS):
        found_something = False
        for pattern in rules:
            for atom in _matched_atoms(pattern, host):
                if pattern.name not in host.whitelist[atom]:
                    host.whitelist[atom].add(pattern.name)
                    host.blacklist[atom] |= pattern.overrides
                    found_something = True
        if not found_something:
            break
    return [{"whitelist": w, "blacklist": b} for w, b in zip(host.whitelist, host.blacklist)]


def resolve_atomtypes(typemap: list[dict]) -> list[str | None]:
    """Type = whitelist - blacklist; None if empty, error if ambiguous."""
    types = []
    for index, atom in enumerate(typemap):
        candidates = atom["whitelist"] - atom["blacklist"]
        if len(candidates) > 1:
            raise TypingError(f"Found multiple types for atom {index}: {sorted(candidates)}.")
        types.append(next(iter(candidates)) if candidates else None)
    return types
