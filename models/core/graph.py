# models/core/graph.py
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple, Callable, Any

from pydantic import BaseModel
import ast
import inspect
import textwrap
import yaml
import numpy as np

from graphviz import Digraph

from models.core.types import Input, Calculation, Output, Tensor  # type: ignore

# ------------------ manual dependency annotation ----------------------------


def depends_on(*names: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Attach a manual dependency list to a computed-property function.

    Usage:
        @computed_field
        @property
        @depends_on("capex", "inflation_rate")
        def maintenance_cost(self) -> ModelCalculation:
            ...
    """

    def _wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, "_depends_on", tuple(names))
        return func

    return _wrap


# ------------------ AST-based dependency extraction -------------------------


def _iter_self_attribute_names(py_src: str, valid_names: Set[str]) -> Set[str]:
    """Parse python source and return names accessed as `self.<name>` that are in `valid_names`."""
    try:
        tree = ast.parse(textwrap.dedent(py_src))
    except SyntaxError:
        return set()
    deps: Set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            attr = node.attr
            if attr in valid_names:
                deps.add(attr)
    return deps


@dataclass(frozen=True)
class GraphBuildOptions:
    include_roles: Tuple[str, ...] = ("input", "calc", "output")
    use_manual_first: bool = True  # prefer @depends_on overrides when available


class TensorGraph(BaseModel):
    """Role-indexed tensor collection + dependency graph builder.

    Responsibilities:
      - Discover tensors by role at class-level (cached).
      - Provide instance accessors (names/fields).
      - Build dependency graph for calc/output properties via AST (with optional @depends_on overrides).
      - Render Graphviz diagrams (no UI framework required).
    """

    scenario: str
    years: int
    iterations: int

        # ------ IO ------------------------------------------------
    @staticmethod
    def _load_yaml(path: str) -> dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    @classmethod
    def load_model(cls, yaml_path: str, *, iterations: int = 500, years: int = 25, scenario: str = "Base"):
        """
        Generic YAML to model loader.
        Expects the YAML to be a mapping of field -> param dict (p10/p50/p90, use_fixed, fixed_value, years, etc).
        Passes those as kwargs; your model/types should coerce them to Input/Output tensors.
        """
        cfg = cls._load_yaml(yaml_path)
        # Common knobs your Base/Financial layer typically expects:
        kwargs = dict(cfg)
        kwargs.setdefault("iterations", iterations)
        kwargs.setdefault("years", years)
        kwargs.setdefault("scenario", scenario)
        # Allow model_cls to interpret kwargs appropriately (e.g., via pydantic model/validators).
        model = cls(**kwargs)
        model.update_inputs(yaml_path)
        return model
    
    def update_inputs(self, config_path: str) -> None:
        """
        Reload every ModelInput from the given YAML.
        """
        for name in type(self).role_names_cls("input"):

            mi = Input.from_config(
                self.scenario,
                name,
                years=self.years,
                iterations=self.iterations,
                config_path=config_path,
            )
            setattr(self, name, mi)


    # --------- Role discovery (class-level, cached) ---------

    @staticmethod
    def _ann_matches_role(
        ann: Any, role_types: Tuple[type, ...], type_names: Set[str]
    ) -> bool:
        # forward-ref string
        if isinstance(ann, str):
            return ann in type_names
        # direct or subclass match
        if isinstance(ann, type):
            return any(ann is t or issubclass(ann, t) for t in role_types)
        return False

    @classmethod
    @lru_cache(maxsize=None)
    def role_names_cls(cls, role: str) -> Tuple[str, ...]:
        """Names of attributes (declared + computed) for `role` on this class."""
        role = role.lower()
        role_types: Tuple[type, ...]
        if role == "input":
            role_types = (Input,)
        elif role == "calc":
            role_types = (Calculation,)
        elif role == "output":
            role_types = (Output,)
        else:
            # Unknown role: nothing to match
            return tuple()

        type_names = {t.__name__ for t in role_types}

        names: List[str] = []

        # Declared fields (Pydantic model_fields)
        for name, finfo in getattr(cls, "model_fields", {}).items():
            ann = getattr(finfo, "annotation", None)
            if cls._ann_matches_role(ann, role_types, type_names):
                names.append(name)

        # Computed fields (Pydantic model_computed_fields)
        for name, cinfo in getattr(cls, "model_computed_fields", {}).items():
            ann = getattr(cinfo, "return_type", None)
            if cls._ann_matches_role(ann, role_types, type_names):
                names.append(name)

        return tuple(names)

    @classmethod
    @lru_cache(maxsize=None)
    def role_fields_cls(cls, role: str) -> Dict[str, object]:
        """Class attribute objects for `role` (may include property objects)."""
        return {name: getattr(cls, name) for name in cls.role_names_cls(role)}

    # --------- Instance accessors ---------

    def role_names(self, role: str) -> List[str]:
        return list(type(self).role_names_cls(role))

    def role_fields(self, role: str) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        for name in self.role_names(role):
            val = getattr(self, name, None)
            if isinstance(val, Tensor):
                out[name] = val
        return out

    # Common shortcuts
    @property
    def input_names(self) -> List[str]:
        return self.role_names("input")

    @property
    def calc_names(self) -> List[str]:
        return self.role_names("calc")

    @property
    def output_names(self) -> List[str]:
        return self.role_names("output")

    # --------- Dependency graph ---------

    @classmethod
    @lru_cache(maxsize=None)
    def _valid_graph_names_cls(cls) -> Set[str]:
        """All names eligible for the graph (inputs + calcs + outputs)."""
        names: Set[str] = set()
        for role in ("input", "calc", "output"):
            names.update(cls.role_names_cls(role))
        return names

    @classmethod
    @lru_cache(maxsize=None)
    def _prop_dependencies_cls(cls, prop_name: str) -> Tuple[str, ...]:
        """Extract dependencies for a computed property via @depends_on or AST."""
        valid = cls._valid_graph_names_cls()
        # Locate fget of property/descriptor
        fget = None
        try:
            attr = getattr(cls, prop_name)
            fget = getattr(attr, "fget", None) or attr  # property or function
        except Exception:
            pass

        # Manual override?
        if fget is not None and hasattr(fget, "_depends_on"):
            return tuple([n for n in getattr(fget, "_depends_on") if n in valid])

        # Fallback to AST parsing
        if fget is None:
            return tuple()

        try:
            src = inspect.getsource(fget)
        except (
            OSError,
            TypeError,
        ):  # source may be unavailable (C-extensions, REPL, etc.)
            return tuple()
        deps = _iter_self_attribute_names(src, valid)
        return tuple(sorted(deps))

    def build_dependency_graph(
        self, *, options: Optional[GraphBuildOptions] = None
    ) -> Dict[str, Set[str]]:
        """Return adjacency: name -> set(dependencies). Only for calc/output by default."""
        if options is None:
            options = GraphBuildOptions()

        valid = type(self)._valid_graph_names_cls()

        targets: List[str] = []
        for role in options.include_roles:
            targets.extend(self.role_names(role))

        deps_map: Dict[str, Set[str]] = {t: set() for t in targets}
        for name in targets:
            deps = set(type(self)._prop_dependencies_cls(name))
            deps &= valid
            deps.discard(name)  # no self-deps
            deps_map[name] = deps

        return deps_map

    # --------- Analysis helpers ---------

    def find_cycles(
        self, deps: Optional[Dict[str, Set[str]]] = None
    ) -> List[List[str]]:
        """Detect simple cycles using DFS; return list of cycles (as node lists)."""
        if deps is None:
            deps = self.build_dependency_graph()

        visited: Dict[str, int] = {}  # 0=unseen, 1=visiting, 2=done
        stack: List[str] = []
        cycles: List[List[str]] = []

        def dfs(u: str) -> None:
            state = visited.get(u, 0)
            if state == 1:
                # Found a back-edge; slice stack to form a cycle
                if u in stack:
                    i = stack.index(u)
                    cycles.append(stack[i:] + [u])
                return
            if state == 2:
                return

            visited[u] = 1
            stack.append(u)
            for v in deps.get(u, ()):
                dfs(v)
            stack.pop()
            visited[u] = 2

        for node in deps.keys():
            if visited.get(node, 0) == 0:
                dfs(node)

        return cycles

    def topo_order(self, deps: Optional[Dict[str, Set[str]]] = None) -> List[str]:
        """Topological order (Kahn). Raises if cycles present."""
        if deps is None:
            deps = self.build_dependency_graph()

        # Build reverse edges
        rev: Dict[str, Set[str]] = {u: set() for u in deps}
        indeg: Dict[str, int] = {u: len(deps[u]) for u in deps}
        for u, vs in deps.items():
            for v in vs:
                rev.setdefault(v, set()).add(u)

        order: List[str] = [u for u, d in indeg.items() if d == 0]
        out: List[str] = []
        while order:
            u = order.pop()
            out.append(u)
            for w in rev.get(u, ()):
                indeg[w] -= 1
                if indeg[w] == 0:
                    order.append(w)

        if len(out) != len(indeg):
            raise ValueError("Cycle detected; cannot compute topological order")
        return out

    # --------- Graphviz rendering ---------

    def trace_calculation_graph(
        self,
        filename: Optional[str] = None,
        *,
        format: str = "png",
        rankdir: str = "LR",
        cluster_roles: bool = True,
        show_orphans: bool = True,
    ) -> "Digraph":
        """Render a dependency graph with roles, cycles, and orphans highlighted."""
        if Digraph is None:
            raise RuntimeError("graphviz is not installed")

        deps = self.build_dependency_graph()
        inputs = set(self.input_names)
        calcs = set(self.calc_names)
        outputs = set(self.output_names)
        all_nodes = inputs | calcs | outputs

        dot = Digraph(name="TensorGraph", format=format)
        dot.attr(rankdir=rankdir, fontsize="10")

        if cluster_roles:

            def cluster(name: str, nodes: Set[str], *, shape: str, fill: str):
                if not nodes:
                    return
                with dot.subgraph(name=f"cluster_{name}") as c:
                    c.attr(label=name.capitalize(), style="rounded", color="gray70")
                    for n in sorted(nodes):
                        c.node(n, shape=shape, style="filled", fillcolor=fill)

            cluster("inputs", inputs, shape="box", fill="lightgray")
            cluster("calcs", calcs, shape="ellipse", fill="lightblue")
            cluster("outputs", outputs, shape="ellipse", fill="white")
        else:
            for n in sorted(inputs):
                dot.node(n, shape="box", style="filled", fillcolor="lightgray")
            for n in sorted(calcs):
                dot.node(n, shape="ellipse", style="filled", fillcolor="lightblue")
            for n in sorted(outputs):
                dot.node(n, shape="ellipse", style="filled", fillcolor="white")

        # Edges
        for tgt, srcs in deps.items():
            for src in srcs:
                if src in all_nodes and tgt in all_nodes:
                    dot.edge(src, tgt)

        # Orphans (nodes with no in/out edges)
        if show_orphans:
            connected = set()
            for tgt, srcs in deps.items():
                if srcs:
                    connected.add(tgt)
                    connected |= set(srcs)
            for n in sorted(all_nodes - connected):
                dot.node(
                    n,
                    shape="diamond",
                    style="filled",
                    fillcolor="gold",
                    xlabel="orphan",
                )

        # Cycle highlighting: add red edges for back-edges inside cycles
        cycles = self.find_cycles(deps)
        for cyc in cycles:
            for i in range(len(cyc) - 1):
                dot.edge(cyc[i], cyc[i + 1], color="red")

        if filename:
            dot.render(filename, cleanup=True)

        return dot
