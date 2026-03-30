from __future__ import annotations

import ast
from pathlib import Path


TARGET_MODULES = [
    Path(r"D:\EpiGraph_PH\src\epigraph_ph\core\node_graph.py"),
    Path(r"D:\EpiGraph_PH\src\epigraph_ph\core\province_archetypes.py"),
    Path(r"D:\EpiGraph_PH\src\epigraph_ph\phase1\pipeline.py"),
    Path(r"D:\EpiGraph_PH\src\epigraph_ph\phase2\pipeline.py"),
    Path(r"D:\EpiGraph_PH\src\epigraph_ph\phase15\pipeline.py"),
    Path(r"D:\EpiGraph_PH\src\epigraph_ph\phase3\pipeline.py"),
    Path(r"D:\EpiGraph_PH\src\epigraph_ph\phase4\pipeline.py"),
    Path(r"D:\EpiGraph_PH\src\epigraph_ph\validate\literature_review.py"),
]


def _numeric_literal(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        if isinstance(node.operand.value, (int, float)) and not isinstance(node.operand.value, bool):
            return -float(node.operand.value)
    return None


def _container_numeric_issues(path: Path) -> list[tuple[int, list[float]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    issues: list[tuple[int, list[float]]] = []

    class Visitor(ast.NodeVisitor):
        def _capture(self, lineno: int, values: list[float]) -> None:
            filtered = [value for value in values if value not in {0.0, 1.0, -1.0}]
            if len(filtered) >= 4:
                issues.append((lineno, filtered))

        def visit_List(self, node: ast.List) -> None:
            self._capture(node.lineno, [_numeric_literal(item) for item in node.elts if _numeric_literal(item) is not None])
            self.generic_visit(node)

        def visit_Tuple(self, node: ast.Tuple) -> None:
            self._capture(node.lineno, [_numeric_literal(item) for item in node.elts if _numeric_literal(item) is not None])
            self.generic_visit(node)

        def visit_Set(self, node: ast.Set) -> None:
            self._capture(node.lineno, [_numeric_literal(item) for item in node.elts if _numeric_literal(item) is not None])
            self.generic_visit(node)

        def visit_Dict(self, node: ast.Dict) -> None:
            self._capture(node.lineno, [_numeric_literal(item) for item in node.values if _numeric_literal(item) is not None])
            self.generic_visit(node)

    Visitor().visit(tree)
    return issues


def test_behavior_tables_are_plugin_owned_in_high_risk_modules() -> None:
    for path in TARGET_MODULES:
        issues = _container_numeric_issues(path)
        assert issues == [], f"{path} still contains inline numeric lookup tables: {issues}"
