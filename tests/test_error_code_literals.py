"""Error codes come from vl.errors.ErrorCode, not string literals, in every src/vl module."""

import ast
import re
from pathlib import Path

SRC = Path(__file__).parent.parent / "src" / "vl"
DEFINITION = SRC / "errors.py"  # the one module that defines the code strings


def code_literals(path: Path) -> list[str]:
    """Return the ``VL-nnn`` string constants in one module (docstrings do not match).

    Args:
        path: The Python source file to scan.

    Returns:
        The literals found, in source order.
    """
    return [
        node.value
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and re.fullmatch(r"VL-\d{3}", node.value)
    ]


def test_no_error_code_string_literal() -> None:
    modules = [path for path in sorted(SRC.rglob("*.py")) if path != DEFINITION]
    assert len(modules) > 10  # the scan really walks the package
    for path in modules:
        assert code_literals(path) == [], str(path.relative_to(SRC))
