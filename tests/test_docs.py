"""Documentation checks: the required sections exist and the README names real make targets."""

import re
from pathlib import Path

from vl.errors import ErrorCode

ROOT = Path(__file__).resolve().parent.parent


def test_required_doc_sections_exist() -> None:
    codes = (ROOT / "docs/error-codes.md").read_text()
    assert [c.value for c in ErrorCode if c.value not in codes] == []
    notes = (ROOT / "docs/release-notes.md").read_text()
    for heading in ("Product description", "Usage", "Known issues", "Coming soon"):
        assert re.search(rf"^## {heading}\s*$", notes, re.M), heading
    architecture = (ROOT / "docs/architecture.md").read_text()
    section = architecture.split("## Use cases", 1)[1].split("\n## ", 1)[0]
    assert len(re.findall(r"^\d+\. ", section, re.M)) >= 3


def test_readme_make_targets_exist_in_the_makefile() -> None:
    readme = (ROOT / "README.md").read_text()
    mentioned = set(re.findall(r"\bmake ([a-z][a-z0-9-]*)", readme))
    defined = set(re.findall(r"^([a-z][a-z0-9-]*):", (ROOT / "Makefile").read_text(), re.M))
    assert mentioned, "README mentions no make target"
    assert mentioned <= defined, sorted(mentioned - defined)
