#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repository-wide validation to detect stale code and broken internal imports.

Checks:
1) Syntax compile every .py file (fast failure on syntax errors).
2) Import-time sanity for key internal packages (utils, data_ingestion, etc.).
3) AST-based internal import verification:
   - For "from X import a, b, c" where X is internal (e.g., utils.microstructure),
     assert that each symbol exists in module X at runtime.
4) Verify utils.microstructure.__all__ contains the required export surface.

Exit nonzero on any failure; CI will go red.
"""

from __future__ import annotations

import ast
import importlib
import os
import sys
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = os.path.abspath(os.getcwd())
IGNORE_DIRS = {".git", ".github", "__pycache__", ".venv", "venv", "env", ".mypy_cache", ".ruff_cache"}

# Internal module roots considered "our" code
INTERNAL_ROOTS = {"utils", "data_ingestion", "prepare_training_data", "ml_classifier", "config"}

REQUIRED_MICRO_EXPORTS = {
    "calc_spread",
    "midprice",
    "micro_price",
    "imbalance",
    "add_l1_features",
    "compute_l1_metrics",
    "compute_ofi",
    "compute_trade_signed_volume",
    "compute_vpin",
}

def fail(msg: str, code: int = 1) -> None:
    print(f"[REPO-FAIL] {msg}", file=sys.stderr)
    sys.exit(code)

def ok(msg: str) -> None:
    print(f"[REPO-OK] {msg}")

def iter_py_files(root: str) -> List[str]:
    files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                files.append(os.path.join(dirpath, fn))
    return files

def rel_module_name(py_path: str) -> str | None:
    rel = os.path.relpath(py_path, PROJECT_ROOT)
    if rel.startswith(".."):
        return None
    parts = rel.replace(os.sep, "/").split("/")
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    if parts and parts[0] in INTERNAL_ROOTS:
        return ".".join([p for p in parts if p])
    return None

def syntax_check(py_file: str) -> None:
    with open(py_file, "rb") as f:
        src = f.read()
    try:
        compile(src, py_file, "exec")
    except SyntaxError as e:
        fail(f"Syntax error in {py_file}: {e}")

def parse_internal_imports(py_file: str) -> List[Tuple[str, List[str]]]:
    with open(py_file, "r", encoding="utf-8") as f:
        src = f.read()
    try:
        tree = ast.parse(src, filename=py_file)
    except SyntaxError:
        return []
    imports: List[Tuple[str, List[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in INTERNAL_ROOTS:
                names = [n.name for n in node.names if isinstance(n, ast.alias)]
                imports.append((node.module, names))
    return imports

def verify_import_symbols(module_name: str, names: List[str], origin: str) -> List[str]:
    missing: List[str] = []
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        fail(f"Import error: from {module_name} ... in {origin} -> {e}")
    for n in names:
        if not hasattr(mod, n):
            missing.append(n)
    return missing

def import_sanity(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except Exception as e:
        fail(f"Import-time error for {module_name}: {e}")

def verify_microstructure_exports() -> None:
    try:
        micro = importlib.import_module("utils.microstructure")
    except Exception as e:
        fail(f"Cannot import utils.microstructure: {e}")
    maybe_all = getattr(micro, "__all__", None)
    if maybe_all is not None:
        present = set(maybe_all)
    else:
        present = {name for name in REQUIRED_MICRO_EXPORTS if hasattr(micro, name)}
    missing = sorted(REQUIRED_MICRO_EXPORTS - present)
    if missing:
        fail(f"utils.microstructure missing exports: {missing}")

def main() -> None:
    py_files = iter_py_files(PROJECT_ROOT)
    if not py_files:
        fail("No Python files found to validate")
    for f in py_files:
        syntax_check(f)
    ok(f"Syntax check OK for {len(py_files)} files")

    for mod in sorted(INTERNAL_ROOTS):
        if os.path.exists(os.path.join(PROJECT_ROOT, mod)):
            import_sanity(mod)
    ok("Top-level internal modules import OK")

    total_checks = 0
    for f in py_files:
        imports = parse_internal_imports(f)
        for module_name, names in imports:
            missing = verify_import_symbols(module_name, names, origin=f)
            total_checks += len(names)
            if missing:
                fail(f"Missing symbols in {module_name} imported by {f}: {missing}")
    ok(f"Internal import symbol checks OK ({total_checks} symbols verified)")

    verify_microstructure_exports()
    ok("utils.microstructure export surface OK")

    print("[REPO-OK] Repository validation passed.")

if __name__ == "__main__":
    main()
