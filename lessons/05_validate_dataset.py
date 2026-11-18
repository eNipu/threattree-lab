"""Validate the curated attack-tree dataset before splitting or training."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR.parent / "data" / "curated" / "synthetic_attack_trees.jsonl"

REQUIRED_FIELDS = {
    "id",
    "title",
    "task",
    "system",
    "assets",
    "assumptions",
    "attack_tree",
    "mitigations",
    "uncertainty",
    "evidence_refs",
    "review_status",
    "split",
}
STRING_FIELDS = ("id", "title", "task", "system", "review_status", "split")
LIST_FIELDS = ("assets", "assumptions", "mitigations", "uncertainty", "evidence_refs")
NODE_TYPES = {"goal", "attack"}
NODE_OPERATORS = {"OR", "AND", "LEAF"}
EDGE_RELATIONS = {"OR", "AND"}
VALID_SPLITS = {"train_candidate", "train", "validation", "test"}
VALID_REVIEW_STATUSES = {"synthetic_unreviewed", "reviewed"}
SENSITIVE_PATTERNS = {
    "absolute macOS home path": re.compile(r"/Users/[\w.\-]+"),
    "absolute Linux home path": re.compile(r"/home/[\w.\-]+"),
    "absolute Windows user path": re.compile(r"[A-Za-z]:[\\/](?:Users|home)[\\/][\w.\-]+"),
    "email address": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
}


def is_nonempty_str(value: Any) -> bool:
    """Return True only for non-blank strings."""
    return isinstance(value, str) and bool(value.strip())


def check_string_list(value: Any, field: str, prefix: str, errors: list[str]) -> None:
    """Validate a list of non-blank strings."""
    if not isinstance(value, list) or not value:
        errors.append(f"{prefix}: '{field}' must be a non-empty list of strings")
        return
    for index, item in enumerate(value, start=1):
        if not is_nonempty_str(item):
            errors.append(f"{prefix}: '{field}' item {index} must be a non-blank string")


def check_risk_value(node: dict[str, Any], field: str, prefix: str, errors: list[str]) -> None:
    """Validate an optional 1-5 likelihood or impact score."""
    if field not in node:
        return
    value = node[field]
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        errors.append(f"{prefix}: '{field}' must be an integer from 1 to 5")


def validate_tree(tree: Any, prefix: str, errors: list[str]) -> None:
    """Validate node/edge integrity, reachability, cycles, and AND/OR semantics."""
    if not isinstance(tree, dict):
        errors.append(f"{prefix}: 'attack_tree' must be an object")
        return

    required = {"root_id", "nodes", "edges"}
    missing = sorted(required - set(tree))
    if missing:
        errors.append(f"{prefix}: attack tree missing fields: {', '.join(missing)}")
        return

    root_id = tree["root_id"]
    nodes = tree["nodes"]
    edges = tree["edges"]
    if not is_nonempty_str(root_id):
        errors.append(f"{prefix}: 'root_id' must be a non-blank string")
    if not isinstance(nodes, list) or not nodes:
        errors.append(f"{prefix}: 'nodes' must be a non-empty list")
        return
    if not isinstance(edges, list):
        errors.append(f"{prefix}: 'edges' must be a list")
        return

    by_id: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(nodes, start=1):
        node_prefix = f"{prefix}: node {index}"
        if not isinstance(node, dict):
            errors.append(f"{node_prefix} must be an object")
            continue
        node_id = node.get("id")
        if not is_nonempty_str(node_id):
            errors.append(f"{node_prefix}: 'id' must be a non-blank string")
        elif node_id in by_id:
            errors.append(f"{prefix}: duplicate node id '{node_id}'")
        else:
            by_id[node_id] = node
        if node.get("type") not in NODE_TYPES:
            errors.append(f"{node_prefix}: 'type' must be one of {sorted(NODE_TYPES)}")
        if not is_nonempty_str(node.get("label")):
            errors.append(f"{node_prefix}: 'label' must be a non-blank string")
        if node.get("operator") not in NODE_OPERATORS:
            errors.append(f"{node_prefix}: 'operator' must be one of {sorted(NODE_OPERATORS)}")
        check_risk_value(node, "likelihood", node_prefix, errors)
        check_risk_value(node, "impact", node_prefix, errors)

    if not isinstance(root_id, str) or root_id not in by_id:
        errors.append(f"{prefix}: 'root_id' must reference an existing node")
        return
    root = by_id[root_id]
    if root.get("type") != "goal":
        errors.append(f"{prefix}: root node must have type 'goal'")
    if root.get("operator") not in {"OR", "AND"}:
        errors.append(f"{prefix}: root node must have operator 'OR' or 'AND'")

    adjacency: dict[str, list[str]] = {node_id: [] for node_id in by_id}
    in_degree: dict[str, int] = {node_id: 0 for node_id in by_id}
    seen_edges: set[tuple[str, str]] = set()

    for index, edge in enumerate(edges, start=1):
        edge_prefix = f"{prefix}: edge {index}"
        if not isinstance(edge, dict):
            errors.append(f"{edge_prefix} must be an object")
            continue
        parent = edge.get("parent")
        child = edge.get("child")
        relation = edge.get("relation")
        if relation not in EDGE_RELATIONS:
            errors.append(f"{edge_prefix}: 'relation' must be 'OR' or 'AND'")
        if not all(is_nonempty_str(value) for value in (parent, child)):
            errors.append(f"{edge_prefix}: 'parent' and 'child' must be node ids")
            continue
        if parent not in by_id or child not in by_id:
            errors.append(f"{edge_prefix}: 'parent' or 'child' references an unknown node")
            continue
        if parent == child:
            errors.append(f"{edge_prefix}: self-loop is not allowed")
            continue
        key = (parent, child)
        if key in seen_edges:
            errors.append(f"{edge_prefix}: duplicate edge {parent} -> {child}")
            continue
        seen_edges.add(key)
        parent_operator = by_id[parent].get("operator")
        if relation != parent_operator:
            errors.append(
                f"{edge_prefix}: relation '{relation}' does not match parent operator "
                f"'{parent_operator}'"
            )
        if parent_operator == "LEAF":
            errors.append(f"{edge_prefix}: a LEAF node cannot have children")
        adjacency[parent].append(child)
        in_degree[child] += 1

    roots = [node_id for node_id, degree in in_degree.items() if degree == 0]
    if len(roots) != 1:
        errors.append(f"{prefix}: expected exactly one root node, found {len(roots)}")
    elif roots[0] != root_id:
        errors.append(f"{prefix}: the only root is '{roots[0]}', not '{root_id}'")

    pending = [node_id for node_id, degree in in_degree.items() if degree == 0]
    visited = 0
    while pending:
        node_id = pending.pop()
        visited += 1
        for child in adjacency[node_id]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                pending.append(child)
    if visited != len(by_id):
        errors.append(f"{prefix}: attack tree contains a cycle or disconnected component")

    reachable = {root_id}
    stack = [root_id]
    while stack:
        node_id = stack.pop()
        for child in adjacency[node_id]:
            if child not in reachable:
                reachable.add(child)
                stack.append(child)
    unreachable = sorted(set(by_id) - reachable)
    if unreachable:
        errors.append(f"{prefix}: {len(unreachable)} node(s) unreachable from root")

    for node_id, node in by_id.items():
        node_prefix = f"{prefix}: node '{node_id}'"
        children = adjacency[node_id]
        expected_type = "attack" if not children else "goal"
        if node.get("type") != expected_type:
            errors.append(f"{node_prefix}: type should be '{expected_type}'")
        expected_operator = "LEAF" if not children else node.get("operator")
        if expected_operator == "LEAF":
            if node.get("operator") != "LEAF":
                errors.append(f"{node_prefix}: leaf node must have operator 'LEAF'")
        elif node.get("operator") not in {"OR", "AND"}:
            errors.append(f"{node_prefix}: internal node must have operator 'OR' or 'AND'")


def validate_record(
    record: Any,
    line_no: int,
    seen_ids: set[str],
    seen_titles: set[str],
) -> tuple[list[str], list[str]]:
    """Validate one dataset record and return errors and warnings."""
    errors: list[str] = []
    warnings: list[str] = []
    prefix = f"line {line_no}"
    if not isinstance(record, dict):
        errors.append(f"{prefix}: record must be a JSON object")
        return errors, warnings

    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        errors.append(f"{prefix}: record missing fields: {', '.join(missing)}")
        return errors, warnings

    for field in STRING_FIELDS:
        if not is_nonempty_str(record[field]):
            errors.append(f"{prefix}: '{field}' must be a non-blank string")
    for field in LIST_FIELDS:
        check_string_list(record[field], field, prefix, errors)

    record_id = record["id"]
    if is_nonempty_str(record_id):
        prefix = f"{prefix} (id={record_id})"
        if record_id in seen_ids:
            errors.append(f"{prefix}: duplicate id '{record_id}'")
        else:
            seen_ids.add(record_id)

    title = record["title"]
    if is_nonempty_str(title):
        normalized = " ".join(title.lower().split())
        if normalized in seen_titles:
            errors.append(f"{prefix}: duplicate title '{title}'")
        else:
            seen_titles.add(normalized)

    if record["split"] not in VALID_SPLITS:
        errors.append(f"{prefix}: 'split' must be one of {sorted(VALID_SPLITS)}")
    if record["review_status"] not in VALID_REVIEW_STATUSES:
        errors.append(f"{prefix}: 'review_status' must be one of {sorted(VALID_REVIEW_STATUSES)}")
    if record["review_status"] == "synthetic_unreviewed":
        warnings.append(f"{prefix}: record is synthetic and unreviewed; exclude from training")

    validate_tree(record["attack_tree"], prefix, errors)

    for index, reference in enumerate(record["evidence_refs"], start=1):
        if ":" not in reference or not reference.split(":", 1)[1].strip():
            errors.append(f"{prefix}: evidence ref {index} must use 'prefix:value' format")

    serialized = json.dumps(record, ensure_ascii=False)
    for name, pattern in SENSITIVE_PATTERNS.items():
        if pattern.search(serialized):
            errors.append(f"{prefix}: contains {name}")

    return errors, warnings


def validate_file(path: Path) -> tuple[int, list[str], list[str]]:
    """Validate a JSONL dataset file and return record count, errors, and warnings."""
    errors: list[str] = []
    warnings: list[str] = []
    count = 0
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()

    if not path.is_file():
        return 0, [f"dataset file not found: {path}"], warnings

    with path.open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            count += 1
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON ({exc.msg})")
                continue
            record_errors, record_warnings = validate_record(
                record, line_no, seen_ids, seen_titles
            )
            errors.extend(record_errors)
            warnings.extend(record_warnings)

    if count == 0:
        errors.append(f"dataset contains no records: {path}")
    return count, errors, warnings


def main(argv: list[str] | None = None) -> int:
    """Run the validator from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"JSONL dataset to validate (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with failure when warnings are present",
    )
    args = parser.parse_args(argv)

    count, errors, warnings = validate_file(args.dataset)
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")

    if errors or (args.strict and warnings):
        error_count = len(errors)
        warning_count = len(warnings)
        print(f"FAIL: {count} record(s) checked; {error_count} errors, {warning_count} warnings")
        return 1
    print(f"PASS: {count} record(s) validated; {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
