"""Semantic cache versions, retaining namespaces for the released implementation."""

import ast
from functools import lru_cache
import hashlib
import inspect
import json
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent


class _SemanticTree(ast.NodeTransformer):
    def visit_Import(self, node):
        return None

    def visit_ImportFrom(self, node):
        return None

    def generic_visit(self, node):
        node = super().generic_visit(node)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
        return node


def _tree(text):
    import textwrap
    return ast.dump(_SemanticTree().visit(ast.parse(textwrap.dedent(text))), include_attributes=False)


@lru_cache(maxsize=None)
def fingerprint(kind):
    from . import resolver, values, wikidata
    from .cache import PersistentLookupNormalizer
    if kind == "resolution":
        text = "".join(_tree((DIRECTORY / name).read_text()) for name in ("resolver.py", "values.py", "schemas.py", "wikidata.py"))
    elif kind == "lookup":
        methods = ("_historical_description", "_historical_candidate", "_level_from_description", "_normalize_query", "_ancestor_qids",
                   "_has_administrative_region_role", "_search", "_lexical_score", "_context_search", "_search_place_candidates")
        text = "".join(_tree(inspect.getsource(getattr(resolver.AuditableWikidataNormalizer, name))) for name in methods)
        text += "".join(_tree(inspect.getsource(function)) for function in (resolver.canonical_component, resolver.contextual_component, resolver.explicit_component_level))
        for module, names in ((wikidata, ("WIKIDATA_API_URL", "WIKIDATA_SPARQL_URL", "WIKIDATA_LANGUAGE", "WIKIDATA_SEARCH_LIMIT", "_LEVEL_PRIORITY", "_DESCRIPTION_KEYWORDS", "_COMPONENT_SUFFIXES", "_SUFFIX_LEVELS")),
                              (values, ("LOCATION_PRECISION", "LOCATION_FIELDS", "LOCATION_CONTEXT_FIELDS")),
                              (resolver, ("_COMPONENT_ALIASES", "_LEXICAL_ALIASES", "_CONTEXT_RELATIONS", "_TYPE_PREFIX_LEVELS", "PLACE_SEARCH_LIMIT", "COMPONENT_SEARCH_LIMIT"))):
            for name in names:
                value = getattr(module, name)
                text += json.dumps(value, sort_keys=True, default=lambda value: sorted(value) if isinstance(value, set) else str(value))
        for pattern in (wikidata._HISTORICAL_POLITY, resolver._CONTEXT_QUALIFIER, resolver._DISTANCE_CONTEXT, resolver._DIRECTIONAL_AREA, resolver._TYPE_PREFIX):
            text += repr((pattern.pattern, pattern.flags))
    elif kind == "profile":
        functions = (PersistentLookupNormalizer._fetch_entity_profile, resolver.AuditableWikidataNormalizer._metadata_from_bindings,
                     resolver.AuditableWikidataNormalizer._historical_description, resolver.AuditableWikidataNormalizer._level_from_description)
        text = "".join(_tree(inspect.getsource(function)) for function in functions)
    else:
        raise ValueError("Unknown semantic cache layer")
    return hashlib.sha256(text.encode()).hexdigest()


@lru_cache(maxsize=None)
def _identity(kind):
    path = DIRECTORY / "cache_versions.json"
    records = json.loads(path.read_text()) if path.exists() else {}
    record = records.get(kind, {})
    current = fingerprint(kind)
    if record.get("fingerprint") == current:
        return record["id"]
    return f"location-{kind}-v1-{current[:16]}"


def lookup_cache_identity():
    return _identity("lookup")


def profile_cache_identity():
    return _identity("profile")


def resolution_cache_identity(*, refresh_evidence=False):
    identity = _identity("resolution")
    return identity + "-evidence-" + profile_cache_identity().rsplit("-", 1)[1] if refresh_evidence else identity
