"""Document schema extraction and field resolution.

Compiles high-performance extractors for document text (field, path, concat,
template, messages, auto) and resolves document IDs and domains.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from textscour.config import TextSpec
from textscour.errors import ScourError

AUTO_TEXT_CANDIDATES = (
    "text",
    "content",
    "body",
    "article",
    "document",
    "raw_text",
    "prompt",
    "response",
    "output",
)

ID_EXCLUDED_KEYS = {
    "id",
    "_id",
    "doc_id",
    "uid",
    "uuid",
    "key",
    "identifier",
    "domain",
    "source",
    "site",
    "url",
    "category",
    "timestamp",
    "meta",
    "metadata",
    "score",
    "label",
    "target",
    "split",
    "version",
    "hash",
    "kind",
    "type",
}


def _resolve_path(data: Any, path: str) -> Any:
    """Resolve a dotted path or array indexing like 'meta.data.0.text'."""
    tokens = path.split(".")
    curr = data
    for tok in tokens:
        if curr is None:
            return None
        if isinstance(curr, dict):
            curr = curr.get(tok)
        elif isinstance(curr, (list, tuple)):
            try:
                idx = int(tok)
                if 0 <= idx < len(curr):
                    curr = curr[idx]
                else:
                    return None
            except ValueError:
                return None
        else:
            return None
    return curr


def _extract_messages(messages: Any, role_filter: list[str] | None, sep: str) -> str | None:
    """Extract and format chat / conversation messages into single document text."""
    if not isinstance(messages, (list, tuple)):
        return None
    parts: list[str] = []
    allowed_roles = set(role_filter) if role_filter else None
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role", "") or item.get("from", "")
        if allowed_roles is not None and role not in allowed_roles:
            continue
        content = item.get("content")
        if content is None:
            content = item.get("value")
        if content is None:
            continue
        if not isinstance(content, str):
            content = str(content)
        if role:
            parts.append(f"{role}: {content}")
        else:
            parts.append(content)
    return sep.join(parts) if parts else None


def compile_extractor(spec: TextSpec | dict[str, Any]) -> Callable[[dict[str, Any] | str], str | None]:
    """Compile an extractor function according to TextSpec mode."""
    if isinstance(spec, dict):
        spec = TextSpec(**spec)
    mode = spec.mode

    if mode == "field":
        field_name = spec.field or "text"

        def _extract_field(doc: dict[str, Any] | str) -> Any:
            if isinstance(doc, str):
                s = doc.strip()
                return s if s else None
            if isinstance(doc, dict):
                if field_name.lower() in ID_EXCLUDED_KEYS:
                    return None
                val = doc.get(field_name)
                if val is None:
                    return None
                if not isinstance(val, str):
                    return val
                s = val.strip()
                doc_id = extract_id(doc, None, "")
                if s and s != doc_id:
                    return s
                return None
            return None

        return _extract_field

    if mode == "path":
        path_str = spec.field or "text"

        def _extract_path(doc: dict[str, Any] | str) -> Any:
            if isinstance(doc, str):
                s = doc.strip()
                return s if s else None
            if isinstance(doc, dict):
                last_tok = path_str.split(".")[-1].lower()
                if last_tok in ID_EXCLUDED_KEYS:
                    return None
                val = _resolve_path(doc, path_str)
                if val is None:
                    return None
                if not isinstance(val, (str, list, tuple)):
                    return val
                doc_id = extract_id(doc, None, "")
                if isinstance(val, str):
                    s = val.strip()
                    if s and s != doc_id:
                        return s
                    return None
                if isinstance(val, (list, tuple)):
                    if last_tok in ("messages", "conversations"):
                        return _extract_messages(val, spec.role_filter, spec.sep)
                    parts = [str(x).strip() for x in val if x is not None and str(x).strip()]
                    return "\n".join(parts) if parts else None
            return None

        return _extract_path

    if mode == "concat":
        field_list = spec.fields
        sep = spec.sep

        def _extract_concat(doc: dict[str, Any] | str) -> str | None:
            if isinstance(doc, str):
                s = doc.strip()
                return s if s else None
            if isinstance(doc, dict):
                doc_id = extract_id(doc, None, "")
                parts = []
                for f in field_list:
                    if f.lower() in ID_EXCLUDED_KEYS:
                        continue
                    val = _resolve_path(doc, f)
                    if val is not None and str(val).strip() and str(val).strip() != doc_id:
                        parts.append(str(val).strip())
                return sep.join(parts) if parts else None
            return None

        return _extract_concat

    if mode == "template":
        template_str = spec.template or "{text}"

        def _extract_template(doc: dict[str, Any] | str) -> str | None:
            if isinstance(doc, str):
                s = doc.strip()
                return s if s else None
            if isinstance(doc, dict):
                try:
                    res = template_str.format(**doc)
                except KeyError:
                    # Partial match with regex fallback
                    def _replace_var(match: re.Match) -> str:
                        key = match.group(1)
                        return str(doc.get(key, ""))

                    res = re.sub(r"\{([a-zA-Z0-9_]+)\}", _replace_var, template_str)
                s = res.strip()
                return s if s else None
            return None

        return _extract_template

    if mode == "messages":
        field_name = spec.field or "messages"
        role_filter = spec.role_filter
        sep = spec.sep

        def _extract_msgs(doc: dict[str, Any] | str) -> str | None:
            if isinstance(doc, str):
                s = doc.strip()
                return s if s else None
            if isinstance(doc, dict):
                raw_msgs = doc.get(field_name)
                if raw_msgs is None and "conversations" in doc:
                    raw_msgs = doc.get("conversations")
                return _extract_messages(raw_msgs, role_filter, sep)
            return None

        return _extract_msgs

    # Auto mode
    def _extract_auto(doc: dict[str, Any] | str) -> str | None:
        if isinstance(doc, str):
            s = doc.strip()
            return s if s else None
        if not isinstance(doc, dict):
            return None

        doc_id = extract_id(doc, None, "")

        # 1. Direct candidate fields at top level
        for cand in AUTO_TEXT_CANDIDATES:
            if cand in doc and doc[cand] is not None and cand.lower() not in ID_EXCLUDED_KEYS:
                val = doc[cand]
                if isinstance(val, str) and val.strip() and val.strip() != doc_id:
                    return val.strip()
                if isinstance(val, (list, tuple)) and cand in ("messages", "conversations"):
                    res = _extract_messages(val, None, "\n\n")
                    if res and res.strip() and res.strip() != doc_id:
                        return res.strip()
                if isinstance(val, dict):
                    for subcand in AUTO_TEXT_CANDIDATES:
                        if subcand in val and val[subcand] is not None and subcand.lower() not in ID_EXCLUDED_KEYS:
                            subval = val[subcand]
                            if isinstance(subval, str) and subval.strip() and subval.strip() != doc_id:
                                return subval.strip()

        # 2. Check if messages / conversations key exists at top level
        for cand in ("messages", "conversations"):
            if cand in doc and isinstance(doc[cand], (list, tuple)):
                res = _extract_messages(doc[cand], None, "\n\n")
                if res and res.strip() and res.strip() != doc_id:
                    return res.strip()

        # 3. Check nested dictionaries for candidate fields (e.g. doc["data"]["article"]["body"])
        for k, v in doc.items():
            if k.lower() in ID_EXCLUDED_KEYS:
                continue
            if isinstance(v, dict):
                for cand in AUTO_TEXT_CANDIDATES:
                    if cand in v and v[cand] is not None and cand.lower() not in ID_EXCLUDED_KEYS:
                        subval = v[cand]
                        if isinstance(subval, str) and subval.strip() and subval.strip() != doc_id:
                            return subval.strip()
                        if isinstance(subval, dict):
                            for subcand in AUTO_TEXT_CANDIDATES:
                                if subcand in subval and subval[subcand] is not None and subcand.lower() not in ID_EXCLUDED_KEYS:
                                    leaf = subval[subcand]
                                    if isinstance(leaf, str) and leaf.strip() and leaf.strip() != doc_id:
                                        return leaf.strip()
                for cand in ("messages", "conversations"):
                    if cand in v and isinstance(v[cand], (list, tuple)):
                        res = _extract_messages(v[cand], None, "\n\n")
                        if res and res.strip() and res.strip() != doc_id:
                            return res.strip()

        # 4. Fallback: Search non-metadata string values (top-level or nested)
        candidate_strings: list[str] = []
        for k, v in doc.items():
            if k.lower() in ID_EXCLUDED_KEYS:
                continue
            if isinstance(v, str) and v.strip() and v.strip() != doc_id:
                candidate_strings.append(v.strip())
            elif isinstance(v, dict):
                for subk, subv in v.items():
                    if subk.lower() in ID_EXCLUDED_KEYS:
                        continue
                    if isinstance(subv, str) and subv.strip() and subv.strip() != doc_id:
                        candidate_strings.append(subv.strip())

        if candidate_strings:
            best = max(candidate_strings, key=len)
            if best != doc_id and len(best) > 0:
                return best

        return None

    return _extract_auto


def extract_id(doc: dict[str, Any] | str, id_field: str | None, default_id: str) -> str:
    """Extract document identifier or fallback to default_id."""
    if isinstance(doc, dict) and id_field:
        val = doc.get(id_field)
        if val is not None:
            return str(val)
    if isinstance(doc, dict):
        for candidate in ("id", "_id", "doc_id", "uid", "uuid", "key"):
            if candidate in doc and doc[candidate] is not None:
                return str(doc[candidate])
    return default_id


def extract_domain(
    doc: dict[str, Any] | str,
    domain_field: str | None,
    default_domain: str | None = None,
) -> str:
    """Extract document domain or category string."""
    if isinstance(doc, dict) and domain_field:
        val = doc.get(domain_field)
        if val is not None:
            return str(val)
    if isinstance(doc, dict):
        for candidate in ("domain", "category", "source", "site", "url"):
            if candidate in doc and doc[candidate] is not None:
                val = str(doc[candidate])
                if candidate == "url" and "://" in val:
                    # Extract host
                    from urllib.parse import urlparse

                    try:
                        return urlparse(val).netloc or val
                    except Exception:
                        return val
                return val
    return default_domain or ""
