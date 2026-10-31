import json
import re
import ast
from typing import Any, Tuple, Optional, Dict
from pathlib import Path
import time

def extract_json_from_markdown(text):
    """
    Extract a JSON object from:
    - An already parsed dict or list
    - A valid JSON string directly
    - A Python literal string (single quotes, etc.)
    - A Markdown block like ```json ... ```
    """
    if isinstance(text, (dict, list)):
        return text

    if isinstance(text, str):
        # Strip thinking tags
        text = re.sub(r"<(?:think|thought)>.*?</(?:think|thought)>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Cleanup trailing commas before closing braces/brackets
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        # 1. Try to parse directly as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Try to search for ```json ... ``` block
        pattern = re.compile(
            r"```json\s*(?P<json>[\s\S]*?)\s*```",
            re.IGNORECASE
        )
        match = pattern.search(text)
        if match:
            json_str = match.group('json')
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
                
        # 3. Try to search for first { ... } block
        match_brace = re.search(r"(\{[\s\S]*\})", text)
        if match_brace:
            json_str = match_brace.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # 4. Try ast.literal_eval as fallback (for Python-like syntax)
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            pass

        raise ValueError("No se encontró bloque JSON válido en la respuesta:\n" + str(text))

    raise TypeError("El argumento debe ser un str, dict o list.")

def try_json_repair(text: str) -> Tuple[Optional[Any], bool, Optional[str]]:
    """
    Attempt to fix invalid JSON using json_repair if available.
    Returns: (parsed_obj | None, used_repair: bool, error_str | None)
    """
    try:
        from json_repair import repair_json
    except ImportError:
        return None, False, "json_repair not installed"

    try:
        repaired = repair_json(text)
        return json.loads(repaired), True, None
    except Exception as e:
        return None, True, f"json_repair failed: {e}"

REFUSAL_PATTERNS = [
    r"\bI cannot\b", r"\bI can’t\b", r"\bI won't\b", r"\bI will not\b",
    r"\bI'm unable to\b", r"\bI am unable to\b", r"\bAs an AI\b",
    r"\bgo against (?:policy|policies|guidelines)\b",
    r"\bviolat(?:e|es|ing) (?:policy|policies|guidelines)\b",
    r"\bI must refuse\b", r"\bI cannot create a response\b"
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), flags=re.IGNORECASE)

def is_refusal(raw_text: str) -> bool:
    return bool(REFUSAL_RE.search(raw_text or ""))

def parse_with_repair(raw_text: Any) -> Tuple[Optional[dict], Dict[str, Any]]:
    """
    Try extractor first; if it fails, try json_repair.
    Returns: (parsed_dict or None, diagnostics dict)
    """
    if isinstance(raw_text, dict):
        return raw_text, {"used_preparsed_dict": True, "json_valid": True}

    if isinstance(raw_text, str):
        raw_text = re.sub(r"<(?:think|thought)>.*?</(?:think|thought)>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
        raw_text = raw_text.strip()

    diag = {
        "used_extract_json_from_markdown": False,
        "used_json_repair": False,
        "json_valid": False,
        "error_message": None,
        "is_refusal": is_refusal(raw_text) if isinstance(raw_text, str) else False,
    }
    
    try:
        obj = extract_json_from_markdown(raw_text)
        if isinstance(obj, (dict, list)):
            if isinstance(obj, list):
                obj = obj[0] if obj and isinstance(obj[0], dict) else None
        if isinstance(obj, dict):
            diag["used_extract_json_from_markdown"] = True
            diag["json_valid"] = True
            return obj, diag
    except Exception as e1:
        diag["error_message"] = f"extract_json_from_markdown failed: {e1}"

    obj2, repaired, err2 = try_json_repair(raw_text)
    diag["used_json_repair"] = repaired
    if isinstance(obj2, dict):
        diag["json_valid"] = True
        diag["error_message"] = None
        return obj2, diag

    if err2:
        diag["error_message"] = (diag["error_message"] or "") + f" | {err2}"
    return None, diag
