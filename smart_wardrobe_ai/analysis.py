"""Clothing analysis: request validation, prompt, output schema and
normalization (docs/08 §5, ADR-031). No network here, so it is testable.

The app sends its taxonomy with each request; the output schema turns every
list into an enum, so the model can only answer with identifiers the app
knows. Whatever comes back is still checked again, here and in the app.
"""

import json
import re

PROMPT_VERSION = "clothing-2026.10-v1"
SCHEMA_VERSION = "1"
MAX_TAXONOMY_SIZE = 200
LOCALES = {"fr": "French", "en": "English"}

_ID = re.compile(r"^[a-z0-9_]{1,64}$")
_TAXONOMIES = ("categories", "colors", "styles", "materials", "patterns",
               "fits", "seasons")


class RequestError(Exception):
    """A request the proxy refuses; [code] follows docs/10 §2."""

    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code = code
        self.status = status


class AnalysisRequest:
    def __init__(self, locale, taxonomy, hints):
        self.locale = locale
        self.taxonomy = taxonomy
        self.hints = hints


def parse_request(payload):
    """Validates the JSON part sent by the app."""
    if not isinstance(payload, dict) or payload.get("schemaVersion") != SCHEMA_VERSION:
        raise RequestError("validation", "Unsupported request schema.")
    locale = payload.get("locale")
    if locale not in LOCALES:
        raise RequestError("validation", "Unsupported locale.")
    raw = payload.get("taxonomy")
    if not isinstance(raw, dict):
        raise RequestError("validation", "Missing taxonomy.")
    taxonomy = {}
    for key in _TAXONOMIES:
        values = raw.get(key)
        if (not isinstance(values, list) or not values
                or len(values) > MAX_TAXONOMY_SIZE
                or not all(isinstance(v, str) and _ID.match(v) for v in values)):
            raise RequestError("validation", f"Invalid taxonomy: {key}.")
        taxonomy[key] = sorted(set(values))
    hints = payload.get("hints") or {}
    category = hints.get("categoryId") if isinstance(hints, dict) else None
    clean_hints = {}
    if category in taxonomy["categories"]:
        clean_hints["categoryId"] = category
    return AnalysisRequest(locale, taxonomy, clean_hints)


# Stable across requests, so that it can be cached.
SYSTEM_PROMPT = """You describe one piece of clothing, footwear or accessory \
from a photo, for a personal wardrobe app. The user reviews every value \
before it is used.

Rules:
- Describe only the main piece in the photo. Ignore the background, the \
person wearing it and other objects.
- Choose values only from the identifiers given. Use "unknown" when the \
photo does not show it; never guess a value you cannot see.
- category: the most specific identifier that fits.
- Colours: the primary colour covers most of the piece; secondary colours \
are clearly visible other colours, at most three, never the primary one.
- Materials: a photo rarely proves a fibre. Give a material only when its \
look is distinctive (denim, leather, knitted wool), with a lower confidence.
- formality: 0 = sport or lounge wear, 50 = smart casual, 100 = black tie.
- seasons: when the piece is comfortable to wear; "all_year" alone for \
versatile pieces.
- name: a short name for the piece, three to five words, in the requested \
language, without brand.
- description: one or two plain sentences in the requested language.
- confidence: from 0 (a guess) to 1 (certain), for each field separately.
- warnings: short notes in the requested language when the photo limits the \
analysis (blurred, dark, several pieces, not clothing). Empty otherwise.
- is_clothing: false when the photo shows no clothing, footwear or \
accessory."""


def build_user_text(request):
    lines = [
        f"Language for name, description and warnings: {LOCALES[request.locale]}.",
        "Allowed identifiers:",
    ]
    for key in _TAXONOMIES:
        lines.append(f"- {key}: {', '.join(request.taxonomy[key])}")
    if "categoryId" in request.hints:
        lines.append(
            f"The user already chose the category {request.hints['categoryId']}; "
            "keep it unless the photo clearly shows something else.")
    return "\n".join(lines)


def _confidence():
    return {"type": "number"}


def _single(values):
    return {
        "type": "object",
        "properties": {
            "value": {"type": "string", "enum": [*values, "unknown"]},
            "confidence": _confidence(),
        },
        "required": ["value", "confidence"],
        "additionalProperties": False,
    }


def _multiple(values):
    return {
        "type": "object",
        "properties": {
            "values": {"type": "array", "items": {"type": "string", "enum": values}},
            "confidence": _confidence(),
        },
        "required": ["values", "confidence"],
        "additionalProperties": False,
    }


def build_schema(request):
    t = request.taxonomy
    properties = {
        "is_clothing": {"type": "boolean"},
        "name": {
            "type": "object",
            "properties": {"value": {"type": "string"}, "confidence": _confidence()},
            "required": ["value", "confidence"],
            "additionalProperties": False,
        },
        "category": _single(t["categories"]),
        "primary_color": _single(t["colors"]),
        "secondary_colors": _multiple(t["colors"]),
        "styles": _multiple(t["styles"]),
        "materials": _multiple(t["materials"]),
        "seasons": _multiple(t["seasons"]),
        "pattern": _single(t["patterns"]),
        "fit": _single(t["fits"]),
        "formality": {
            "type": "object",
            "properties": {"value": {"type": "integer"}, "confidence": _confidence()},
            "required": ["value", "confidence"],
            "additionalProperties": False,
        },
        "description": {"type": "string"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _clamp(value, low, high):
    return max(low, min(high, value))


def _confidence_of(field):
    value = field.get("confidence") if isinstance(field, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(_clamp(float(value), 0.0, 1.0), 2)


def _text(value, limit):
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    return value[:limit] if value else None


def normalize(raw, request):
    """The fields the app may show: known identifiers only, values in range,
    unknown fields left out. Never trusts the structured output alone."""
    if not isinstance(raw, dict):
        raise ValueError("Analysis is not an object.")
    t = request.taxonomy
    fields = {}
    warnings = [w for w in (_text(w, 200) for w in raw.get("warnings") or []) if w][:3]
    description = _text(raw.get("description"), 400)
    if raw.get("is_clothing") is False:
        return {"fields": {}, "description": description, "warnings": warnings,
                "isClothing": False}

    def single(key, out, allowed):
        field = raw.get(key)
        confidence = _confidence_of(field)
        value = field.get("value") if isinstance(field, dict) else None
        if value in allowed and confidence is not None:
            fields[out] = {"value": value, "confidence": confidence}

    def multiple(key, out, allowed, limit, exclude=None):
        field = raw.get(key)
        confidence = _confidence_of(field)
        values = field.get("values") if isinstance(field, dict) else None
        if confidence is None or not isinstance(values, list):
            return
        kept = []
        for value in values:
            if value in allowed and value != exclude and value not in kept:
                kept.append(value)
        if kept:
            fields[out] = {"value": kept[:limit], "confidence": confidence}

    name = raw.get("name")
    name_value = _text(name.get("value") if isinstance(name, dict) else None, 60)
    if name_value and _confidence_of(name) is not None:
        fields["name"] = {"value": name_value, "confidence": _confidence_of(name)}
    single("category", "category", t["categories"])
    single("primary_color", "primaryColor", t["colors"])
    primary = fields.get("primaryColor", {}).get("value")
    multiple("secondary_colors", "secondaryColors", t["colors"], 3, exclude=primary)
    multiple("styles", "styles", t["styles"], 3)
    multiple("materials", "materials", t["materials"], 3)
    multiple("seasons", "seasons", t["seasons"], 5)
    single("pattern", "pattern", t["patterns"])
    single("fit", "fit", t["fits"])
    formality = raw.get("formality")
    value = formality.get("value") if isinstance(formality, dict) else None
    if isinstance(value, int) and not isinstance(value, bool) \
            and _confidence_of(formality) is not None:
        fields["formality"] = {"value": _clamp(value, 0, 100),
                               "confidence": _confidence_of(formality)}
    return {"fields": fields, "description": description, "warnings": warnings,
            "isClothing": True}


def parse_model_text(text):
    """The JSON object written by the model under the output schema."""
    return json.loads(text)
