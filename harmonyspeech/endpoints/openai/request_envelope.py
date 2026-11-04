"""JSON / multipart request envelope for the audio-input endpoints.

Phase 1-3 chose a **content-type dispatch on the same path** rather than a
sibling ``/multipart`` route: the OpenAI clients that upload audio already send
``multipart/form-data`` (``curl -F file=@speech.mp3``) to the same paths, and a
second route registered on the same path+method would be shadowed by the first
one. Each audio-input handler therefore takes ``Request`` and calls
:func:`parse_dual_request`, which:

* parses ``multipart/form-data`` (or urlencoded) uploads, mapping uploaded file
  parts onto the same base64 fields the JSON path uses, so the handler and the
  engine never see a different request shape;
* parses a JSON body exactly as before (the legacy/documented alternative);
* raises :class:`fastapi.exceptions.RequestValidationError` for malformed
  bodies, preserving the existing 400 error response;
* enforces the configurable upload size limit and raises
  :class:`UploadTooLargeError` (mapped to HTTP 413 by ``api_server``).

The in-process base64 round-trip is deliberate: it keeps payload semantics
identical to the JSON transport. Phase 2 (artifacts) removes it if it matters.
"""

from __future__ import annotations

import base64
import json
import os
import types
from collections.abc import Mapping, Sequence
from typing import Any, Union, get_args, get_origin

from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
from starlette.requests import Request

__all__ = [
    "DEFAULT_MAX_UPLOAD_SIZE_MB",
    "MAX_UPLOAD_SIZE_ENV",
    "UploadTooLargeError",
    "dual_body_openapi",
    "parse_dual_request",
    "resolve_max_upload_bytes",
]

DEFAULT_MAX_UPLOAD_SIZE_MB = 50
MAX_UPLOAD_SIZE_ENV = "HARMONYSPEECH_MAX_UPLOAD_MB"

#: Nested request fields that arrive as JSON strings in a multipart form.
JSON_FORM_FIELDS = frozenset(
    {"generation_options", "output_options", "pre_processing_filters", "post_generation_filters"}
)

_MULTIPART_CONTENT_TYPES = frozenset({"multipart/form-data", "application/x-www-form-urlencoded"})


class UploadTooLargeError(ValueError):
    """A multipart upload exceeded the configured maximum request size."""


def resolve_max_upload_bytes(app: Any = None) -> int:
    """Resolve the configured upload limit in bytes.

    Priority: ``app.state.max_upload_bytes`` (set by ``build_app`` from the CLI
    argument) -> ``HARMONYSPEECH_MAX_UPLOAD_MB`` -> 50 MiB.
    """
    configured = getattr(getattr(app, "state", None), "max_upload_bytes", None)
    if configured:
        return int(configured)
    raw = os.environ.get(MAX_UPLOAD_SIZE_ENV)
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return int(value * 1024 * 1024)
        except ValueError:
            pass
    return DEFAULT_MAX_UPLOAD_SIZE_MB * 1024 * 1024


def _as_validation_error(exc: Exception, loc: tuple[str, ...] = ("body",)) -> RequestValidationError:
    if isinstance(exc, ValidationError):
        return RequestValidationError(exc.errors())
    return RequestValidationError([{"type": "value_error", "loc": loc, "msg": str(exc)}])


async def _read_uploads(raw_request: Request, upload_fields: Mapping[str, Sequence[str]], limit: int) -> dict[str, Any]:
    try:
        form = await raw_request.form()
    except Exception as exc:  # malformed multipart body
        raise _as_validation_error(exc) from exc

    data: dict[str, Any] = {}
    for key, value in form.multi_items():
        if hasattr(value, "read"):  # UploadFile part: handled via upload_fields
            continue
        if key in JSON_FORM_FIELDS:
            try:
                data[key] = json.loads(value)
            except (TypeError, json.JSONDecodeError) as exc:
                raise _as_validation_error(exc, loc=("body", key)) from exc
        elif key not in data:
            data[key] = value

    consumed: set[str] = set()
    for model_field, candidates in upload_fields.items():
        upload = None
        for candidate in candidates:
            value = form.get(candidate)
            if value is not None and hasattr(value, "read") and candidate not in consumed:
                upload = value
                consumed.add(candidate)
                break
        if upload is None:
            # Allow pre-encoded base64 to be supplied as a form text field
            # (for example target_embedding on /v1/voice/convert).
            continue
        payload = await upload.read()
        if len(payload) > limit:
            raise UploadTooLargeError(
                f"Uploaded file '{upload.filename or model_field}' exceeds the maximum upload size "
                f"of {limit // (1024 * 1024)} MiB."
            )
        data[model_field] = base64.b64encode(payload).decode("ascii")

    return data


async def parse_dual_request(
    raw_request: Request,
    model_cls: type[BaseModel],
    *,
    upload_fields: Mapping[str, Sequence[str]] | None = None,
) -> BaseModel:
    """Parse a JSON or multipart body into ``model_cls``.

    :param upload_fields: mapping of request-model field name to the accepted
        upload part names, in priority order (for example
        ``{"input_audio": ("input_audio", "file", "audio")}``).
    """
    content_type = (raw_request.headers.get("content-type") or "").split(";")[0].strip().lower()
    limit = resolve_max_upload_bytes(raw_request.app)

    try:
        if content_type in _MULTIPART_CONTENT_TYPES:
            data = await _read_uploads(raw_request, upload_fields or {}, limit)
            return model_cls.model_validate(data)

        raw = await raw_request.body()
        if not raw:
            data = {}
        else:
            data = json.loads(raw)
        return model_cls.model_validate(data)
    except UploadTooLargeError:
        raise
    except RequestValidationError:
        raise
    except Exception as exc:
        raise _as_validation_error(exc) from exc


# ---------------------------------------------------------------------------
# OpenAPI support for the manually dispatched body
# ---------------------------------------------------------------------------


def _unwrap_annotation(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        return args[0] if args else str
    if origin in (list, tuple, Sequence):
        return list
    if origin is dict:
        return dict
    return annotation


def _json_type(annotation: Any) -> str:
    annotation = _unwrap_annotation(annotation)
    if annotation is bool:
        return "boolean"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is str:
        return "string"
    if annotation is list:
        return "array"
    if annotation is dict:
        return "object"
    return "object"


def dual_body_openapi(model_cls: type[BaseModel], upload_fields: Mapping[str, str]) -> dict:
    """Build ``openapi_extra`` advertising both supported request bodies."""
    properties: dict[str, Any] = {}
    for name, field in model_cls.model_fields.items():
        schema: dict[str, Any] = {"title": name.replace("_", " ").title()}
        if isinstance(field.annotation, type) and issubclass(field.annotation, BaseModel):
            schema.update({"type": "object"})
        elif get_origin(field.annotation) in (Union, types.UnionType) and any(
            isinstance(arg, type) and issubclass(arg, BaseModel) for arg in get_args(field.annotation)
        ):
            schema.update({"type": "object"})
        else:
            schema.update({"type": _json_type(field.annotation)})
        if field.description:
            schema["description"] = field.description
        properties[name] = schema

    multipart_properties = dict(properties)
    for field_name, description in upload_fields.items():
        multipart_properties[field_name] = {
            "type": "string",
            "format": "binary",
            "description": description,
        }

    return {
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "title": model_cls.__name__,
                        "properties": properties,
                    }
                },
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "title": f"{model_cls.__name__}Multipart",
                        "properties": multipart_properties,
                    }
                },
            },
        }
    }
