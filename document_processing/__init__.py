"""Shared, source-authorized document processing runtime for Hermes."""

from .engine import (
    DocumentExtractError,
    SourceArtifact,
    check_requirements,
    handle_document_extract,
)
from .schema import DOCUMENT_EXTRACT_SCHEMA

__all__ = [
    "DOCUMENT_EXTRACT_SCHEMA",
    "DocumentExtractError",
    "SourceArtifact",
    "check_requirements",
    "handle_document_extract",
]
