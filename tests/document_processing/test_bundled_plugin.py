from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "document-extract"


def _load_plugin():
    package = "test_fleet_document_extract_plugin"
    spec = importlib.util.spec_from_file_location(
        package,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    try:
        spec.loader.exec_module(module)
        return package, module
    except Exception:
        sys.modules.pop(package, None)
        raise


def test_bundled_plugin_manifest_is_a_single_bounded_edge_tool():
    manifest = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8"))

    assert manifest["name"] == "document-extract"
    assert manifest["kind"] == "standalone"
    assert manifest["provides_tools"] == ["document_extract"]
    assert not (PLUGIN_DIR / "engine.py").exists()
    assert not (PLUGIN_DIR / "paddle_runner.py").exists()


def test_plugin_registers_the_shared_engine_handler_and_schema():
    from document_processing import (
        DOCUMENT_EXTRACT_SCHEMA,
        check_requirements,
        handle_document_extract,
    )

    package, plugin = _load_plugin()
    calls: list[dict] = []

    class Context:
        def register_tool(self, **kwargs):
            calls.append(kwargs)

    try:
        plugin.register(Context())
    finally:
        sys.modules.pop(package, None)

    assert len(calls) == 1
    call = calls[0]
    assert call["name"] == "document_extract"
    assert call["toolset"] == "document_extract"
    assert call["schema"] is DOCUMENT_EXTRACT_SCHEMA
    assert call["handler"] is handle_document_extract
    assert call["check_fn"] is check_requirements
    assert call["emoji"] == "📄"
