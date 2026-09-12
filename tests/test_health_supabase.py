"""Hardening Fix 1: /health debe detectar Supabase con las variables reales.

El proyecto usa SUPABASE_ANON_KEY (preferente) y SUPABASE_SERVICE_ROLE_KEY;
SUPABASE_KEY se mantiene solo como fallback legacy. Estos tests demuestran
que la deteccion ya no depende de SUPABASE_KEY y que /health conserva su
contrato de respuesta.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import backend.app.api.routes.health as health_mod

_ALL_KEYS = ("SUPABASE_ANON_KEY", "SUPABASE_KEY", "SUPABASE_SERVICE_ROLE_KEY")


def _clear_keys(monkeypatch) -> None:
    for name in _ALL_KEYS:
        monkeypatch.delenv(name, raising=False)


def test_health_key_prefers_anon(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    monkeypatch.setenv("SUPABASE_KEY", "legacy-key")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    assert health_mod._supabase_health_key() == "anon-key"


def test_health_key_accepts_legacy_supabase_key(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("SUPABASE_KEY", "legacy-key")
    assert health_mod._supabase_health_key() == "legacy-key"


def test_health_key_accepts_service_role(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    assert health_mod._supabase_health_key() == "service-key"


def test_health_key_none_when_unconfigured(monkeypatch):
    _clear_keys(monkeypatch)
    assert health_mod._supabase_health_key() == ""


def test_check_supabase_not_configured(monkeypatch):
    _clear_keys(monkeypatch)
    assert health_mod._check_supabase() == "not_configured"


def test_check_supabase_connected_with_anon_key(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    fake_client = MagicMock()
    with patch("supabase.create_client", return_value=fake_client):
        assert health_mod._check_supabase() == "connected"
    fake_client.table.assert_called_once_with("scan_results")


def test_check_supabase_disconnected_on_error(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    with patch("supabase.create_client", side_effect=Exception("boom")):
        assert health_mod._check_supabase() == "disconnected"


def test_health_components_contract_with_configured_supabase(monkeypatch):
    """Contrato de _check_components y pipeline_mode cuando Supabase está configurado."""
    _clear_keys(monkeypatch)
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    with patch.object(health_mod, "_check_onnx", return_value="loaded"), \
         patch.object(health_mod, "_check_yara", return_value="available"), \
         patch.object(health_mod, "_check_supabase", return_value="connected"), \
         patch.object(health_mod, "_check_n8n", return_value="disabled"), \
         patch.object(health_mod, "_check_psutil", return_value="available"), \
         patch.object(health_mod, "_check_offline_queue", return_value=0):
        components = health_mod._check_components()

    assert set(components) == {
        "onnx_model", "yara_scanner", "supabase", "n8n", "psutil", "offline_queue_size",
    }
    assert components["supabase"] == "connected"
    assert health_mod._compute_pipeline_mode(components) == "full"


def test_health_endpoint_keeps_contract(test_client):
    """GET /health mantiene status/pipeline_mode/components/max_upload_mb."""
    components = {
        "onnx_model": "loaded",
        "yara_scanner": "available",
        "supabase": "connected",
        "n8n": "disabled",
        "psutil": "available",
        "offline_queue_size": 0,
    }
    with patch.object(health_mod, "_check_components", return_value=components):
        resp = test_client.get("/health")

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["status"] == "ok"
    assert data["pipeline_mode"] == "full"
    assert data["components"]["supabase"] == "connected"
    assert "max_upload_mb" in data
