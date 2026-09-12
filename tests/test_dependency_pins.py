"""Hardening Fix 2 y Fix 4: pin de scikit-learn y eliminacion de MODEL_PATH muerto.

Fix 2: los scalers StandardScaler de produccion fueron serializados con
scikit-learn 1.6.1; el proyecto debe quedar fijado a esa version para evitar
el InconsistentVersionWarning al deserializar.

Fix 4: la variable de entorno MODEL_PATH=./models/model.onnx no es leida por
ninguna ruta de ejecucion; se elimino del .env. Estos tests verifican por
busqueda estatica que no queden referencias funcionales.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import joblib
import pytest
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

REPO = Path(__file__).resolve().parent.parent
VALIDATED_SKLEARN = "1.6.1"


# ---------------------------------------------------------------------------
# Fix 2 — scikit-learn fijado a la version validada
# ---------------------------------------------------------------------------

def test_scikit_learn_pinned_in_base_in():
    text = (REPO / "requirements" / "base.in").read_text(encoding="utf-8")
    assert f"scikit-learn=={VALIDATED_SKLEARN}" in text


def test_scikit_learn_pinned_in_base_lock():
    text = (REPO / "requirements" / "base.lock.txt").read_text(encoding="utf-8")
    assert f"scikit-learn=={VALIDATED_SKLEARN}" in text


def test_installed_scikit_learn_matches_pin():
    assert sklearn.__version__ == VALIDATED_SKLEARN


def test_production_scalers_load_without_version_warning():
    """Cargar los scalers de produccion no debe emitir InconsistentVersionWarning."""
    from configs.settings import SCALER_EMBER_PATH, SCALER_OVERLAY_PATH

    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        scaler_ember = joblib.load(SCALER_EMBER_PATH)
        scaler_overlay = joblib.load(SCALER_OVERLAY_PATH)

    assert scaler_ember.n_features_in_ == 2381
    assert scaler_overlay.n_features_in_ == 6


# ---------------------------------------------------------------------------
# Fix 4 — MODEL_PATH env var muerta
# ---------------------------------------------------------------------------

_MODEL_PATH_ENV = re.compile(
    r"os\.(?:getenv|environ(?:\.get)?)\s*[\(\[]\s*[\"']MODEL_PATH[\"']"
)
_SKIP_DIRS = {".venv", "__pycache__", ".git", "build", "dist", ".pytest_cache"}


def test_no_code_reads_model_path_env_var():
    """Ningun modulo Python lee la variable de entorno MODEL_PATH."""
    offenders = []
    for py in REPO.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in py.relative_to(REPO).parts):
            continue
        if _MODEL_PATH_ENV.search(py.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(str(py.relative_to(REPO)))
    assert offenders == [], f"referencias funcionales a MODEL_PATH: {offenders}"


def test_env_file_has_no_model_path_when_present():
    """El .env local ya no debe declarar MODEL_PATH (configuracion muerta)."""
    env = REPO / ".env"
    if not env.exists():
        pytest.skip(".env ausente en este entorno")
    content = env.read_text(encoding="utf-8", errors="ignore")
    assert "MODEL_PATH=" not in content
