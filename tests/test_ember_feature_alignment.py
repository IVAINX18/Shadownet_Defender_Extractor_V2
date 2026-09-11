"""
tests/test_ember_feature_alignment.py — Contrato de alineacion EMBER v2.

Verifica que el extractor local produce features en el espacio de entrenamiento
(SOREL-20M / EMBER v2 canonico) y que la explosion sistematica de z-scores
(|z| ~ 1e11 por el layout legacy desalineado) no existe.

Nota honesta sobre el umbral: los bloques hasheados (Header/Section/Imports)
pueden activar buckets raros con |z| de hasta ~50 de forma LEGITIMA (valor
hasheado +-1 en un bucket que SOREL casi no usa; ver cross-check SOREL en
scripts/compare_ember_vs_sorel.py). Por eso este test prohibe la explosion
sistematica (max|z| < 100, <1% de dims con |z|>10) en lugar de exigir |z|<=10
en todo, y NUNCA modifica ni recorta valores para pasar.
"""
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from extractors.extractor import PEFeatureExtractor
from extractors.ember_features import EmberPEFeatureExtractor, EMBER_V2_DIM
from core.overlay_features import OVERLAY_FEATURE_NAMES, N_OVERLAY

CORPUS = [
    "samples/sample1.exe",
    "samples/sample2.exe",
    "samples/procexp64.exe",
]

# Umbrales anti-explosion (el bug legacy daba |z| ~ 4e11 en ~30 columnas).
# Los extremos hasheados legitimos observados llegan a |z| ~ 48 en <0.4% de dims.
MAX_ABS_Z = 100.0
MAX_FRAC_BIG_Z = 0.01


def _corpus_vectors():
    ext = PEFeatureExtractor()
    out = {}
    for rel in CORPUS:
        p = _PROJECT_ROOT / rel
        if not p.exists():
            pytest.skip(f"corpus no encontrado: {rel}")
        out[rel] = np.asarray(ext.extract(str(p)), dtype=np.float32)
    return out


def _ember_scaler():
    import joblib

    p = _PROJECT_ROOT / "models" / "scaler_ember_v1.1.pkl"
    if not p.exists():
        pytest.skip("scaler EMBER v1.1 no encontrado")
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        return joblib.load(str(p))


def test_1_dimensions():
    """EMBER=2381, OVERLAY=6, TOTAL=2387."""
    assert EMBER_V2_DIM == 2381
    assert N_OVERLAY == 6
    assert len(OVERLAY_FEATURE_NAMES) == 6
    vecs = _corpus_vectors()
    for rel, v in vecs.items():
        assert v.shape == (2381,), f"{rel}: {v.shape}"
    from models.features_v1_1 import build_features_2387

    sc_e, sc_o = _ember_scaler(), None
    import joblib

    p = _PROJECT_ROOT / "models" / "scaler_overlay_v1.1.pkl"
    if not p.exists():
        pytest.skip("scaler OVERLAY v1.1 no encontrado")
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        sc_o = joblib.load(str(p))
    for rel, v in vecs.items():
        x = build_features_2387(v, sc_e, sc_o)
        assert x.shape == (2387,), f"{rel}: {x.shape}"


def test_2_layout_ranges():
    """BLOCK_RANGES cubre 2381 con DataDirectories al final, en orden canonico."""
    expected = {
        "ByteHistogram": (0, 256),
        "ByteEntropy": (256, 512),
        "Strings": (512, 616),
        "General": (616, 626),
        "Header": (626, 688),
        "Section": (688, 943),
        "Imports": (943, 2223),
        "Exports": (2223, 2351),
        "DataDirectories": (2351, 2381),
    }
    assert dict(PEFeatureExtractor.BLOCK_RANGES) == expected
    total = sum(e - s for s, e in PEFeatureExtractor.BLOCK_RANGES.values())
    assert total == 2381


def test_3_finite():
    """Sin NaN/Inf en el corpus local."""
    for rel, v in _corpus_vectors().items():
        assert np.all(np.isfinite(v)), f"{rel} con NaN/Inf"


def test_4_no_systematic_explosion():
    """Sin explosion sistematica de z-scores (el bug legacy daba |z| ~ 4e11).

    Permite extremos hasheados legitimos aislados (documentados), sin recortar.
    """
    sc = _ember_scaler()
    for rel, v in _corpus_vectors().items():
        z = (v.astype(np.float64) - sc.mean_) / sc.scale_
        assert np.all(np.isfinite(z)), f"{rel}: z no finito"
        assert float(np.abs(z).max()) < MAX_ABS_Z, f"{rel}: max|z|={np.abs(z).max():.3g}"
        frac = float((np.abs(z) > 10).mean())
        assert frac < MAX_FRAC_BIG_Z, f"{rel}: {frac:.3%} dims con |z|>10"


def test_5_parity_with_canonical():
    """extract() == EMBER vendoreado directo sobre los mismos bytes."""
    ext = PEFeatureExtractor()
    ref = EmberPEFeatureExtractor()
    for rel in CORPUS:
        p = _PROJECT_ROOT / rel
        if not p.exists():
            pytest.skip(f"corpus no encontrado: {rel}")
        got = np.asarray(ext.extract(str(p)), dtype=np.float32)
        want = ref.feature_vector(p.read_bytes())
        assert got.shape == want.shape == (2381,)
        assert np.array_equal(got, want), f"{rel}: difieren en {(got != want).sum()} dims"


def test_6_fallback_non_pe(tmp_path):
    """Un no-PE sigue entrando por RAW_FALLBACK (contingencia preservada)."""
    f = tmp_path / "nota.txt"
    f.write_bytes(b"Este texto no es un ejecutable PE ni de lejos.")
    ext = PEFeatureExtractor()
    v = np.asarray(ext.extract(str(f)), dtype=np.float32)
    assert v.shape == (2381,)
    diag = ext.last_diagnostics["diagnostics"]
    assert diag["extraction_mode"] == "RAW_FALLBACK"
    assert "pefile_failed" in (diag.get("degradation_reason") or "")


def test_7_pipeline_extract_scale_model():
    """extract -> 2387 -> predict da score finito en [0,1] (sin integrar nada nuevo)."""
    from models.inference import ShadowNetModel
    from models.features_v1_1 import build_features_2387
    from configs.settings import MODEL_PATH, SCALER_EMBER_PATH, SCALER_OVERLAY_PATH
    import joblib

    if not MODEL_PATH.exists():
        pytest.skip("modelo ONNX no encontrado")
    model = ShadowNetModel(MODEL_PATH, SCALER_EMBER_PATH, SCALER_OVERLAY_PATH)
    ext = PEFeatureExtractor()
    for rel in CORPUS:
        p = _PROJECT_ROOT / rel
        if not p.exists():
            pytest.skip(f"corpus no encontrado: {rel}")
        v = np.asarray(ext.extract(str(p)), dtype=np.float32)
        score = model.predict(v)
        assert np.isfinite(score) and 0.0 <= score <= 1.0, f"{rel}: score={score}"
