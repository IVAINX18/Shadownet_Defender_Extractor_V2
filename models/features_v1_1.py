"""
models/features_v1_1.py — Construye el vector de entrada 2387 (ShadowNetFeatures_v1.1).

Contrato (identico al entrenamiento en Kaggle):

    X = [EMBER_2381 escalado | OVERLAY_6 escalado]

- El extractor local produce el vector EMBER-2381 (sin cambiar su semantica).
- El bloque OVERLAY_6 se deriva del bloque General EMBER con
  `core.overlay_features.compute_overlay_features` (funcion pura, misma logica
  que en entrenamiento; ver tests/test_overlay_contract.py).
- Cada bloque se escala con su StandardScaler propio; nunca se re-entrena aqui.

Uso:
    X2387 = build_features_2387(raw_2381, scaler_ember, scaler_overlay)
"""
from __future__ import annotations

import numpy as np

from core.overlay_features import OVERLAY_FEATURE_NAMES, N_OVERLAY, compute_overlay_features

# Orden canonico del bloque OVERLAY (idem entrenamiento)
OVERLAY_NAMES = list(OVERLAY_FEATURE_NAMES)


def build_features_2387(
    raw: np.ndarray,
    scaler_ember,
    scaler_overlay,
) -> np.ndarray:
    """Concatena EMBER_2381 escalado con OVERLAY_6 escalado → (2387,) o (n, 2387).

    Args:
        raw: vector EMBER-2381 crudo, shape (2381,) o (n, 2381) float32.
        scaler_ember: StandardScaler ajustado sobre las 7M (n_features=2381).
        scaler_overlay: StandardScaler ajustado sobre OVERLAY_6 (n_features=6).

    Returns:
        Array float32 de shape (2387,) o (n, 2387), finito.
    """
    v = np.asarray(raw, dtype=np.float32)
    single = v.ndim == 1
    if single:
        v = v[np.newaxis, :]
    if v.shape[1] < N_OVERLAY or v.shape[1] != 2381:
        raise ValueError(f"vector EMBER inesperado: {v.shape}")

    emb = scaler_ember.transform(v).astype(np.float32)
    ov_raw = compute_overlay_features(v)  # (n, 6) float32, misma logica que entrenamiento
    ov = scaler_overlay.transform(ov_raw).astype(np.float32)
    x = np.concatenate([emb, ov], axis=1).astype(np.float32)
    if not np.all(np.isfinite(x)):
        raise ValueError("vector 2387 contiene NaN/Inf")
    return x[0] if single else x