"""
core/explain/shap_explainer.py — SHAP KernelExplainer sobre ONNX Runtime.

Implementa la Tarea T-12 de F3: explicabilidad del modelo ML sin PyTorch.

Dependencias (solo ml.in, nunca base.in):
    shap>=0.44.0
    onnxruntime  (ya en base.in)
    numpy        (ya en base.in)
    joblib       (ya en base.in)

El modulo usa un patron de lazy-init: la primera llamada a explain() carga
el modelo si aun no esta cargado. Esto permite importar el modulo sin que
falle si shap no esta instalado (se captura en explain()).
"""
from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import onnxruntime as ort

from core.overlay_features import OVERLAY_FEATURE_NAMES
from models.features_v1_1 import build_features_2387

logger = logging.getLogger("core.explain.shap")

# Numero de features del vector SOREL-20M (contrato invariante del extractor)
EMBER_FEATURE_DIM = 2381
# Dimension de entrada del MLP (ShadowNetFeatures_v1.1 = EMBER_2381 + OVERLAY_6)
MODEL_FEATURE_DIM = 2387

# Timeout para la inferencia SHAP en segundos
SHAP_TIMEOUT_SECONDS = 30

# Directorio de modelos relativo a la raiz del proyecto
_HERE = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODEL_PATH = _HERE / "models" / "shadow_net_sorel_7m_v1.1.onnx"
_DEFAULT_SCALER_EMBER_PATH = _HERE / "models" / "scaler_ember_v1.1.pkl"
_DEFAULT_SCALER_OVERLAY_PATH = _HERE / "models" / "scaler_overlay_v1.1.pkl"
_TEST_SET_PATH = _HERE / "data" / "test_set" / "X_test.npy"


def _generate_synthetic_background(n: int = 100) -> np.ndarray:
    """
    Genera un background sintetico cuando X_test.npy no esta disponible.

    Usa una mezcla de ceros y muestras gaussianas para representar
    la distribucion aproximada de un PE benigno sin datos reales.
    El resultado es determinista (seed fijo) para reproducibilidad.
    """
    rng = np.random.default_rng(seed=42)
    zeros = np.zeros((n // 2, EMBER_FEATURE_DIM), dtype=np.float32)
    gaussian = rng.normal(0.0, 0.1, (n - n // 2, EMBER_FEATURE_DIM)).astype(np.float32)
    bg = np.vstack([zeros, gaussian])
    return bg


def _load_background(n: int = 100) -> np.ndarray:
    """
    Carga el background para KernelExplainer.

    Intenta X_test.npy (hasta n muestras). Si no existe o es incompatible,
    genera un background sintetico determinista.

    Args:
        n: Numero de muestras de background.

    Returns:
        Array de shape (n, EMBER_FEATURE_DIM) en float32.
    """
    if _TEST_SET_PATH.exists():
        try:
            data = np.load(str(_TEST_SET_PATH))
            if data.ndim == 2 and data.shape[1] == EMBER_FEATURE_DIM:
                idx = np.random.default_rng(42).integers(0, len(data), min(n, len(data)))
                bg = data[idx].astype(np.float32)
                logger.info(
                    "Background SHAP cargado desde X_test.npy: %d muestras", len(bg)
                )
                return bg
        except Exception as exc:
            logger.warning("No se pudo cargar X_test.npy para background SHAP: %s", exc)

    logger.info("Usando background SHAP sintetico (%d muestras)", n)
    return _generate_synthetic_background(n)


def _build_feature_names() -> List[str]:
    """
    Genera los nombres de las 2381 features del extractor SOREL-20M.

    Los nombres siguen el mismo orden que PEFeatureExtractor.extract():
      - byte_histogram[0..255]      (256)
      - byte_entropy[0..255]        (256)
      - string_*                    (5)
      - general_*                   (10)
      - header_*                    (62)
      - section_*                   (255)
      - imports_*                   (1280)
      - exports_*                   (128)
      - misc_*                      (resto hasta 2381)
    """
    names: List[str] = []
    names += [f"byte_histogram_{i}" for i in range(256)]
    names += [f"byte_entropy_{i}" for i in range(256)]
    names += ["strings_numstrings", "strings_avlength", "strings_printables",
              "strings_entropy", "strings_paths"]
    names += [f"general_{i}" for i in range(10)]
    names += [f"header_{i}" for i in range(62)]
    names += [f"section_{i}" for i in range(255)]
    names += [f"imports_{i}" for i in range(1280)]
    names += [f"exports_{i}" for i in range(128)]
    # Rellenar el resto con nombres genericos hasta EMBER_FEATURE_DIM
    remaining = EMBER_FEATURE_DIM - len(names)
    names += [f"feature_{len(names) + i}" for i in range(remaining)]
    # Bloque OVERLAY_6 (ShadowNetFeatures_v1.1) al final, en el orden canonico
    names += [f"overlay_{n}" for n in OVERLAY_FEATURE_NAMES]
    assert len(names) == MODEL_FEATURE_DIM, len(names)
    return names


class ShapExplainer:
    """
    SHAP KernelExplainer sobre ONNX Runtime — sin PyTorch.

    Carga el modelo 2387 (ShadowNetFeatures_v1.1) y sus dos scalers una sola vez y
    expone el metodo explain() para obtener las top-k features con mayor contribucion SHAP.

    Uso:
        explainer = ShapExplainer()
        result = explainer.explain(feature_vector_ember_2381, top_k=20)
        # result["top_features"] → lista de {feature_idx, feature_name, shap_value}
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_ember_path: Optional[Path] = None,
        scaler_overlay_path: Optional[Path] = None,
        background_n: int = 100,
    ):
        """
        Inicializa el explainer cargando modelos y background.

        Args:
            model_path:          Ruta al ONNX 2387. Por defecto usa models/.
            scaler_ember_path:   Ruta al scaler EMBER (2381). Por defecto usa models/.
            scaler_overlay_path: Ruta al scaler OVERLAY (6). Por defecto usa models/.
            background_n:        Numero de muestras para el background de KernelExplainer.
        """
        self.model_path = Path(model_path or _DEFAULT_MODEL_PATH)
        self.scaler_ember_path = Path(scaler_ember_path or _DEFAULT_SCALER_EMBER_PATH)
        self.scaler_overlay_path = Path(scaler_overlay_path or _DEFAULT_SCALER_OVERLAY_PATH)

        logger.info(
            "Inicializando ShapExplainer: model=%s scaler_ember=%s scaler_overlay=%s background_n=%d",
            self.model_path.name,
            self.scaler_ember_path.name,
            self.scaler_overlay_path.name,
            background_n,
        )

        self.scaler_ember = joblib.load(str(self.scaler_ember_path))
        self.scaler_overlay = joblib.load(str(self.scaler_overlay_path))
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.feature_names = _build_feature_names()
        self.background = _load_background(background_n)

        logger.info(
            "ShapExplainer listo: input=%s features=%d background=%d",
            self.input_name,
            len(self.feature_names),
            len(self.background),
        )

    def _predict_fn(self, X: np.ndarray) -> np.ndarray:
        """
        Funcion de prediccion compatible con shap.KernelExplainer.

        Recibe el vector 2387 ya construido (EMBER escalado | OVERLAY escalado)
        y ejecuta solo la inferencia ONNX.

        Args:
            X: Array de shape (n, 2387) en float32 ya escalado.

        Returns:
            Array de scores de shape (n,) en float32.
        """
        return (
            self.session.run(None, {self.input_name: X.astype(np.float32)})[0]
            .ravel()
            .astype(np.float64)
        )

    def explain(
        self, features: np.ndarray, top_k: int = 20
    ) -> Dict[str, Any]:
        """
        Calcula los valores SHAP para el vector de features dado.

        Escala las features con StandardScaler, ejecuta KernelExplainer
        con nsamples=100 (suficiente para < 30s) y retorna las top_k
        features con mayor contribucion absoluta.

        Args:
            features: Vector de features de shape (2381,) o (1, 2381).
            top_k:    Numero de features a retornar (default 20).

        Returns:
            Dict con:
                - top_features: Lista de {feature_idx, feature_name, shap_value}
                - base_value:   Valor base del explainer (expected value)
                - model_score:  Score ONNX sobre las features escaladas

            O en caso de error:
                - error: "shap_not_installed" | "shap_timeout"
                - top_features: []
        """
        # Verificar que shap esta instalado (dependencia opcional)
        try:
            import shap  # noqa: F401
        except ImportError:
            logger.warning("shap no instalado — retornando shap_not_installed")
            return {"error": "shap_not_installed", "top_features": []}

        # Normalizar shape del vector de entrada (EMBER-2381 del extractor)
        feat = np.asarray(features, dtype=np.float32).reshape(1, EMBER_FEATURE_DIM)

        # Construir el vector 2387 = [EMBER escalado | OVERLAY escalado] para el
        # vector y para el background, con los mismos scalers de produccion.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled_feat = build_features_2387(feat, self.scaler_ember, self.scaler_overlay)
            scaled_bg = build_features_2387(
                np.asarray(self.background, dtype=np.float32),
                self.scaler_ember,
                self.scaler_overlay,
            )

        model_score = float(self._predict_fn(scaled_feat)[0])

        def _run_shap() -> Dict[str, Any]:
            import shap as _shap

            explainer = _shap.KernelExplainer(self._predict_fn, scaled_bg)
            shap_values = explainer.shap_values(scaled_feat, nsamples=100, silent=True)

            base_value = float(
                explainer.expected_value
                if hasattr(explainer, "expected_value")
                else 0.0
            )

            # shap_values puede ser list (clasificacion) o ndarray (regresion)
            vals = (
                shap_values[0]
                if isinstance(shap_values, list)
                else np.asarray(shap_values).ravel()
            )

            # Top-k features por contribucion absoluta
            top_idx = np.argsort(np.abs(vals))[::-1][:top_k]
            top_features = [
                {
                    "feature_idx": int(i),
                    "feature_name": self.feature_names[i],
                    "shap_value": float(vals[i]),
                }
                for i in top_idx
            ]

            logger.info(
                "SHAP calculado: top_feature=%s shap=%.4f base=%.4f score=%.4f",
                top_features[0]["feature_name"] if top_features else "?",
                top_features[0]["shap_value"] if top_features else 0.0,
                base_value,
                model_score,
            )

            return {
                "top_features": top_features,
                "base_value": base_value,
                "model_score": model_score,
            }

        # Ejecutar SHAP con timeout para no bloquear el pipeline
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_shap)
            try:
                return future.result(timeout=SHAP_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                logger.error(
                    "SHAP timeout tras %ds — retornando shap_timeout",
                    SHAP_TIMEOUT_SECONDS,
                )
                return {"error": "shap_timeout", "top_features": []}
            except Exception as exc:
                logger.error("Error en SHAP: %s", exc)
                return {"error": str(exc), "top_features": []}
