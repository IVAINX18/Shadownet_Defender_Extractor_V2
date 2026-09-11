import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Any
from utils.logger import setup_logger
from utils.runtime_checks import validate_python_version, import_optional_dependency
from models.features_v1_1 import build_features_2387

validate_python_version()
ort = import_optional_dependency(
    "onnxruntime",
    install_profile="requirements/base.lock.txt",
)

logger = setup_logger(__name__)

class ShadowNetModel:
    """
    Wrapper para ShadowNet Defender ONNX model (ShadowNetFeatures_v1.1, 2387 dims).

    Contrato de inferencia:
      - Entrada: vector EMBER-2381 del extractor local (contrato invariable).
      - Se deriva OVERLAY_6 desde el bloque General EMBER y se escala cada bloque
        con su StandardScaler propio (EMBER 2381 + OVERLAY 6) → [EMBER | OVERLAY] = 2387.
      - El ONNX ya incluye sigmoid: devuelve probabilidad en [0, 1].
    """

    def __init__(self, model_path: Path, scaler_ember_path: Path, scaler_overlay_path: Path):
        self.model_path = model_path
        self.scaler_ember_path = scaler_ember_path
        self.scaler_overlay_path = scaler_overlay_path
        self.session: Optional[Any] = None
        self.scaler_ember = None
        self.scaler_overlay = None
        self.input_name = None

        self.load()

    def load(self):
        """Carga el modelo ONNX y los dos scalers (EMBER 2381 + OVERLAY 6)."""
        try:
            for p in (self.model_path, self.scaler_ember_path, self.scaler_overlay_path):
                if not p.exists():
                    raise FileNotFoundError(f"Artefacto no encontrado: {p}")

            logger.info(f"Loading model from {self.model_path}...")
            self.session = ort.InferenceSession(str(self.model_path))
            self.input_name = self.session.get_inputs()[0].name
            in_shape = self.session.get_inputs()[0].shape
            if in_shape and len(in_shape) == 2 and in_shape[1] not in (None, 2387):
                logger.warning(
                    "Entrada ONNX inesperada: %s (esperado [batch, 2387])", in_shape
                )

            logger.info(f"Loading scaler EMBER from {self.scaler_ember_path}...")
            self.scaler_ember = joblib.load(self.scaler_ember_path)
            logger.info(f"Loading scaler OVERLAY from {self.scaler_overlay_path}...")
            self.scaler_overlay = joblib.load(self.scaler_overlay_path)
            if getattr(self.scaler_ember, "n_features_in_", None) != 2381:
                logger.warning("scaler EMBER con n_features != 2381")
            if getattr(self.scaler_overlay, "n_features_in_", None) != 6:
                logger.warning("scaler OVERLAY con n_features != 6")

            logger.info("Model and scalers loaded successfully.")

        except Exception as e:
            logger.critical(f"Failed to load model/scalers: {e}")
            raise

    def predict(self, features: np.ndarray) -> float:
        """
        Realiza inferencia sobre el vector EMBER-2381 del extractor.

        Args:
            features: vector 1D de 2381 features (EMBER crudo).

        Returns:
            Probabilidad de malware en [0.0 - 1.0].
        """
        try:
            features = np.asarray(features, dtype=np.float32).reshape(1, -1)
            if features.shape[1] != 2381:
                logger.warning(
                    "Vector EMBER inesperado: %s (esperado 2381)", features.shape
                )

            # [EMBER_2381 escalado | OVERLAY_6 escalado] = 2387
            scaled_features = build_features_2387(
                features, self.scaler_ember, self.scaler_overlay
            )

            # Inferencia ONNX (el grafo ya incluye sigmoid → probabilidad)
            inputs = {self.input_name: scaled_features}
            outputs = self.session.run(None, inputs)

            output_tensor = outputs[0]
            score = float(np.reshape(output_tensor, -1)[0])
            score = float(np.clip(score, 0.0, 1.0))
            return score

        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise
