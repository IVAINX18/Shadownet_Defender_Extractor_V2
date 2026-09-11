#!/usr/bin/env python3
"""
scripts/export_onnx_v1_1.py — Exporta el MLP 2387 (ShadowNetFeatures_v1.1) a ONNX.

Origen: checkpoint PyTorch v5 `shadow_net_sorel_7m_v1.1_v5_best.pth` (entrenado en
Kaggle: MLP 2387->512->256->128->1, BCEWithLogitsLoss, 2 epocas).

Contrato de exportacion:
  - Entrada:  [batch, 2387] float32 (EMBER_2381 escalado | OVERLAY_6 escalado).
  - Salida:   [batch, 1]    float32 en [0, 1] (probabilidad de malware).
  - Se incluye sigmoid en el grafo: la salida es la probabilidad, identica a la
    semantica del modelo de produccion anterior (score en [0, 1]). El checkpoint
    sigue siendo logits (BCEWithLogitsLoss); el sigmoid se anade solo en el ONNX.
  - BatchNorm en modo eval (running stats congeladas), Dropout identidad.

Valida contra onnxruntime con batchs 1/8/64 (max diff < 1e-5) y onnx.checker.

Uso (desde la raiz del repo, con el venv):
    .venv/bin/python scripts/export_onnx_v1_1.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "Model_Collab/Kaggle-MLP-PE/v5/checkpoints/shadow_net_sorel_7m_v1.1_v5_best.pth"
DEFAULT_OUT = ROOT / "models/shadow_net_sorel_7m_v1.1.onnx"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class ShadowNetMLP(torch.nn.Module):
    """MLP 2387 -> 512 -> 256 -> 128 -> 1 (ReLU + BatchNorm + Dropout 0.3/0.2/0.1)."""

    def __init__(self, d_in: int = 2387):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, 512), torch.nn.BatchNorm1d(512), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256), torch.nn.BatchNorm1d(256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128), torch.nn.BatchNorm1d(128), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


class ShadowNetMLPProbability(torch.nn.Module):
    """Wrapper para ONNX: devuelve la probabilidad (sigmoid sobre logits)."""

    def __init__(self, d_in: int = 2387):
        super().__init__()
        self.net = ShadowNetMLP(d_in).net

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def export(checkpoint_path: Path, output_path: Path, opset: int = 17) -> None:
    torch.manual_seed(42)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    model = ShadowNetMLPProbability(2387)
    model.load_state_dict(sd)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"checkpoint: {checkpoint_path}")
    print(f"  epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss'):.4f} params={n_params}")
    assert n_params == 1388801, n_params

    dummy = torch.zeros(1, 2387, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["features"],
        output_names=["score"],
        dynamic_axes={"features": {0: "batch"}, "score": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,  # exportador legacy (TorchScript): evita dependencia de bz2/onnxscript en este venv
    )
    print(f"ONNX escrito: {output_path} ({output_path.stat().st_size/2**20:.2f} MB)")

    # Validacion estructural
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("onnx.checker: OK")
    inp = onnx_model.graph.input[0]
    out = onnx_model.graph.output[0]
    print(f"  input:  {inp.name} {[d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]}")
    print(f"  output: {out.name} {[d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]}")

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    i_name = session.get_inputs()[0].name

    # Paridad numerica torch (con sigmoid) vs onnxruntime
    max_diff = 0.0
    for n in (1, 8, 64):
        x = torch.randn(n, 2387, dtype=torch.float32)
        with torch.no_grad():
            expected = torch.sigmoid(model.net(x)).numpy()
        got = session.run(None, {i_name: x.numpy()})[0]
        got = got.reshape(-1, 1)
        d = float(np.abs(got - expected).max())
        max_diff = max(max_diff, d)
        assert got.min() >= 0.0 and got.max() <= 1.0 + 1e-6, "salida fuera de [0,1]"
        assert got.shape == (n, 1), got.shape
        print(f"  batch={n}: shape={got.shape} rango=[{got.min():.6f},{got.max():.6f}] max_diff={d:.3e}")
    assert max_diff < 1e-5, f"max_diff={max_diff} >= 1e-5"

    print(f"sha256: {sha256_file(output_path)}")
    print(f"EXPORT ONNX v1.1: PASS ({output_path})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Exportar MLP 2387 a ONNX")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.opset)
    return 0


if __name__ == "__main__":
    sys.exit(main())