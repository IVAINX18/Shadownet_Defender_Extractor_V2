#!/usr/bin/env python3
"""
scripts/compare_ember_vs_sorel.py — Cross-check del EMBER vendoreado contra
vectores EMBER almacenados por SOREL-20M (solo lectura, descargas minimas).

Procedimiento:
  manifest local (row_idx, sha256, is_malware)
      -> seleccionar N muestras de malware espaciadas con binario pequeno en S3
      -> descargar binario (zlib) + descomprimir
      -> Range-fetch del vector EMBER almacenado en train-features.npz (fila row_idx)
      -> extraer con extractors/ember_features.py (vendoreado)
      -> comparar por bloque (sin modificar ningun vector)

No cambia tolerancias para esconder errores. Si algo no coincide, el reporte
muestra el bloque, la magnitud y los extremos para diagnosticar.

Uso (desde la raiz del repo, con el venv):
    .venv/bin/python scripts/compare_ember_vs_sorel.py --n 5
    .venv/bin/python scripts/compare_ember_vs_sorel.py --n 3 --max-bytes 2097152
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NPZ_URL = "http://sorel-20m.s3.amazonaws.com/09-DEC-2020/lightGBM-features/train-features.npz"
BIN_URL = "http://sorel-20m.s3.amazonaws.com/09-DEC-2020/binaries/{}"
ROW_BYTES = 2381 * 4
EXPECTED_COLS = 2381
PACE = 0.35

BLOCKS = {
    "histogram": (0, 256),
    "byteentropy": (256, 512),
    "strings": (512, 616),
    "general": (616, 626),
    "header": (626, 688),
    "section": (688, 943),
    "imports": (943, 2223),
    "exports": (2223, 2351),
    "datadirectories": (2351, 2381),
}


def _http_range(url: str, a: int, b: int, tries: int = 6, timeout: int = 120) -> bytes:
    import urllib.error

    last = None
    for k in range(tries):
        try:
            time.sleep(PACE if k == 0 else 6 * k)
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
            r = urllib.request.urlopen(req, timeout=timeout)
            if r.status != 206:
                raise RuntimeError(f"Range no honrado: {r.status}")
            d = r.read()
            if len(d) != b - a + 1:
                raise RuntimeError(f"truncado: {len(d)} vs {b - a + 1}")
            return d
        except urllib.error.HTTPError as e:
            if e.code in (416, 503, 429, 500):
                last = e
                continue
            raise
    raise RuntimeError(f"rango agotado {a}-{b}: {last}")


def _find_pay0() -> tuple:
    """Localiza arr_0 STORED en el npz y devuelve (PAY0, shape)."""
    # Tamano total verificado del npz.
    probe = urllib.request.Request(NPZ_URL, headers={"Range": "bytes=0-0"})
    r = urllib.request.urlopen(probe, timeout=60)
    content_range = r.headers.get("Content-Range", "")
    total = int(content_range.split("/")[-1])
    tail = _http_range(NPZ_URL, total - 131072, total - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd == -1:
        raise RuntimeError("EOCD no encontrado")
    vals = struct.unpack("<HHHHIIH", tail[eocd + 4:eocd + 22])
    _, _, _, n_total, cd_size, cd_off, _ = vals
    if cd_off == 0xFFFFFFFF:
        loc = tail.rfind(b"PK\x06\x07")
        if loc == -1:
            raise RuntimeError("sin locator ZIP64")
        _, zoff, _ = struct.unpack("<IQI", tail[loc + 4:loc + 20])
        z = _http_range(NPZ_URL, zoff, zoff + 55)
        if z[:4] != b"PK\x06\x06":
            raise RuntimeError("firma ZIP64 ausente")
        _, _, _, _, _, _, n_total, cd_size, cd_off = struct.unpack("<QHHIIQQQQ", z[4:56])
    cd_blob = _http_range(NPZ_URL, cd_off, cd_off + cd_size - 1)
    entries, pos = [], 0
    while pos < len(cd_blob):
        if cd_blob[pos:pos + 4] != b"PK\x01\x02":
            raise RuntimeError(f"firma CD corrupta en {pos}")
        f = struct.unpack("<HHHHHIIIHHHHHII", cd_blob[pos + 6:pos + 46])
        _, _, comp, _, _, _, csz32, usz32, nlen, elen, clen, _, _, _, lho32 = f
        name = cd_blob[pos + 46:pos + 46 + nlen].decode()
        extra = cd_blob[pos + 46 + nlen:pos + 46 + nlen + elen]
        csize, usize, lho = csz32, usz32, lho32
        j = 0
        while j + 4 <= len(extra):
            hid, dsz = struct.unpack("<HH", extra[j:j + 4])
            if hid == 0x0001:
                nq = dsz // 8
                v = struct.unpack("<" + "Q" * nq, extra[j + 4:j + 4 + 8 * nq])
                vi = 0
                if usize == 0xFFFFFFFF:
                    usize = v[vi]; vi += 1
                if csize == 0xFFFFFFFF:
                    csize = v[vi]; vi += 1
                if lho == 0xFFFFFFFF:
                    lho = v[vi]; vi += 1
                break
            j += 4 + dsz
        entries.append({"name": name, "compress": comp, "lho": lho})
        pos += 46 + nlen + elen + clen
    arr = [e for e in entries if e["name"].endswith(".npy")]
    if not arr or any(e["compress"] != 0 for e in arr):
        raise RuntimeError("arr_0 no STORED")
    lho = arr[0]["lho"]
    lh = _http_range(NPZ_URL, lho, lho + 29)
    if lh[:4] != b"PK\x03\x04" or struct.unpack("<H", lh[8:10])[0] != 0:
        raise RuntimeError("Local File Header no STORED")
    nlen = struct.unpack("<H", lh[26:28])[0]
    elen = struct.unpack("<H", lh[28:30])[0]
    data_off = lho + 30 + nlen + elen
    head = _http_range(NPZ_URL, data_off, data_off + 127)
    if head[:6] != b"\x93NUMPY":
        raise RuntimeError("magia NPY ausente")
    hlen = struct.unpack("<H", head[8:10])[0]
    hdr = _http_range(NPZ_URL, data_off, data_off + 10 + hlen - 1)[10:].decode("latin1")
    d = eval(hdr)  # noqa: S307 - formato controlado por numpy, magia ya validada
    shape = tuple(d["shape"])
    if shape[1] != EXPECTED_COLS or np.dtype(d["descr"]) != np.dtype("<f4"):
        raise RuntimeError(f"shape/dtype inesperados: {shape} {d['descr']}")
    return data_off + 10 + hlen, shape


def _fetch_row(pay0: int, row: int) -> np.ndarray:
    d = _http_range(NPZ_URL, pay0 + row * ROW_BYTES, pay0 + (row + 1) * ROW_BYTES - 1)
    v = np.frombuffer(d, dtype=np.float32).reshape(-1)
    if v.shape != (EXPECTED_COLS,) or not np.all(np.isfinite(v)):
        raise RuntimeError(f"fila {row} invalida: {v.shape}")
    return v.astype(np.float32)


def _head_size(sha: str):
    req = urllib.request.Request(BIN_URL.format(sha), method="HEAD")
    try:
        r = urllib.request.urlopen(req, timeout=30)
        if r.status != 200:
            return None
        return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return None


def _fetch_binary(sha: str) -> bytes:
    req = urllib.request.Request(BIN_URL.format(sha))
    r = urllib.request.urlopen(req, timeout=300)
    if r.status != 200:
        raise RuntimeError(f"binario {sha[:12]} status={r.status}")
    data = r.read()
    # Los binarios de SOREL estan comprimidos con zlib; si no, usar tal cual.
    try:
        out = zlib.decompress(data)
        return out
    except Exception:
        return data


def compare_block(a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "n": int(a.shape[0]),
        "ndiff_1e6": int((diff > 1e-6).sum()),
        "ndiff_1e3": int((diff > 1e-3).sum()),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-check EMBER vendoreado vs SOREL")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-bytes", type=int, default=5 * 1024 * 1024)
    ap.add_argument("--manifest", type=str,
                    default="Model_Collab/Kaggle-MLP-PE/v5/dataset/train_7m_manifest.parquet")
    args = ap.parse_args()

    import pandas as pd
    from extractors.ember_features import EmberPEFeatureExtractor

    df = pd.read_parquet(args.manifest, columns=["row_idx", "sha256", "is_malware"])
    mal = df[df["is_malware"] == 1].reset_index(drop=True)
    # Candidatos espaciados; quedarse con los que existen en S3 y son pequenos.
    step = max(1, len(mal) // 40)
    chosen = []
    for i in range(0, len(mal), step):
        if len(chosen) >= args.n:
            break
        r = mal.iloc[i]
        size = _head_size(r["sha256"])
        if size and 0 < size <= args.max_bytes:
            chosen.append({"row_idx": int(r["row_idx"]), "sha256": r["sha256"], "bytes": size})
    report = {"n_requested": args.n, "n_selected": len(chosen), "samples": []}
    if not chosen:
        report["error"] = "no se encontraron binarios SOREL pequenos; cross-check inconcluso"
        print(json.dumps(report, indent=1))
        return 2

    pay0, shape = _find_pay0()
    report["npz_shape"] = list(shape)
    ex = EmberPEFeatureExtractor()
    for c in chosen:
        entry = dict(c)
        try:
            binary = _fetch_binary(c["sha256"])
            entry["binary_bytes"] = len(binary)
            entry["is_pe_mz"] = bool(binary[:2] == b"MZ")
            stored = _fetch_row(pay0, c["row_idx"])
            entry["stored_shape"] = list(stored.shape)
            mine = ex.feature_vector(binary)
            entry["mine_shape"] = list(mine.shape)
            entry["blocks"] = {}
            for name, (a, b) in BLOCKS.items():
                entry["blocks"][name] = compare_block(stored[a:b], mine[a:b])
            entry["global_max_abs"] = max(v["max_abs"] for v in entry["blocks"].values())
            entry["status"] = "ok"
        except Exception as e:
            entry["status"] = f"error: {type(e).__name__}: {e}"
        report["samples"].append(entry)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
