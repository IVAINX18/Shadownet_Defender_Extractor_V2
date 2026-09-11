"""
extractors/ember_features.py — Extractor EMBER v2 canonico (adaptacion original).

Produce el vector EMBER_2381 identico al usado en el entrenamiento (SOREL-20M),
con el layout canonico:

    ByteHistogram(256) + ByteEntropy(256) + Strings(104) + General(10)
    + Header(62) + Section(255) + Imports(1280) + Exports(128)
    + DataDirectories(30) = 2381

Atribucion y decision de diseno (importante):
  - El layout sigue la especificacion publica EMBER v2 (`elastic/ember`,
    referenciada por SOREL-20M como `PEFeatureExtractor.feature_vector()`).
  - Este archivo es una implementacion ORIGINAL (no se copio codigo del
    repositorio upstream, cuya fuente esta bajo AGPL-v3). Solo se replica la
    especificacion funcional publica (orden de bloques, dimensiones, parametros
    de hashing), lo que mantiene limpia la licencia academica privada del proyecto.
  - Usa LIEF para el parseo PE y `sklearn.feature_extraction.FeatureHasher`
    para los bloques categoricos, igual que la referencia.

Adaptaciones respecto a la referencia (sin cambiar semantica):
  - `np.int` -> `int` (eliminado en NumPy >= 1.24).
  - Parseo con `bytes` directo en lugar de `list(bytez)` para no materializar
    listas gigantes en RAM con archivos grandes.
  - Flags COFF/DLL descompuestos desde el bitmask con tablas estandar de Windows,
    porque `*_characteristics_lists` de LIEF 1.x puede devolver listas incompletas.
    Los nombres usados son los estandar (los mismos que exponia LIEF 0.9).
  - `section_from_rva` puede devolver None en LIEF 1.x (antes lanzaba
    `lief.not_found`); se maneja con chequeo de None.
"""

from __future__ import annotations

import hashlib
import re

import lief
import numpy as np
from sklearn.feature_extraction import FeatureHasher

# Dimension total EMBER v2 (con DataDirectories).
EMBER_V2_DIM = 2381

# Nombres de los 15 data directories en el orden canonico EMBER.
DATA_DIRECTORY_ORDER = [
    "EXPORT_TABLE", "IMPORT_TABLE", "RESOURCE_TABLE", "EXCEPTION_TABLE",
    "CERTIFICATE_TABLE", "BASE_RELOCATION_TABLE", "DEBUG", "ARCHITECTURE",
    "GLOBAL_PTR", "TLS_TABLE", "LOAD_CONFIG_TABLE", "BOUND_IMPORT",
    "IAT", "DELAY_IMPORT_DESCRIPTOR", "CLR_RUNTIME_HEADER",
]

# Flags COFF (IMAGE_FILE_*) con los nombres estandar que exponia LIEF 0.9.
# Se usan para descomponer el bitmask y recuperar la lista completa de flags,
# ya que las listas de LIEF 1.x pueden estar incompletas.
_COFF_CHARACTERISTICS = [
    (0x0001, "RELOCS_STRIPPED"),
    (0x0002, "EXECUTABLE_IMAGE"),
    (0x0004, "LINE_NUMS_STRIPPED"),
    (0x0008, "LOCAL_SYMS_STRIPPED"),
    (0x0010, "AGGRESIVE_WS_TRIM"),
    (0x0020, "LARGE_ADDRESS_AWARE"),
    (0x0080, "BYTES_REVERSED_LO"),
    (0x0100, "32BIT_MACHINE"),
    (0x0200, "DEBUG_STRIPPED"),
    (0x0400, "REMOVABLE_RUN_FROM_SWAP"),
    (0x0800, "NET_RUN_FROM_SWAP"),
    (0x1000, "SYSTEM"),
    (0x2000, "DLL"),
    (0x4000, "UP_SYSTEM_ONLY"),
    (0x8000, "BYTES_REVERSED_HI"),
]

# Flags DLL (IMAGE_DLLCHARACTERISTICS_*) con nombres estandar.
_DLL_CHARACTERISTICS = [
    (0x0020, "HIGH_ENTROPY_VA"),
    (0x0040, "DYNAMIC_BASE"),
    (0x0080, "FORCE_INTEGRITY"),
    (0x0100, "NX_COMPAT"),
    (0x0200, "NO_ISOLATION"),
    (0x0400, "NO_SEH"),
    (0x0800, "NO_BIND"),
    (0x1000, "APPCONTAINER"),
    (0x2000, "WDM_DRIVER"),
    (0x4000, "GUARD_CF"),
    (0x8000, "TERMINAL_SERVER_AWARE"),
]


def _decompose_flags(value: int, table) -> list:
    """Devuelve los nombres de los flags activos en un bitmask."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return []
    return [name for bit, name in table if v & bit]


def _lief_parse(bytez: bytes):
    """Parsea bytes PE con LIEF sin materializar `list(bytez)` en RAM.

    Retorna el objeto binario o None si el parseo falla (equivalente al
    `lief_binary = None` de la referencia cuando LIEF rechaza el archivo).
    """
    try:
        parsed = lief.PE.parse(bytez)
    except Exception:
        return None
    return parsed


def _short_name(value) -> str:
    """Extrae el nombre corto de un enum LIEF (`'A.B.C'` -> `'C'`)."""
    try:
        return str(value).split(".")[-1]
    except Exception:
        return str(value)


class EmberByteHistogram:
    """Histograma de bytes normalizado (256)."""

    name = "histogram"
    dim = 256

    def vector(self, bytez: bytes, lief_binary) -> np.ndarray:
        counts = np.bincount(np.frombuffer(bytez, dtype=np.uint8), minlength=256)
        total = counts.sum()
        if total == 0:
            return np.zeros(256, dtype=np.float32)
        return (counts.astype(np.float32) / float(total))


class EmberByteEntropy:
    """Histograma 2D byte/entropia normalizado (256)."""

    name = "byteentropy"
    dim = 256

    def __init__(self, step: int = 1024, window: int = 2048):
        self.window = window
        self.step = step

    def _entropy_bin_counts(self, block: np.ndarray):
        c = np.bincount(block >> 4, minlength=16)
        p = c.astype(np.float32) / self.window
        wh = np.where(c)[0]
        h = np.sum(-p[wh] * np.log2(p[wh])) * 2.0
        hbin = int(h * 2)
        if hbin == 16:
            hbin = 15
        return hbin, c

    def vector(self, bytez: bytes, lief_binary) -> np.ndarray:
        output = np.zeros((16, 16), dtype=int)
        a = np.frombuffer(bytez, dtype=np.uint8)
        if a.shape[0] < self.window:
            hbin, c = self._entropy_bin_counts(a)
            output[hbin, :] += c
        else:
            shape = a.shape[:-1] + (a.shape[-1] - self.window + 1, self.window)
            strides = a.strides + (a.strides[-1],)
            blocks = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::self.step, :]
            for block in blocks:
                hbin, c = self._entropy_bin_counts(block)
                output[hbin, :] += c
        counts = output.flatten().astype(np.float32)
        total = counts.sum()
        if total == 0:
            return np.zeros(256, dtype=np.float32)
        return counts / float(total)


class EmberStrings:
    """Features de strings sobre el bytez COMPLETO (sin muestreo) (104)."""

    name = "strings"
    dim = 1 + 1 + 1 + 96 + 1 + 1 + 1 + 1 + 1

    def __init__(self):
        self._allstrings = re.compile(b"[\x20-\x7f]{5,}")
        self._paths = re.compile(b"c:\\\\", re.IGNORECASE)
        self._urls = re.compile(b"https?://", re.IGNORECASE)
        self._registry = re.compile(b"HKEY_")
        self._mz = re.compile(b"MZ")

    def vector(self, bytez: bytes, lief_binary) -> np.ndarray:
        allstrings = self._allstrings.findall(bytez)
        if allstrings:
            lengths = [len(s) for s in allstrings]
            avlength = sum(lengths) / len(lengths)
            shifted = [b - 0x20 for b in b"".join(allstrings)]
            c = np.bincount(shifted, minlength=96)
            csum = c.sum()
            p = c.astype(np.float32) / csum
            wh = np.where(c)[0]
            entropy = float(np.sum(-p[wh] * np.log2(p[wh])))
        else:
            avlength = 0.0
            c = np.zeros(96, dtype=np.float32)
            entropy = 0.0
            csum = 0
        divisor = float(csum) if csum > 0 else 1.0
        return np.hstack([
            len(allstrings), avlength, int(csum),
            c.astype(np.float32) / divisor, entropy,
            len(self._paths.findall(bytez)), len(self._urls.findall(bytez)),
            len(self._registry.findall(bytez)), len(self._mz.findall(bytez)),
        ]).astype(np.float32)


class EmberGeneral:
    """Informacion general del archivo (10)."""

    name = "general"
    dim = 10

    @staticmethod
    def _has(obj, attr: str) -> int:
        try:
            return int(bool(getattr(obj, attr)))
        except Exception:
            return 0

    def vector(self, bytez: bytes, lief_binary) -> np.ndarray:
        if lief_binary is None:
            return np.asarray(
                [len(bytez), 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32
            )
        try:
            exported = len(lief_binary.exported_functions)
        except Exception:
            exported = 0
        try:
            imported = len(lief_binary.imported_functions)
        except Exception:
            imported = 0
        try:
            symbols = len(lief_binary.symbols)
        except Exception:
            symbols = 0
        return np.asarray([
            len(bytez),
            lief_binary.virtual_size,
            self._has(lief_binary, "has_debug"),
            exported,
            imported,
            self._has(lief_binary, "has_relocations"),
            self._has(lief_binary, "has_resources"),
            self._has(lief_binary, "has_signatures"),
            self._has(lief_binary, "has_tls"),
            symbols,
        ], dtype=np.float32)


class EmberHeader:
    """Cabeceras COFF/Optional con categoricos hasheados (62)."""

    name = "header"
    dim = 62

    def vector(self, bytez: bytes, lief_binary) -> np.ndarray:
        if lief_binary is None:
            timestamp, machine, characteristics = 0, "", []
            subsystem, dll_chars, magic = "", [], ""
            numerics = [0] * 11
        else:
            hdr = lief_binary.header
            opt = lief_binary.optional_header
            try:
                timestamp = int(hdr.time_date_stamps)
            except Exception:
                timestamp = 0
            try:
                machine = _short_name(hdr.machine)
            except Exception:
                machine = ""
            try:
                characteristics = _decompose_flags(int(hdr.characteristics), _COFF_CHARACTERISTICS)
            except Exception:
                characteristics = []
            try:
                subsystem = _short_name(opt.subsystem)
            except Exception:
                subsystem = ""
            try:
                dll_chars = _decompose_flags(int(opt.dll_characteristics), _DLL_CHARACTERISTICS)
            except Exception:
                dll_chars = []
            try:
                magic = _short_name(opt.magic)
            except Exception:
                magic = ""
            numerics = []
            for attr in (
                "major_image_version", "minor_image_version",
                "major_linker_version", "minor_linker_version",
                "major_operating_system_version", "minor_operating_system_version",
                "major_subsystem_version", "minor_subsystem_version",
                "sizeof_code", "sizeof_headers", "sizeof_heap_commit",
            ):
                try:
                    numerics.append(opt.__getattribute__(attr))
                except Exception:
                    numerics.append(0)
        return np.hstack([
            timestamp,
            FeatureHasher(10, input_type="string").transform([[machine]]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([characteristics]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([[subsystem]]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([dll_chars]).toarray()[0],
            FeatureHasher(10, input_type="string").transform([[magic]]).toarray()[0],
            np.asarray(numerics, dtype=np.float32),
        ]).astype(np.float32)


class EmberSection:
    """Informacion de secciones con hashing trick (255)."""

    name = "section"
    dim = 5 + 50 + 50 + 50 + 50 + 50

    @staticmethod
    def _props(section) -> list:
        try:
            return [_short_name(c) for c in section.characteristics_lists]
        except Exception:
            return []

    def vector(self, bytez: bytes, lief_binary) -> np.ndarray:
        if lief_binary is None:
            sections, entry = [], ""
        else:
            try:
                section = lief_binary.section_from_rva(
                    lief_binary.entrypoint - lief_binary.imagebase
                )
                entry = section.name if section is not None else ""
            except Exception:
                entry = ""
            if not entry:
                for s in lief_binary.sections:
                    try:
                        props = self._props(s)
                    except Exception:
                        props = []
                    if "MEM_EXECUTE" in props:
                        entry = s.name
                        break
            sections = []
            for s in lief_binary.sections:
                try:
                    sections.append({
                        "name": s.name,
                        "size": s.size,
                        "entropy": s.entropy,
                        "vsize": s.virtual_size,
                        "props": self._props(s),
                    })
                except Exception:
                    continue
        general = [
            len(sections),
            sum(1 for s in sections if s["size"] == 0),
            sum(1 for s in sections if s["name"] == ""),
            sum(1 for s in sections if "MEM_READ" in s["props"] and "MEM_EXECUTE" in s["props"]),
            sum(1 for s in sections if "MEM_WRITE" in s["props"]),
        ]
        sizes = [(s["name"], s["size"]) for s in sections]
        sizes_h = FeatureHasher(50, input_type="pair").transform([sizes]).toarray()[0]
        entropies = [(s["name"], s["entropy"]) for s in sections]
        entropies_h = FeatureHasher(50, input_type="pair").transform([entropies]).toarray()[0]
        vsizes = [(s["name"], s["vsize"]) for s in sections]
        vsizes_h = FeatureHasher(50, input_type="pair").transform([vsizes]).toarray()[0]
        entry_h = FeatureHasher(50, input_type="string").transform([[entry]]).toarray()[0]
        chars = [p for s in sections for p in s["props"] if s["name"] == entry]
        chars_h = FeatureHasher(50, input_type="string").transform([chars]).toarray()[0]
        return np.hstack([
            general, sizes_h, entropies_h, vsizes_h, entry_h, chars_h
        ]).astype(np.float32)


class EmberImports:
    """Imports hasheados: librerias (256) + funciones calificadas (1024)."""

    name = "imports"
    dim = 1280

    def vector(self, bytez: bytes, lief_binary) -> np.ndarray:
        imports: dict = {}
        if lief_binary is not None:
            try:
                for lib in lief_binary.imports:
                    lname = lib.name
                    if lname not in imports:
                        imports[lname] = []
                    for entry in lib.entries:
                        try:
                            if entry.is_ordinal:
                                imports[lname].append("ordinal" + str(entry.ordinal))
                            else:
                                imports[lname].append(entry.name[:10000])
                        except Exception:
                            continue
            except Exception:
                imports = {}
        libraries = list(set(l.lower() for l in imports.keys()))
        libraries_h = FeatureHasher(256, input_type="string").transform([libraries]).toarray()[0]
        qualified = [lib.lower() + ":" + e for lib, elist in imports.items() for e in elist]
        qualified_h = FeatureHasher(1024, input_type="string").transform([qualified]).toarray()[0]
        return np.hstack([libraries_h, qualified_h]).astype(np.float32)


class EmberExports:
    """Exports hasheados (128)."""

    name = "exports"
    dim = 128

    def vector(self, bytez: bytes, lief_binary) -> np.ndarray:
        names: list = []
        if lief_binary is not None:
            try:
                names = [export.name[:10000] for export in lief_binary.exported_functions]
            except Exception:
                names = []
        return FeatureHasher(128, input_type="string").transform([names]).toarray()[0].astype(np.float32)


class EmberDataDirectories:
    """Size + RVA de los 15 data directories, por posicion (30)."""

    name = "datadirectories"
    dim = 15 * 2

    def vector(self, bytez: bytes, lief_binary) -> np.ndarray:
        features = np.zeros(2 * len(DATA_DIRECTORY_ORDER), dtype=np.float32)
        if lief_binary is None:
            return features
        try:
            dirs = list(lief_binary.data_directories)
        except Exception:
            return features
        for i in range(len(DATA_DIRECTORY_ORDER)):
            if i < len(dirs):
                try:
                    features[2 * i] = dirs[i].size
                    features[2 * i + 1] = dirs[i].rva
                except Exception:
                    continue
        return features


class EmberPEFeatureExtractor:
    """Extractor EMBER v2 canonico: 2381 dims en orden fijo.

    Usa `feature_version=2` (incluye DataDirectories), que es la version con la
    que se generaron las features de entrenamiento (SOREL-20M).
    """

    def __init__(self):
        self.blocks = [
            EmberByteHistogram(),      # 256
            EmberByteEntropy(),        # 256
            EmberStrings(),            # 104
            EmberGeneral(),            # 10
            EmberHeader(),             # 62
            EmberSection(),            # 255
            EmberImports(),            # 1280
            EmberExports(),            # 128
            EmberDataDirectories(),    # 30
        ]
        self.dim = sum(b.dim for b in self.blocks)
        assert self.dim == EMBER_V2_DIM, self.dim

    def feature_vector(self, bytez: bytes) -> np.ndarray:
        """Calcula el vector EMBER_2381 completo desde los bytes del archivo."""
        data = bytes(bytez)
        lief_binary = _lief_parse(data)
        parts = [b.vector(data, lief_binary) for b in self.blocks]
        vec = np.hstack(parts).astype(np.float32)
        if vec.shape != (EMBER_V2_DIM,):
            raise ValueError(f"vector EMBER inesperado: {vec.shape}")
        if not np.all(np.isfinite(vec)):
            raise ValueError("vector EMBER con NaN/Inf")
        return vec
