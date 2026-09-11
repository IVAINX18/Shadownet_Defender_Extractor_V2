# Métricas y Resultados — Estado Actual

> Auditoría ejecutada 2026-08-18. Solo se documentan métricas calculadas
> sobre datos reales disponibles en el repositorio. Los datos faltantes
> se indican explícitamente.

---

## Estado de los datos disponibles

| Dataset | Disponible | Válido para métricas | Observación |
|---------|------------|----------------------|-------------|
| `data/test_set/X_test.npy` (1000 muestras) | ✅ | ❌ | Sintético, incompatible con scaler de producción |
| `samples/` (archivos PE) | ✅ | Parcial | Sin ground truth externo verificado |
| SOREL-20M (datos de entrenamiento) | ❌ | — | No incluido en repositorio (selección 7M seed 42 documentada en `Model_Collab/Kaggle-MLP-PE/v5/`; vectores en el npz público de SOREL-20M) |
| Datos de campo reales | ❌ | — | No disponibles |

---

## Métricas del modelo ML — CALCULADAS

### Sobre el test set disponible (diagnóstico)

**Condición**: `data/test_set/X_test.npy`, 1000 muestras (500 benign, 500 malware)

El test set contiene features en rango [0, 1]. El scaler EMBER de producción fue ajustado sobre datos con distribuciones radicalmente diferentes:
- Media post-escalado: 21.73 (esperado: ~0)
- Desviación post-escalado: 114.74 (esperado: ~1)

Al aplicar el pipeline de producción sobre el test set, el modelo produce score=0.0000 para el 100% de las muestras, con AUC=0.50 tanto con etiquetas convencionales como invertidas. Esto confirma que el test set es **sintético y no informativo** — no representativo de datos reales.

**Conclusión**: Las métricas de ML sobre `data/test_set/` no son válidas para reportar en un artículo científico.

---

## Métricas de entrenamiento — modelo v1.1.0 vigente (Fase 5 v5)

> **Fuente**: `Model_Collab/Kaggle-MLP-PE/v5/metrics/` (métricas y metadata generadas
> por el entrenamiento; artefactos desplegados en `models/` y verificados en
> `models/model_manifest.json`). Estas métricas **corresponden al modelo en producción**.

### Configuración del entrenamiento

| Parámetro | Valor |
|-----------|-------|
| Dataset | SOREL-20M, selección 7M seed 42 (4 187 321 malware / 2 812 679 benignos) |
| Características | 2,387 (`ShadowNetFeatures_v1.1` = EMBER_2381 + OVERLAY_6) |
| División | Temporal 90/10 (Train 6,300,000 · Val 700,000) |
| Arquitectura | MLP 2387 → 512 → 256 → 128 → 1 (BatchNorm + ReLU + Dropout 0.3/0.2/0.1) |
| Parámetros | 1,388,801 |
| Pérdida / Optimizador | `BCEWithLogitsLoss` · Adam (lr=0.001) |
| Scheduler | Ninguno; 2 épocas, batch 8192 |
| Dispositivo | GPU Tesla P100 (torch 2.4.1+cu121) |

### Métricas sobre validación temporal (700,000 muestras, threshold 0.5)

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.9708 (97.08%) |
| Precision | 0.9435 (94.35%) |
| Recall (TPR) | 0.9652 (96.52%) |
| F1-Score | 0.9542 (95.42%) |
| ROC-AUC | 0.9956 |
| PR-AUC | 0.9927 |

**Matriz de confusión**:

| | Pred. Malware | Pred. Benigno |
|---|---|---|
| **Real Malware** | TP = 213,019 | FN = 7,690 |
| **Real Benigno** | FP = 12,766 | TN = 466,525 |

### Alcance y validez de estas métricas

1. **Distribución evaluada**: validación temporal (10% más reciente del split TRAIN). No mide generalización a dominios nuevos ni sustituye un corpus de campo (ver experimentos faltantes).
2. **Sin padding artificial**: a diferencia del modelo anterior, no hay muestras rellenadas con ceros.
3. **Convergencia sana**: `train_loss` 0.1105 → 0.0767, `val_loss` 0.0918 → 0.0864, sin sobreajuste.

> **Modelo anterior v1.0.1 (histórico, respaldado en `models/legacy_2381/`)**: MLP 2381, 1,385,729 parámetros, entrenado sobre 5.1M (5M SOREL + 100K con padding 33→2381), `BCELoss`, `ReduceLROnPlateau`, early stopping en época 14. Métricas recuperadas de su notebook: accuracy 0.9815, F1 0.9845 sobre su test híbrido de 765k. Se conservan como registro, no como métricas vigentes.

---

## Métricas declaradas en el proyecto (medidas en validación v1.1.0)

Las siguientes métricas están respaldadas por los artefactos de validación en `Model_Collab/Kaggle-MLP-PE/v5/metrics/`:

| Métrica | Valor medido | Fuente |
|---------|--------------|--------|
| AUC-ROC | 0.9956 | Validación temporal 700k |
| AUC-PR | 0.9927 | Validación temporal 700k |
| FPR @ TPR=90% | No especificado | — |
| TPR @ FPR=1% | No especificado | — |
| Latencia inferencia ONNX | ~15 ms | Documentación del proyecto |
| Latencia total E2E | ~400–500 ms | Documentación del proyecto |

> **IMPORTANTE**: Los puntos operativos (FPR@TPR, TPR@FPR) requieren corpus de campo y no se reportan valores no medidos.

---

## Métricas del sistema multicapa — CALCULADAS sobre samples reales

Ejecuciones reales del pipeline completo con el modelo v1.1.0 (2026-09-11):

### Tabla de resultados por archivo

| Archivo | Tamaño | ML score | ML label | Operational Status | Risk Level | YARA | Tiempo |
|---------|--------|----------|----------|--------------------|------------|------|--------|
| `sample1.exe` | 20.9 MB | 0.0913 | SUSPICIOUS | SUSPICIOUS | CRITICAL | 0 matches | 2,026 ms |
| `sample2.exe` | ~2 MB | 0.4660 | SUSPICIOUS | SUSPICIOUS | MEDIUM | 0 matches | 402 ms |
| `eicar.txt` | ~68 B | 0.0040 | BENIGN | CLEAN | LOW | 0 matches | 283 ms |
| `procexp64.exe` | ~2 MB | 0.0101 | SUSPICIOUS | SUSPICIOUS | HIGH | 1 match | 475 ms |

### Observaciones sobre los resultados

**sample1.exe**: Divergencia ML vs. Heurística. ML=SUSPICIOUS (0.0913), Heurística=CRITICAL. Detectado por Overlay Analysis (98.7% overlay, entropía 7.9987). El modelo ML aislado no marca la amenaza; la arquitectura multicapa sí.

**sample2.exe**: Binario .NET. ML=SUSPICIOUS (0.4660, contribuye como `ml_onnx` en la correlación). DotNet Analysis detectó ofuscación (Unknown Obfuscator, dotnet_risk_score=28 MEDIUM). Resultado final: SUSPICIOUS/MEDIUM.

**eicar.txt**: Archivo de prueba EICAR estándar (texto, no PE). El extractor usó RAW_FALLBACK. El modelo produjo score=0.0040. YARA no activó. El sistema no detectó EICAR — esto es una limitación documentada: las reglas YARA no incluyen la firma EICAR estándar.

**procexp64.exe**: Herramienta legítima (Sysinternals Process Explorer). YARA activó `Keylogger_Generic` (degradado a SUSPICIOUS por whitelist) y el ML produjo score=0.0101. El falso positivo del modelo anterior (score 1.0) quedó corregido.

---

## Métricas del sistema de tests — CALCULADAS

Ejecución real: `.venv/bin/pytest tests/ -q` (2026-09-11)

| Total | Passed | Failed | Skipped |
|-------|--------|--------|---------|
| **321** | **294** | **0** | **27** |

Incluye `tests/test_ember_feature_alignment.py` (7 tests del contrato EMBER v2 + OVERLAY_6). Los skipped dependen de datos externos o componentes opcionales.

---

## Métricas de rendimiento — MEDIDAS en ejecuciones reales

| Operación | Medición real (2026-09-11) |
|-----------|----------------------------|
| Extracción PE_FASTLOAD (sample1.exe, 20.9 MB) | 1,501 ms |
| Extracción PE normal (sample2.exe, ~2 MB) | 186 ms |
| Extracción RAW_FALLBACK (eicar.txt, 68 B) | 121 ms |
| Pipeline completo sample1.exe | 2,026 ms |
| Pipeline completo sample2.exe (con IL) | 402 ms |
| Pipeline procexp64.exe (YARA match + whitelist) | 475 ms |
| Inferencia ONNX (medición directa) | ~15 ms (según documentación) |

---

## Experimentos faltantes para métricas completas

Los siguientes experimentos son necesarios para completar el cuadro de métricas y no están disponibles actualmente:

1. **Corpus de evaluación real**: mínimo 1000 muestras de malware real + 1000 benignos con ground truth externo (VirusTotal, sandbox).

2. **Métricas de FPR/FNR del sistema multicapa**: comparar tasa de detección del sistema completo vs. solo-ML sobre el mismo corpus.

3. **Evaluación de la capa YARA**: calcular FPR de las reglas YARA actuales sobre un corpus de software legítimo conocido. El falso positivo de `procexp64.exe` indica que la tasa puede ser no despreciable.

4. **Latencia en percentil 95**: medir tiempos de ejecución sobre N≥100 archivos para reportar distribución, no solo casos individuales.

5. **Evaluación de IL Behavioral sobre malware .NET real**: los tests actuales usan datos sintéticos. Falta evaluación sobre ensamblados .NET maliciosos reales (AgentTesla, XWorm, AsyncRAT).

---

## Métricas de campo (T-13)

> **PENDIENTE**: Requiere CorpusReal (≥1000 malware VT≥5 + ≥1000 benignos VT=0).
> Para adquirir el corpus: `python tools/fetch_corpus.py --help`
> Script de evaluación: `python evaluation/evaluate_real_corpus.py --corpus data/eval_real/`

### Proceso de adquisición reproducible

- **Malware**: VirusTotal API (`positives ≥5`, `type=peexe`, `size<10MB`, `first_seen ≥2024`)
- **Benignos**: `C:\Windows\System32`, Sysinternals, fresh installs (`positives == 0`)
- **Balance**: 1000/1000 mínimo; 2000 ideal
- **Legal**: Solo `manifest.csv` commiteado (hashes + metadatos). Binarios en `.gitignore`.
  - Cualquier revisor puede reproducir con: `python tools/fetch_corpus.py --manifest data/eval_real/manifest.csv --vt-api-key <KEY>`

### Tabla de métricas de campo (pendiente)

| Sistema | Accuracy | F1 | AUC-ROC | AUC-PR | FPR@TPR=90% | TPR@FPR=1% | N |
|---|---|---|---|---|---|---|---|
| solo-ML (ONNX) | — | — | — | — | — | — | ≥2000 |
| híbrido | — | — | — | — | — | — | ≥2000 |
| delta | — | — | — | — | — | — | — |

> Tabla a completar tras ejecutar `evaluation/evaluate_real_corpus.py`.
> Metodología: StratifiedKFold k=5, seed=42, MLflow tracking, Bonferroni α/3.

### Scaler drift (T-13)

| Campo | Corpus Real (esperado) | Sintético (medido 2026-09-11) |
|---|---|---|
| mean_post | \|mean\| < 2.0 | **21.73** |
| std_post | 0.5 < std < 2.0 | **114.74** |

> Nota histórica: el modelo anterior usaba 100K muestras rellenadas con 2,348 ceros
> (33→2381) como posible atajo discriminativo (ver L-01 en `13_limitaciones.md`).
> El modelo v1.1.0 no usa padding: entrena sobre 7M muestras SOREL puras.

> ⚠️  `data/test_set/X_test.npy` — **SINTETICO — no usar para métricas de campo**.
> `docs/academico/figures/fig7_roc_pr_SINTETICO.png` — advertencia: métricas sintéticas.

---

## Modelo v5 (7M) y alineación del extractor — MEDIDO 2026-09-11

En este experimento entreno el MLP `ShadowNetFeatures_v1.1 = EMBER_2381 + OVERLAY_6 = 2387`
sobre 7,000,000 muestras SOREL (seed 42, split temporal 6.3M/0.7M, 2 épocas, Tesla P100).
Obtengo, sobre validación temporal 700k (threshold 0.5, sin tuning):

| Época | train_loss | val_loss | val_acc | val F1 | val ROC-AUC | val PR-AUC |
|---|---|---|---|---|---|---|
| 1 | 0.1105 | 0.0918 | 96.97% | 0.9520 | 0.9941 | 0.9899 |
| **2 (best=final)** | **0.0767** | **0.0864** | **97.08%** | **0.9542** | **0.9956** | **0.9927** |

Final: **Accuracy 97.08% · Precision 94.35% · Recall 96.52% · F1 95.42%** ·
TN 466,525 / FP 12,766 / FN 7,690 / TP 213,019. Fuente: `Model_Collab/Kaggle-MLP-PE/v5/metrics/`.

Al integrar el modelo compruebo que el extractor local pefile NO reproduce el layout
EMBER: tras el escalado SOREL, ~30 columnas del bloque Header explotan a `|z| ~ 4.3e11`
(p. ej. índice 653 = ImageBase local `0x140000000` donde SOREL espera un flag `has_*`
con escala 0.0125) y el MLP se satura a `score = 0.0` degenerado. El modelo antiguo
sufre lo mismo con el extractor actual, así que es un defecto preexistente del
extractor, no del reentrenamiento.

Implemento `extractors/ember_features.py` (adaptación original del layout público EMBER v2:
histogramas normalizados, `FeatureHasher`, `DataDirectories(30)`, parser LIEF) y lo conecto
como ruta primaria del extractor, conservando `RAW_FALLBACK` como contingencia. Tras el fix,
mido en el corpus local:

| Archivo | max\|z\| | ML score (nuevo) | ML score (antes) |
|---|---|---|---|
| `sample1.exe` | 48.31 | 0.0913 | 0.0 (saturado) |
| `sample2.exe` | 48.31 | 0.4660 | 0.0 (saturado) |
| `procexp64.exe` | 48.31 | 0.0101 | 0.0 (saturado) |

Verifico el cross-check contra vectores EMBER almacenados por SOREL
(`scripts/compare_ember_vs_sorel.py`, N=5 binarios): `strings/general/imports/exports/
datadirectories` exactos (0 diffs); histogramas solo redondeo float (max ~2e-5);
`header` 5/62 y `section` 5-7/255 con diff máxima 2.0 en dims hasheadas, atribuida a
nombres de enums LIEF 0.9 vs 1.0 y a resolución de entry (RVA vs offset). Sin explosiones,
sin cambios de orden, sin bloques faltantes.

Limitación conocida: los `|z|>10` restantes (máx. 48.31, <0.4% de dims) son buckets
hasheados sparse legítimos (±1 en buckets que SOREL casi no activa), no un mismatch de
unidades; el modelo los tolera sin saturarse. `tests/test_ember_feature_alignment.py`
fija este contrato (max|z| < 100, <1% dims con |z|>10, sin recortes).

