# Modelo ML — SOREL-20M y Arquitectura Neuronal

> Fuente: `models/model_manifest.json`, `models/inference.py`, `extractors/extractor.py`,
> auditoría directa de artefactos en `models/`. Actualizado 2026-09-11 al modelo v1.1.0.

---

## Dataset SOREL-20M

SOREL-20M (Sophos/ReversingLabs Open Dataset, 20 Million samples) es uno de los conjuntos de datos de malware más grandes disponibles públicamente. Contiene aproximadamente 20 millones de muestras de archivos PE con metadatos, features pre-extraídas y etiquetas de clasificación binaria (benign/malware).

El modelo de ShadowNet Defender fue entrenado sobre:
- **Dataset base**: SOREL-20M (selección 7M, seed 42; 4 187 321 malware / 2 812 679 benignos)
- **Split**: temporal 6.300.000 train / 700.000 val
- **Total**: 7.000.000 muestras de entrenamiento

> NOTA: Los datos de entrenamiento no están incluidos en el repositorio.
> Las métricas de entrenamiento son las declaradas en el manifiesto y documentación del proyecto.
> No fue posible reproducir el proceso de entrenamiento durante esta auditoría.

---

## Vector de 2381 features

El extractor produce un vector de 2381 dimensiones derivado de análisis estático del binario PE:

```
[ByteHistogram: 256] [ByteEntropy: 256] [Strings: 104]
[General: 10] [Header: 62] [Section: 255]
[Imports: 1280] [Exports: 128] [DataDirectories: 30]
Total EMBER: 256+256+104+10+62+255+1280+128+30 = 2381
+ OVERLAY_6 → entrada del modelo: 2387
```

### Justificación del diseño de features

**ByteHistogram (256)**: Captura la distribución estadística de bytes en el binario. Malware empaquetado/cifrado exhibe distribuciones más uniformes (mayor entropía), mientras que código compilado normal muestra patrones específicos (alta frecuencia de 0x00, 0xFF).

**ByteEntropy (256)**: Complementa el histograma con la entropía de Shannon calculada en ventanas deslizantes. Detecta regiones de alta entropía (código cifrado, overlays comprimidos) que el histograma global puede promediar.

**Strings (104)**: Features derivadas de strings extraídos del binario. Incluye indicadores de comportamiento: URLs, dominios, APIs Windows de riesgo, registry keys de persistencia, comandos de shell.

**Imports (1280)**: Feature hashing de la Import Address Table (IAT) con murmurhash (`FeatureHasher`: 256 librerías + 1024 funciones). Las APIs de alto riesgo (VirtualAlloc, WriteProcessMemory, CreateRemoteThread) dejan huella estadística consistente entre familias de malware.

---

## Arquitectura neuronal

```
Input layer:   2387 neuronas (2381 EMBER + 6 OVERLAY, normalizado por bloques)
Hidden 1:       512 neuronas + BatchNorm + ReLU + Dropout(p=0.3)
Hidden 2:       256 neuronas + BatchNorm + ReLU + Dropout(p=0.2)
Hidden 3:       128 neuronas + BatchNorm + ReLU + Dropout(p=0.1)
Output:           1 neurona + Sigmoid → score ∈ [0.0, 1.0]
```

**Parámetros de entrenamiento (v1.1.0, medidos en `Model_Collab/Kaggle-MLP-PE/v5/metrics/`):**
- Framework: PyTorch 2.4.1+cu121 (Tesla P100)
- Loss: `BCEWithLogitsLoss`
- Optimizer: Adam (lr=0.001)
- Scheduler: ninguno; 2 épocas, batch 8192, threshold 0.5
- Exportación: ONNX Opset 17 (sigmoid incluido; 1 388 801 parámetros)

---

## Preprocesamiento — StandardScaler

Antes de la inferencia, el vector de features se normaliza con Z-score:

```
x_norm[i] = (x[i] - μ[i]) / σ[i]
```

donde `μ` y `σ` fueron calculados sobre las 7M muestras de entrenamiento: el bloque EMBER usa `models/scaler_ember_v1.1.pkl` (57 KB) y el bloque OVERLAY_6 usa `models/scaler_overlay_v1.1.pkl` (formato joblib, nunca re-entrenar).

---

## Artefactos del modelo (verificados)

| Artefecto | Tamaño | SHA-256 (primeros 16 chars) |
|-----------|--------|-----------------------------|
| `shadow_net_sorel_7m_v1.1.onnx` | 5.6 MB | 468744956a1f3db3 |
| `scaler_ember_v1.1.pkl` | 57 KB | a20feb5b227f2ece |
| `scaler_overlay_v1.1.pkl` | 594 B | 42ab1cfd3e745f95 |

Versión del modelo: `v1.1.0`, creado `2026-09-10` (`models/model_manifest.json`).
Formato: ONNX Runtime. Umbral de producción: 0.5. Modelo anterior respaldado en `models/legacy_2381/`.

---

## Métricas declaradas en el proyecto

Las siguientes métricas corresponden a la validación temporal del modelo v1.1.0 (700k muestras, threshold 0.5; ver `Model_Collab/Kaggle-MLP-PE/v5/metrics/`):

| Métrica | Valor medido |
|---------|--------------|
| AUC-ROC | 0.9956 |
| AUC-PR | 0.9927 |
| Accuracy / F1 / Precision / Recall | 97.08% / 95.42% / 94.35% / 96.52% |
| Latencia inferencia ONNX | ~15 ms |
| Latencia total extracción + inferencia | ~400–500 ms |

---

## Hallazgo sobre el test set disponible

Durante la auditoría se identificó que `data/test_set/X_test.npy` **no es compatible** con el scaler EMBER de producción:

- `X_test.npy`: 1000 muestras, features en rango [0, 1], dtype float32
- Media post-escalado: **21.73** (esperado: ~0)
- Desviación post-escalado: **114.74** (esperado: ~1)
- AUC-ROC con scaler aplicado: **0.50** (equivalente a aleatorio)

Al aplicar el modelo directamente sobre `X_test` (sin scaler), el modelo produce AUC-ROC = 0.0 con etiquetas convencionales, pero AUC-ROC = 1.0 con etiquetas invertidas, lo que indica que el test set es **sintético y perfectamente separable**.

**Conclusión**: El test set disponible en el repositorio es un conjunto de validación sintético generado para verificar el pipeline de integración, no para medir el rendimiento estadístico real del modelo sobre datos de campo.

Las métricas reales del modelo sobre datos de producción no pueden calcularse a partir de los artefactos disponibles en el repositorio.

---

## ¿Por qué el modelo solo no fue suficiente?

El análisis de `sample1.exe` demuestra el problema central:

```
ML score:           0.0913 → label: BENIGN
Overlay ratio:      98.7% → overlay de 19.7 MB
Overlay entropy:    7.9987 → máxima aleatoriedad (cifrado/comprimido)
Global entropy:     7.9861
Risk score:         105 (CRITICAL)
Operational status: DANGEROUS
```

El modelo ML clasifica `sample1.exe` como benigno (score=0.0913 < 0.5). Sin embargo, el 98.7% del archivo es un overlay con entropía 7.9987 — indicador forense de payload cifrado o comprimido que no pertenece a la estructura PE declarada.

Este escenario corresponde a la técnica de evasión conocida como **overlay payload**: el binario PE es una cáscara pequeña y legítima (o vacía), y el contenido malicioso se almacena en datos adicionales al final del archivo que no son analizados por el parser PE estándar — y por tanto tampoco por las features del extractor.

La capa de Overlay Analysis detecta esto independientemente del modelo ML. Esta es la evidencia empírica del hallazgo principal del proyecto.

---

## Limitaciones identificadas

1. **Test set sintético**: No permite calcular FPR/FNR reales del modelo en campo.
2. **Scaler incompatible**: El scaler EMBER de producción no puede aplicarse al test set disponible.
3. **Evasión por muestreo distribuido**: Para archivos >10 MB, el extractor analiza solo el 50.2% del binario. Un atacante puede concentrar código malicioso en las regiones no muestreadas.
4. **Features de imports limitadas**: El feature hashing (módulo 1280) introduce colisiones. APIs con nombres distintos pero mismo hash son indistinguibles para el modelo.
5. **Sin reentrenamiento continuo**: El modelo v1.1.0 es estático. Nuevas familias de malware que no están en SOREL-20M podrían no ser detectadas.
6. **Sin validación sobre datos de campo reales**: No existe en el repositorio un conjunto de evaluación proveniente de análisis forense real.
