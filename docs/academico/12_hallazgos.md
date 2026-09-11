# Hallazgos — Numerados y con Evidencia

> Solo se documentan hallazgos derivados de código real, ejecuciones reales
> y tests verificados. Fecha de auditoría: 2026-08-18.

---

## H-01 — Divergencia ML vs. Heurística en presencia de overlay payload

**Evidencia**: Ejecución real de `engine.scan_file('samples/sample1.exe')`, 2026-08-18.

```
ML score:          0.0000 → BENIGN (confianza: High)
Overlay ratio:     98.7%
Overlay entropy:   7.9987
Risk score:        105 (CRITICAL)
Operational status: DANGEROUS
```

**Descripción**: El modelo neuronal (2387 features, SOREL-20M, score 0.0913) asignó probabilidad baja de malware a un binario donde el 98.7% de su contenido (19.7 MB) es un overlay con entropía máxima (7.9987). La capa de Overlay Analysis detectó correctamente la anomalía y el Risk Engine elevó el riesgo a CRITICAL con 6 indicadores activados (`operational_status` SUSPICIOUS).

**Impacto**: Un sistema que solo usara ML hubiera producido un falso negativo. El sistema multicapa detectó la amenaza mediante una capa analítica independiente.

**Relevancia científica**: Demuestra empíricamente que las técnicas de overlay payload constituyen un vector de evasión efectivo contra ML estático sobre features PE. La solución no es mejorar el modelo ML, sino añadir capas de análisis ortogonales.

---

## H-02 — Test set sintético incompatible con el scaler de producción

**Evidencia**: Análisis directo de `data/test_set/X_test.npy` y `models/scaler_ember_v1.1.pkl`, 2026-09-11.

```python
# Post-scaling statistics (pipeline completo 2387):
mean  = 21.73  (expected: ~0)
std   = 114.74 (expected: ~1)
AUC-ROC con scaler = 0.50 (equivalente a aleatorio)
AUC-ROC sin scaler, etiquetas invertidas = 0.50 (no informativo)
```

**Descripción**: El archivo `data/test_set/X_test.npy` contiene 1000 muestras con features en rango [0, 1] que no son compatibles con el scaler EMBER de producción. Al aplicar el scaler, las features se distorsionan severamente. El test set es sintético y no informativo — diseñado para validar el pipeline de integración, no para medir rendimiento estadístico del modelo.

**Impacto**: No es posible reportar métricas de accuracy, FPR, FNR del modelo ML a partir de los artefactos disponibles en el repositorio.

**Relevancia científica**: Revela una deuda técnica importante: el proyecto carece de un conjunto de evaluación real y representativo para el modelo ML.

**Actualización 2026-09-11**: El modelo vigente es v1.1.0 (MLP 2387, 7M SOREL). Sus métricas **sobre validación temporal** (accuracy=0.9708, F1=0.9542, ROC-AUC=0.9956) existen y están documentadas en `07_metricas_y_resultados.md`. El modelo anterior v1.0.1 quedó respaldado en `models/legacy_2381/` como registro histórico.

**Actualización 2026-09-11 (scaler vigente)**: El scaler EMBER v1.1.0 se verifica por hash en `models/model_manifest.json`. La incompatibilidad con el test sintético es desajuste de dominio entre el test set [0,1] y las distribuciones crudas de SOREL-20M, no un defecto del artefacto. Ver L-05a en `13_limitaciones.md`.

---

## H-03 — Falso positivo YARA en software legítimo (Sysinternals)

**Evidencia**: Ejecución real de `engine.scan_file('samples/procexp64.exe')`, 2026-09-11.

```
Archivo:       procexp64.exe (Process Explorer, Sysinternals/Microsoft)
YARA match:    Keylogger_Generic (categoría: spyware, degradado por whitelist)
ML score:      0.0101
ML label:      BENIGN (solo-ML)
Operational:   SUSPICIOUS (riesgo HIGH)
```

**Descripción**: Process Explorer, una herramienta legítima de monitoreo del sistema de Microsoft, activa la regla YARA `Keylogger_Generic`. El modelo v1.1.0 produce score=0.0101 (el falso positivo del modelo anterior quedó corregido) y el `operational_status` es SUSPICIOUS por el match YARA degradado.

**Impacto**: La tasa de falsos positivos de las reglas YARA actuales no es despreciable. Herramientas de administración de sistemas, debuggers y profilers comparten API y patrones de comportamiento con malware de monitoreo.

**Relevancia científica**: Evidencia la necesidad de seguir ajustando reglas YARA y whitelist para reducir FPR en entornos de administración.

---

## H-04 — BehavioralShield implementado pero no integrado al pipeline

**Evidencia**: Revisión directa de `core/dynamic/process_monitor.py` y `core/engine.py`, 2026-08-18.

El código de monitoreo dinámico de procesos (psutil) existe y compila, pero no está conectado a `scan_file()`. El pipeline termina en la fase del Risk Engine sin ejecutar análisis dinámico.

**Impacto**: El sistema actual es completamente estático. No hay detección de comportamiento en tiempo de ejecución. Un malware que sea benigno en análisis estático pero malicioso en ejecución no sería detectado.

**Relevancia científica**: Define claramente el límite del sistema actual (análisis estático + heurístico) y establece la necesidad de integración de monitoreo dinámico como trabajo futuro.

---

## H-05 — n8n no alerta para el caso de detección más crítico

**Evidencia**: Revisión de `backend/app/integrations/` y tests de n8n, 2026-08-18.

```python
# n8n solo envía para:
if result == "malicious":  # label ML
    n8n_client.send()

# NO envía para:
if operational_status == "DANGEROUS" and label == "BENIGN":
    pass  # silencio — el caso más importante del sistema
```

**Descripción**: Las alertas (hoy vía webhook Supabase → Edge Function; n8n deprecated) se envían solo con `result='malicious'` u `operational_status='DANGEROUS'`. El caso de `sample1.exe` — donde ML=SUSPICIOUS pero el riesgo es CRITICAL — no generaría ninguna alerta.

**Impacto**: El hallazgo científico más importante del sistema (detección de overlay payload que el ML no detectó) no dispara la cadena de alertas automatizadas.

**Relevancia científica**: Demuestra una brecha arquitectónica entre la capa de detección heurística y la capa de respuesta automatizada. El sistema detecta pero no actúa sobre su detección más valiosa.

**Test que verifica el comportamiento actual** (pasa, confirmando el bug por diseño):
```
test_send_scan_result_skips_benign   → PASSED
test_send_scan_result_skips_suspicious → PASSED
test_send_scan_result_sends_malicious  → PASSED
```

---

## H-06 — JWT expirado produce HTTP 500 en lugar de HTTP 401

**Evidencia**: Test `TestExpiredJWT::test_expired_token_rejected` → FAILED, 2026-08-18.

```
Expected: 401 Unauthorized
Received: 500 Internal Server Error
Log: "Ni SUPABASE_URL ni SUPABASE_JWT_SECRET configurados"
```

**Descripción**: Cuando las variables de entorno de Supabase no están configuradas y se recibe un JWT expirado, el handler de autenticación lanza una excepción no capturada que produce HTTP 500. El servidor debería retornar 401 de forma controlada.

**Impacto**: Falla de seguridad menor: expone información sobre el estado de configuración del servidor y no implementa el comportamiento de fail-secure.

**Relevancia científica**: Evidencia que la gestión de errores en el flujo de autenticación no está completamente implementada.

---

## H-07 — La detección de familias de malware .NET funciona sobre datos sintéticos

**Evidencia**: Tests de IL Analyzer, 2026-08-18.

```
test_xworm_family_top        → PASSED
test_agenttesla_family_top   → PASSED
test_threat_score_critical_xworm → PASSED
```

Los tests demuestran que el sistema puede identificar familias de malware (XWorm, AgentTesla) y calcular threat scores correctamente. Sin embargo, los datos de prueba son sintéticos (strings y tokens construidos manualmente), no binarios reales.

**Relevancia científica**: El mecanismo de detección de familias está implementado y es funcionalmente correcto sobre datos sintéticos. La validación sobre binarios .NET maliciosos reales es trabajo pendiente.

---

## H-08 — Análisis en archivos >10 MB: solo 50.15% del binario es analizado

**Evidencia**: Log de ejecución real de sample1.exe, 2026-08-18.

```
modo=PE_FASTLOAD
ratio_analizado=50.2%
degradation_reason=file_size_exceeded_10mb
```

**Descripción**: Para archivos >10 MB el diagnóstico reporta muestreo distribuido (inicio, centro y fin) para evitar OOM. Para sample1.exe (20.9 MB), el diagnóstico indica 50.2% analizado; la ruta primaria EMBER, en cambio, usa los bytes completos (límite 150 MB) para el vector de features.

**Impacto**: Un atacante que distribuya features maliciosas específicamente en regiones no muestreadas podría evadir diagnósticos basados en la muestra, pero no el vector ML (bytes completos). En este caso, además, el overlay está al final del archivo — que sí es muestreado — pero la entropía del overlay se mezcla con la entropía del PE en el vector de features.

**Relevancia científica**: Define un límite de cobertura del extractor y un vector de evasión teórico para archivos grandes.
