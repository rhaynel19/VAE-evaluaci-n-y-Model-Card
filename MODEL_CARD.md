# Model Card · Autoencoder y VAE

**Curso:** INF-8239 Ciencia de Datos II · Unidad 03 · Ejercicio 06
**Autor académico:** Edwin Ramón José Nolasco
**Corrida documentada:** 2026-10-10 (ver fecha UTC exacta en `reports/training_metrics.json`)
**Estado:** prototipo didáctico; no apto para decisiones o uso de producción.

## Modelos y uso previsto

Se comparan un autoencoder determinista (AE) y un autoencoder variacional (VAE)
para reconstruir y generar imágenes de Fashion-MNIST. Su uso previsto es
educativo: estudiar compresión, reconstrucción, muestreo e interpolación en un
espacio latente. No clasifican prendas ni se entrenan de forma condicionada por
clase.

**Fuera de alcance:** inferencia sobre personas, identificación, moderación de
contenido, diagnóstico, decisiones de alto impacto, compresión productiva y
afirmaciones de anonimización o privacidad.

## Datos y particiones

El script carga `tensorflow.keras.datasets.fashion_mnist`: imágenes en escala
de grises de 28×28 píxeles, normalizadas a `[0, 1]` y con un canal. En la corrida
registrada se usaron las 60,000 imágenes de entrenamiento y las 10,000 de
prueba. El AE reserva el 10% del entrenamiento para validación; Keras toma esa
fracción del final del arreglo antes de barajar los ejemplos de entrenamiento.
El VAE se ajusta con el conjunto completo de entrenamiento y no tiene una
partición de validación en este script. Las etiquetas no se usan. La semilla
configurada es 42.

Los datos se obtienen mediante el cargador de Keras y no se incluyen en el
repositorio. Esta documentación no hace una auditoría independiente de los
términos del dataset: respete los términos de su fuente. No se usa ni se
redistribuye MovieLens en esta ejecución.

## Arquitectura y entrenamiento

| Modelo | Arquitectura | Espacio latente | Parámetros entrenables |
|---|---|---:|---:|
| AE | `Flatten(28×28) → Dense(16, ReLU) → Dense(784, sigmoid) → Reshape(28×28×1)` | 16 | 25,888 |
| VAE | Encoder `Flatten → Dense(128, ReLU) → z_mean/z_log_var(2)`; decoder `Dense(128, ReLU) → Dense(784, sigmoid) → Reshape` | 2 | Encoder: 100,996; decoder: 101,520; total: 202,516 |

Ambos reciben píxeles normalizados. Se usó Adam con su tasa de aprendizaje
predeterminada y lotes de 256. El AE entrenó hasta 6 épocas con parada temprana
(`patience=2`, restaurando los mejores pesos); el VAE entrenó 8 épocas sin
parada temprana. La corrida registró Python 3.12.14, TensorFlow 2.21.0,
Windows 11 y CPU; no se detectó GPU.

## Resultados observados

Las cifras provienen de `reports/training_metrics.json`. MAE y MSE se calculan
sobre los 10,000 ejemplos de prueba, promediando píxeles. Para el VAE se
reconstruye con la media posterior, de forma determinista. La reconstrucción
BCE se suma sobre los píxeles de cada imagen; el VAE añade la divergencia KL
analítica media por imagen.

| Métrica en prueba | AE | VAE |
|---|---:|---:|
| MAE por píxel ↓ | 0.0820 | 0.1086 |
| MSE por píxel ↓ | 0.0203 | 0.0330 |
| Reconstrucción BCE sumada / imagen | 240.55 | 266.99 |
| KL / imagen | — | 6.30 |
| Reconstrucción + KL / imagen | — | 273.29 |

Las BCE del cuadro se reportan en suma por imagen; la pérdida de ajuste del AE
que muestra Keras es una media por píxel (última `loss=0.3060`,
`val_loss=0.3071`). No se deben comparar esas magnitudes directamente. La
pérdida del VAE al finalizar el ajuste fue `273.58` (reconstrucción `267.32`,
KL `6.26`); las métricas de prueba usan reconstrucción determinista con
`z=media`, no una estimación Monte Carlo del objetivo variacional.

### Muestreo, diversidad y proximidad

Se generaron 256 imágenes a partir de muestras normales estándar en el espacio
latente 2D. En el análisis se usaron 128: la MAE media entre 4,096 pares
muestreados fue **0.1567**. La MAE media al vecino más cercano fue **0.0768**,
con mínimo **0.0370**, comparando con un subconjunto reproducible de 1,000
imágenes de entrenamiento. La figura de reconstrucciones y muestras está en
`reports/ae_vae_panel.png`; la interpolación lineal fijada manualmente está en
`reports/interpolation.png`.

Son aproximaciones en espacio de píxeles: no miden diversidad semántica,
cobertura por clase ni calidad perceptual. La búsqueda de vecinos no cubre todo
el entrenamiento y no constituye una prueba de privacidad o memorización.

### Costo y huella

| Evidencia | AE | VAE |
|---|---:|---:|
| Tiempo de ajuste medido (segundos) | 14.08 | 27.78 |
| Parámetros totales | 25,888 | 202,516 |
| Huella teórica de pesos `float32` | 103,552 bytes | 810,064 bytes |
| Archivos Keras guardados | `autoencoder.keras`: 338,597 bytes | `encoder.keras`: 427,330 bytes; `decoder.keras`: 425,632 bytes |

Los tiempos miden solo el ajuste y dependen del hardware y entorno; excluyen
carga de datos, evaluación, gráficos y guardado. La huella teórica es
parámetros × 4 bytes y no es una medición del pico de RAM. El VAE se persiste
como encoder y decoder funcionales, no como un único objeto de entrenamiento
con su clase personalizada.

## Limitaciones, riesgos y decisión

- Es un MLP poco profundo sobre un único dataset pequeño de imágenes
  monocromáticas; los resultados no generalizan automáticamente a otras
  imágenes o dominios.
- Una sola semilla y una sola configuración no permiten estimar variabilidad
  estadística; no se midieron intervalos de confianza, FID, cobertura de clases
  ni utilidad downstream.
- La búsqueda de vecinos es aproximada y no acredita que los datos generados
  sean originales, anónimos o privados. Evite publicar muestras como garantía
  de privacidad.
- La corrida local obtuvo menor MAE/MSE del AE para reconstrucción de prueba.
  Por ese criterio puntual, se prefiere el AE para reconstruir; el VAE se
  conserva para el objetivo didáctico de muestreo e interpolación regularizados.
  La elección no declara superioridad universal, pues los objetivos de ambos
  modelos no son idénticos.

El uso debe permanecer supervisado y académico. Revise visualmente las salidas,
documente los errores y no interprete semejanza en píxeles como identidad o
ausencia de copia.

## Reproducibilidad

Desde la raíz del proyecto:

```bash
uv sync --extra cpu
uv run python scripts/check_runtime.py
uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8
uv run pytest -q
uv run ruff check .
```

La configuración y los resultados detallados quedan en
`reports/training_metrics.json`; las figuras se regeneran durante el
entrenamiento.
