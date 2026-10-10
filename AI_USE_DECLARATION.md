# Declaración de uso de inteligencia artificial

**Curso:** INF-8239 Ciencia de Datos II · Unidad 03 · Ejercicio 06
**Estudiante:** completar antes de entregar
**Estado:** borrador factual para revisión y aprobación del estudiante.

## Herramientas utilizadas

Se utilizó un asistente de IA (Copilot SDK en VS Code) como apoyo para revisar
el workspace, mejorar instrumentación de evaluación, proponer pruebas y
preparar documentación e informe académico. La ejecución de los scripts y
pruebas se realizó en el entorno local del proyecto con `uv`, Python 3.12 y
TensorFlow 2.21.

## Propósito de cada uso

- Contrastar la Model Card y la declaración de IA existentes con el código y
  los artefactos que realmente estaban presentes.
- Añadir métricas reproducibles de reconstrucción, regularización, diversidad
  aproximada, proximidad a un subconjunto de referencia, costo y tamaño de
  parámetros.
- Añadir pruebas de los contratos de cálculo, y preparar un notebook y un
  script de informe PDF que consumen los resultados guardados.
- Redactar documentación en español, manteniendo explícitos los límites de las
  medidas y la responsabilidad del estudiante.

## Prompts y decisiones relevantes

Solicitudes representativas del estudiante:

> “Revisa y verifica que los archivos `MODEL_CARD.md` y
> `AI_USE_DECLARATION.md` estén completos y alineados con los resultados reales
> obtenidos.”

> “Indícame exactamente qué comandos o scripts faltan por ejecutar [...] para
> tener el repositorio 100% listo para empaquetar y entregar.”

> “poceder para completar”

El estudiante compartió además el mandato del ejercicio y la rúbrica de cinco
criterios. Las decisiones metodológicas implementadas fueron reportar MAE/MSE
comunes para las reconstrucciones, detallar por separado reconstrucción BCE y
KL del VAE, estimar diversidad/distancia en píxeles e indicar que el vecino se
busca solo entre 1,000 imágenes de referencia. Estas medidas no se presentan
como evaluación semántica o garantía de privacidad.

## Elementos verificados y cambios sobre resultados asistidos

En el workspace se ejecutó de nuevo el entrenamiento completo con 6 épocas de
AE y 8 de VAE, semilla 42, y se guardaron historial, evaluación, entorno,
tiempos, parámetros y figuras. En la validación final local, `uv run pytest -q`
terminó con 7 pruebas aprobadas y `uv run ruff check .` con “All checks passed”.
Las cifras declaradas en `MODEL_CARD.md`, `reports/training_metrics.json` y el
notebook proceden de esa corrida, no de valores inventados ni de una
bibliografía generada por IA. El estudiante aún debe revisar y volver a ejecutar
estas verificaciones antes de entregar.

La IA propuso cambios de código, pruebas, documentación y estructura del
informe. El estudiante debe revisar esos cambios, ejecutar las celdas del
notebook y los comandos documentados, comprobar las figuras y métricas y
corregir cualquier interpretación que no pueda explicar. No se debe presentar
como verificación personal del estudiante una comprobación que este no haya
realizado.

## Responsabilidad asumida

La asistencia de IA no sustituye el criterio académico. Antes de entregar,
completar el nombre del estudiante, confirmar personalmente los comandos y
resultados, comprobar las condiciones del dataset y el enlace al repositorio,
y asegurarse de poder explicar arquitectura, pérdidas, particiones,
limitaciones y decisión final. El estudiante responde por todo el material
entregado y no debe afirmar privacidad por el solo hecho de generar datos
sintéticos.
