# INF-8239 · Unidad 03 · Modelos generativos

Autor académico: Edwin Ramón José Nolasco

## Experimento reproducible

El entrenamiento usa Fashion-MNIST descargado por Keras: 60,000 imágenes para
entrenamiento y 10,000 para prueba; el 10% del conjunto de entrenamiento se
reserva para validación del AE. Las etiquetas no se usan para entrenar. La
semilla es 42 y las entradas se normalizan a `[0, 1]`.

## CPU
```bash
uv python install 3.12
uv sync --extra cpu
uv run python scripts/check_runtime.py
uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8
uv run pytest -q
uv run ruff check .
```

El entrenamiento genera los modelos locales en `models/` y las evidencias
`reports/training_metrics.json`, `reports/ae_vae_panel.png` y
`reports/interpolation.png`. El JSON conserva el historial por época, métricas
de reconstrucción en prueba, componentes de pérdida del VAE, una estimación
pixelaria de diversidad/proximidad y conteos de parámetros y tamaños de
artefactos. Las medidas de píxeles no son métricas de calidad semántica ni
garantizan privacidad. Revise también las limitaciones descritas en
`MODEL_CARD.md`.

## GPU
Solo para WSL2/Linux con NVIDIA correctamente configurada:
```bash
uv sync --extra gpu
```

## Instalar con pip

`requirements.txt` contiene las dependencias bloqueadas para CPU y desarrollo:
```bash
python -m pip install -r requirements.txt
```

## Aplicación
```bash
uv run streamlit run app/streamlit_app.py
```

Entrene primero para generar `models/decoder.keras`. Para completar la entrega,
ejecute las celdas del notebook en `notebooks/` y genere el informe breve con:
```bash
uv run python scripts/create_submission_pdf.py --repo-url <URL-del-repositorio>
```

Complete o revise `MODEL_CARD.md` y `AI_USE_DECLARATION.md` antes de entregar.
No afirme privacidad por el solo hecho de generar datos sintéticos y no
redistribuya MovieLens.
