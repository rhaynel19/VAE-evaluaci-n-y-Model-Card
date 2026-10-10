from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_URL = "https://github.com/rhaynel19/VAE-evaluaci-n-y-Model-Card"
PAGE_SIZE = (8.27, 11.69)
LEFT = 0.09
RIGHT = 0.91


def add_paragraph(figure, text: str, y: float, *, width: int = 100, size: float = 9) -> float:
    lines = wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    figure.text(LEFT, y, "\n".join(lines), fontsize=size, va="top", linespacing=1.35)
    return y - (len(lines) * size * 1.35 / 842) - 0.012


def add_heading(figure, text: str, y: float) -> float:
    figure.text(LEFT, y, text, fontsize=12, weight="bold", va="top")
    return y - 0.034


def add_page_header(figure, page: int, total: int = 3) -> None:
    figure.text(
        LEFT,
        0.965,
        "INF-8239 · Ciencia de Datos II · Unidad 03 · Ejercicio 06",
        fontsize=8,
        color="#555555",
    )
    figure.text(RIGHT, 0.035, f"Informe académico · {page}/{total}", fontsize=8, ha="right")


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el informe final PDF del ejercicio.")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="URL pública del repositorio")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "Informe_U03_E06.pdf",
        help="Ruta del PDF de salida",
    )
    args = parser.parse_args()
    if not args.repo_url.startswith(("https://", "http://")):
        parser.error("--repo-url debe ser una URL HTTP o HTTPS.")

    metrics_path = ROOT / "reports" / "training_metrics.json"
    if not metrics_path.exists():
        parser.error("No existe reports/training_metrics.json; ejecute primero el entrenamiento.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    experiment = metrics["experiment"]
    training = metrics["training"]
    evaluation = metrics["evaluation"]
    ae = evaluation["ae"]
    vae = evaluation["vae"]
    generation = evaluation["generation"]
    models = metrics["models"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output, metadata={
        "Title": "INF-8239 · Unidad 03 · Ejercicio 06 · VAE, evaluación y Model Card",
        "Author": "Informe académico basado en resultados experimentales",
        "Subject": "Comparación de autoencoder y autoencoder variacional",
    }) as pdf:
        figure = plt.figure(figsize=PAGE_SIZE)
        add_page_header(figure, 1)
        figure.text(LEFT, 0.92, "VAE, evaluación y Model Card", fontsize=20, weight="bold")
        figure.text(
            LEFT,
            0.885,
            "Estudiante: ____________________________________",
            fontsize=10,
        )
        figure.text(LEFT, 0.86, "Docente: Edwin Ramón José Nolasco · UASD · INF-8239-C2", fontsize=9)
        figure.text(LEFT, 0.835, f"Fecha: 10 de octubre de 2026 · Repositorio: {args.repo_url}",
                    fontsize=8, url=args.repo_url)

        y = 0.79
        y = add_heading(figure, "1. Propósito y alcance", y)
        y = add_paragraph(
            figure,
            "El ejercicio consolida la implementación y evaluación de un autoencoder determinista (AE) "
            "y un autoencoder variacional (VAE). Se estudian reconstrucción y, para el VAE, muestreo e "
            "interpolación en el espacio latente. El producto es una demostración académica, no un sistema "
            "para producción ni para decisiones sobre personas.",
            y,
        )
        y -= 0.008
        y = add_heading(figure, "2. Datos y diseño experimental", y)
        y = add_paragraph(
            figure,
            f"Se usó Fashion-MNIST, cargado mediante Keras, con imágenes monocromáticas de 28 × 28 "
            f"píxeles normalizadas a [0, 1]. Se utilizaron {experiment['train_samples']:,} imágenes de "
            f"entrenamiento y {experiment['test_samples']:,} de prueba; las etiquetas no se emplearon. "
            f"La semilla fue {experiment['seed']}, el lote {experiment['batch_size']} y el optimizador Adam "
            f"con su tasa predeterminada. El AE reservó {experiment['validation_fraction']:.0%} del "
            f"entrenamiento para validación y usó parada temprana; completó "
            f"{experiment['ae_epochs_completed']} épocas. El VAE completó "
            f"{experiment['vae_epochs_completed']} épocas sin partición de validación en el script. "
            f"Entorno: Python {experiment['runtime']['python']}, TensorFlow "
            f"{experiment['runtime']['tensorflow']}, {experiment['runtime']['platform']}, "
            f"dispositivo {', '.join(experiment['runtime']['devices'])}.",
            y,
        )
        y -= 0.006
        y = add_heading(figure, "3. Baselines: arquitectura y capacidad", y)
        y = add_paragraph(
            figure,
            f"AE: Flatten(784) → Dense(16, ReLU) → Dense(784, sigmoid) → reshape; "
            f"espacio latente de 16 dimensiones y {models['ae_parameters']:,} parámetros. "
            f"VAE: codificador Flatten → Dense(128, ReLU) → media y log-varianza (2 dimensiones); "
            f"decodificador Dense(128, ReLU) → Dense(784, sigmoid) → reshape. "
            f"El VAE tiene {models['vae_encoder_parameters']:,} parámetros en el codificador, "
            f"{models['vae_decoder_parameters']:,} en el decodificador y "
            f"{models['vae_total_parameters']:,} en total.",
            y,
        )
        y -= 0.006
        y = add_heading(figure, "4. Reconstrucción en el conjunto de prueba", y)
        table_axes = figure.add_axes([LEFT, y - 0.12, RIGHT - LEFT, 0.105])
        table_axes.axis("off")
        table = table_axes.table(
            cellText=[
                ["AE", f"{ae['mae']:.4f}", f"{ae['mse']:.4f}",
                 f"{ae['binary_crossentropy_sum_per_image']:.2f}", "—"],
                ["VAE (media posterior)", f"{vae['mae']:.4f}", f"{vae['mse']:.4f}",
                 f"{vae['reconstruction_loss_sum_per_image']:.2f}",
                 f"{vae['kl_divergence_per_image']:.2f}"],
            ],
            colLabels=["Modelo", "MAE ↓", "MSE ↓", "BCE suma/imagen", "KL/imagen"],
            cellLoc="center",
            colLoc="center",
            loc="center",
            colWidths=[0.30, 0.14, 0.14, 0.24, 0.14],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        figure.text(
            LEFT,
            y - 0.145,
            f"VAE: reconstrucción + KL = {vae['elbo_loss_per_image']:.2f} por imagen. "
            "MAE/MSE son promedios por píxel; BCE se suma por imagen.",
            fontsize=8,
        )
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        figure = plt.figure(figsize=PAGE_SIZE)
        add_page_header(figure, 2)
        figure.text(LEFT, 0.92, "Evaluación, costo y decisión", fontsize=19, weight="bold")
        y = 0.86
        y = add_heading(figure, "5. Reconstrucción, muestreo e interpolación", y)
        y = add_paragraph(
            figure,
            "La métrica del VAE se calculó decodificando la media posterior de forma determinista; "
            "no es una estimación Monte Carlo del ELBO. El entrenamiento del VAE optimiza reconstrucción "
            "+ divergencia KL. Por ello, los valores de pérdida del AE y del VAE no son directamente "
            "comparables como si fueran el mismo objetivo. El AE obtuvo menor MAE/MSE de reconstrucción "
            "en esta prueba. Las imágenes de originales, reconstrucciones AE/VAE y muestras generadas "
            "están en reports/ae_vae_panel.png. La interpolación lineal entre puntos fijados manualmente "
            "está en reports/interpolation.png.",
            y,
        )
        y -= 0.012
        y = add_heading(figure, "6. Fidelidad, diversidad y proximidad", y)
        y = add_paragraph(
            figure,
            f"Se generaron {generation['generated_samples']} muestras para el análisis. La MAE media "
            f"entre {generation['pairwise_pixel_mae_pairs']:,} pares muestreados fue "
            f"{generation['mean_pairwise_pixel_mae']:.4f}. La MAE media a la muestra de referencia más "
            f"cercana fue {generation['mean_nearest_reference_subset_pixel_mae']:.4f} "
            f"(mínimo {generation['min_nearest_reference_subset_pixel_mae']:.4f}), usando un subconjunto "
            f"de {generation['reference_samples']:,} imágenes de entrenamiento. Son distancias pixelarias: "
            "no miden calidad perceptual, diversidad semántica ni cobertura por clase. La búsqueda no "
            "incluye todos los datos y no prueba privacidad o memorización.",
            y,
        )
        y -= 0.012
        y = add_heading(figure, "7. Costo, memoria y almacenamiento", y)
        y = add_paragraph(
            figure,
            f"El ajuste tardó {training['ae_seconds']:.2f} s para el AE y "
            f"{training['vae_seconds']:.2f} s para el VAE, midiendo solo el ajuste. La huella teórica "
            f"float32 de pesos fue {models['ae_float32_weights_bytes']:,} bytes para el AE y "
            f"{models['vae_float32_weights_bytes']:,} bytes para el VAE. Es el número de parámetros "
            "multiplicado por cuatro; no mide el pico de RAM. Los archivos guardados ocupan "
            f"{models['ae_file_bytes']:,} bytes para el AE y "
            f"{models['vae_encoder_file_bytes'] + models['vae_decoder_file_bytes']:,} bytes para el "
            "codificador y decodificador VAE.",
            y,
        )
        y -= 0.012
        y = add_heading(figure, "8. Limitaciones y decisión final", y)
        y = add_paragraph(
            figure,
            "Se evaluó una sola semilla y una configuración; no se calcularon intervalos de confianza, "
            "FID, utilidad downstream ni pico de memoria. Fashion-MNIST contiene imágenes pequeñas y "
            "monocromáticas. Para reconstrucción en esta corrida se prefiere el AE por su MAE/MSE menor, "
            "menor cantidad de parámetros y menor tiempo de ajuste. El VAE se conserva para estudiar "
            "muestreo e interpolación con regularización latente. La decisión se limita a este experimento "
            "y no declara superioridad general.",
            y,
        )
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        figure = plt.figure(figsize=PAGE_SIZE)
        add_page_header(figure, 3)
        figure.text(LEFT, 0.92, "Uso responsable y reproducibilidad", fontsize=19, weight="bold")
        y = 0.86
        y = add_heading(figure, "9. Uso responsable de inteligencia artificial", y)
        y = add_paragraph(
            figure,
            "Se utilizó Copilot como apoyo para revisar el proyecto, mejorar la persistencia de métricas, "
            "proponer pruebas y preparar documentación, notebook e informe. El estudiante debe revisar "
            "el material, entender y poder defender el código y las conclusiones, y responde por lo "
            "entregado. La generación sintética no garantiza privacidad ni anonimato. No se deben inventar "
            "fuentes ni redistribuir MovieLens.",
            y,
        )
        y -= 0.012
        y = add_heading(figure, "10. Pruebas y reproducción", y)
        y = add_paragraph(
            figure,
            "La suite local finalizó con 7 pruebas aprobadas y Ruff sin errores. El notebook de "
            "evaluación fue ejecutado: 4 celdas de código, sin errores. El paquete contiene src, tests, "
            "reports, Model Card, declaración de IA, README, requirements.txt y notebook.",
            y,
        )
        y -= 0.01
        commands = (
            "uv sync --extra cpu\n"
            "uv run python scripts/check_runtime.py\n"
            "uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8\n"
            "uv run pytest -q\n"
            "uv run ruff check ."
        )
        figure.text(
            LEFT + 0.02,
            y,
            commands,
            fontsize=9,
            va="top",
            family="monospace",
            linespacing=1.5,
        )
        y -= 0.14
        y = add_heading(figure, "Referencias del proyecto", y)
        y = add_paragraph(
            figure,
            f"Repositorio: {args.repo_url}. Evidencia experimental: "
            "reports/training_metrics.json, reports/ae_vae_panel.png e "
            "reports/interpolation.png. Consulte MODEL_CARD.md y "
            "AI_USE_DECLARATION.md para detalles ampliados.",
            y,
        )
        figure.text(
            LEFT,
            max(y - 0.025, 0.07),
            "Estudiante: completar el nombre y confirmar personalmente el contenido antes de entregar.",
            fontsize=9,
            weight="bold",
        )
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

    print(f"PDF generado: {args.output}")


if __name__ == "__main__":
    main()
