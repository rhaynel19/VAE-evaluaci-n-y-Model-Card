from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera un informe PDF breve del experimento.")
    parser.add_argument(
        "--repo-url",
        default="Pendiente de publicar: el workspace aún no tiene URL de repositorio",
        help="URL pública del repositorio que se incluirá en el informe",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "Informe_U03_E06.pdf",
        help="Ruta del PDF de salida",
    )
    args = parser.parse_args()

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
    with PdfPages(args.output) as pdf:
        figure = plt.figure(figsize=(8.27, 11.69))
        figure.patch.set_facecolor("white")
        figure.text(0.08, 0.955, "INF-8239 · Unidad 03 · Ejercicio 06", fontsize=16, weight="bold")
        figure.text(0.08, 0.925, "VAE, evaluación y Model Card", fontsize=13)
        figure.text(0.08, 0.895, "Autor académico: Edwin Ramón José Nolasco", fontsize=9)
        figure.text(0.08, 0.865, f"Repositorio: {args.repo_url}", fontsize=8, wrap=True)
        figure.text(0.08, 0.83, "Diseño experimental", fontsize=11, weight="bold")
        design = (
            f"Dataset: {experiment['dataset']}; entrenamiento={experiment['train_samples']:,}, "
            f"prueba={experiment['test_samples']:,}; semilla={experiment['seed']}; "
            f"épocas AE/VAE={experiment['ae_epochs_completed']}/{experiment['vae_epochs_completed']}; "
            f"batch={experiment['batch_size']}; ejecución={experiment['run_at_utc']}."
        )
        figure.text(0.08, 0.80, fill(design, 108), fontsize=8, va="top")

        figure.text(0.08, 0.73, "Resultados en el conjunto de prueba", fontsize=11, weight="bold")
        axes = figure.add_axes([0.08, 0.575, 0.84, 0.135])
        axes.axis("off")
        table = axes.table(
            cellText=[
                [
                    "AE",
                    f"{ae['mae']:.4f}",
                    f"{ae['mse']:.4f}",
                    f"{ae['binary_crossentropy_sum_per_image']:.2f}",
                    "—",
                ],
                [
                    "VAE (media posterior)",
                    f"{vae['mae']:.4f}",
                    f"{vae['mse']:.4f}",
                    f"{vae['reconstruction_loss_sum_per_image']:.2f}",
                    f"{vae['kl_divergence_per_image']:.2f}",
                ],
            ],
            colLabels=["Modelo", "MAE ↓", "MSE ↓", "Recon. BCE/image", "KL/image"],
            cellLoc="center",
            colLoc="center",
            loc="center",
            colWidths=[0.27, 0.13, 0.13, 0.24, 0.16],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        figure.text(
            0.08,
            0.555,
            f"VAE: objetivo de prueba = reconstrucción + KL = {vae['elbo_loss_per_image']:.2f} por imagen. "
            "Las pérdidas BCE se suman sobre los píxeles; MAE/MSE son medias por píxel.",
            fontsize=7.5,
            wrap=True,
        )

        figure.text(0.08, 0.515, "Costo, capacidad y generación", fontsize=11, weight="bold")
        summary = (
            f"Tiempo de ajuste AE/VAE: {training['ae_seconds']:.2f}/{training['vae_seconds']:.2f} s "
            f"(solo fit; CPU: {', '.join(experiment['runtime']['devices'])}). "
            f"Parámetros AE/VAE completo: {models['ae_parameters']:,}/{models['vae_total_parameters']:,}; "
            f"huella teórica de pesos float32: {models['ae_float32_weights_bytes']:,}/"
            f"{models['vae_float32_weights_bytes']:,} bytes. "
            f"Muestras generadas evaluadas: {generation['generated_samples']}; "
            f"MAE media entre pares: {generation['mean_pairwise_pixel_mae']:.4f}; "
            f"MAE media al vecino del subconjunto de referencia: "
            f"{generation['mean_nearest_reference_subset_pixel_mae']:.4f}."
        )
        figure.text(0.08, 0.485, fill(summary, 108), fontsize=8, va="top")

        figure.text(0.08, 0.40, "Interpretación y límites", fontsize=11, weight="bold")
        limitations = (
            "En esta corrida, el AE obtuvo menor MAE/MSE de reconstrucción con la media posterior del VAE. "
            "No es una comparación de objetivos idénticos: el VAE incorpora KL y se evalúa usando z=media, "
            "no una estimación Monte Carlo del ELBO. La diversidad y el vecino más cercano son aproximaciones "
            "en espacio de píxeles; el vecino se busca entre 1,000 imágenes de entrenamiento, no en todo el "
            "dataset. No miden diversidad semántica ni prueban privacidad. Fashion-MNIST son imágenes grises "
            "pequeñas; no se evaluaron utilidad downstream, intervalos de confianza, ni varias semillas."
        )
        figure.text(0.08, 0.375, fill(limitations, 108), fontsize=8, va="top")

        figure.text(0.08, 0.265, "Decisión", fontsize=11, weight="bold")
        decision = (
            "Para compresión/reconstrucción en esta configuración, el AE es preferible según MAE/MSE. "
            "Para explorar generación e interpolación en un espacio latente regularizado, el VAE cumple "
            "el propósito didáctico, con un costo de ajuste mayor. Esta conclusión aplica a esta corrida "
            "y no constituye una afirmación de superioridad general."
        )
        figure.text(0.08, 0.24, fill(decision, 108), fontsize=8, va="top")

        figure.text(0.08, 0.145, "Reproducibilidad", fontsize=11, weight="bold")
        reproducibility = (
            "Desde la raíz del proyecto: uv sync --extra cpu; "
            "uv run python scripts/train_ae_vae.py --epochs-ae 6 --epochs-vae 8; "
            "uv run pytest -q; uv run ruff check ."
        )
        figure.text(0.08, 0.12, fill(reproducibility, 108), fontsize=8, va="top")
        figure.text(0.08, 0.055, "Consulte MODEL_CARD.md y AI_USE_DECLARATION.md para documentación completa.", fontsize=8)
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

    print(f"PDF generado: {args.output}")


if __name__ == "__main__":
    main()
