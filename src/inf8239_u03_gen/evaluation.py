from __future__ import annotations

import numpy as np


def reconstruction_metrics(
    originals: np.ndarray, reconstructions: np.ndarray
) -> dict[str, float]:
    if originals.shape != reconstructions.shape:
        raise ValueError("Originales y reconstrucciones deben tener la misma forma")
    if originals.size == 0 or not np.isfinite(originals).all() or not np.isfinite(reconstructions).all():
        raise ValueError("Las imágenes deben ser no vacías y contener valores finitos")
    difference = originals - reconstructions
    return {
        "mae": float(np.mean(np.abs(difference))),
        "mse": float(np.mean(np.square(difference))),
    }


def generated_sample_metrics(
    generated: np.ndarray,
    reference_images: np.ndarray,
    *,
    seed: int = 42,
    max_generated_samples: int = 128,
    reference_sample_size: int = 1_000,
    pair_sample_size: int = 4_096,
) -> dict[str, float | int]:
    if generated.ndim != 4 or reference_images.ndim != 4:
        raise ValueError("Las imágenes deben tener cuatro dimensiones (N, alto, ancho, canales)")
    if generated.shape[1:] != reference_images.shape[1:]:
        raise ValueError("Las muestras generadas y de referencia deben tener la misma forma")
    if len(generated) < 2 or len(reference_images) == 0:
        raise ValueError("Se requieren al menos dos imágenes generadas y una de referencia")
    if (
        not np.isfinite(generated).all()
        or not np.isfinite(reference_images).all()
        or generated.min() < 0
        or generated.max() > 1
        or reference_images.min() < 0
        or reference_images.max() > 1
    ):
        raise ValueError("Las imágenes deben contener valores finitos entre 0 y 1")

    rng = np.random.default_rng(seed)
    evaluated = generated[:max_generated_samples]
    left, right = np.triu_indices(len(evaluated), k=1)
    pair_count = min(pair_sample_size, len(left))
    selected_pairs = rng.choice(len(left), size=pair_count, replace=False)
    pairwise_mae = np.mean(
        np.abs(evaluated[left[selected_pairs]] - evaluated[right[selected_pairs]]),
        axis=(1, 2, 3),
    )

    reference_indices = rng.choice(
        len(reference_images),
        size=min(reference_sample_size, len(reference_images)),
        replace=False,
    )
    reference_subset = reference_images[reference_indices]
    nearest_mae = []
    for image in evaluated:
        distances = np.mean(np.abs(reference_subset - image), axis=(1, 2, 3))
        nearest_mae.append(float(np.min(distances)))

    return {
        "generated_samples": len(evaluated),
        "reference_samples": len(reference_subset),
        "pairwise_pixel_mae_pairs": pair_count,
        "mean_pairwise_pixel_mae": float(np.mean(pairwise_mae)),
        "mean_nearest_reference_subset_pixel_mae": float(np.mean(nearest_mae)),
        "min_nearest_reference_subset_pixel_mae": float(np.min(nearest_mae)),
    }
