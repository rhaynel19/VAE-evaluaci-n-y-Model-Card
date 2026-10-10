import numpy as np
import pytest

from inf8239_u03_gen.evaluation import generated_sample_metrics, reconstruction_metrics


def test_reconstruction_metrics_for_identical_images():
    images = np.full((2, 28, 28, 1), 0.5, dtype=np.float32)
    assert reconstruction_metrics(images, images) == {"mae": 0.0, "mse": 0.0}


def test_reconstruction_metrics_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="misma forma"):
        reconstruction_metrics(np.zeros((1, 28, 28, 1)), np.zeros((2, 28, 28, 1)))


def test_generated_sample_metrics_are_reproducible_and_bounded():
    generated = np.random.default_rng(1).random((4, 4, 4, 1), dtype=np.float32)
    reference = np.random.default_rng(2).random((6, 4, 4, 1), dtype=np.float32)

    first = generated_sample_metrics(
        generated, reference, seed=9, max_generated_samples=3, reference_sample_size=4
    )
    second = generated_sample_metrics(
        generated, reference, seed=9, max_generated_samples=3, reference_sample_size=4
    )

    assert first == second
    assert first["generated_samples"] == 3
    assert first["reference_samples"] == 4
    assert first["pairwise_pixel_mae_pairs"] == 3
    assert 0 <= first["mean_pairwise_pixel_mae"] <= 1
    assert 0 <= first["mean_nearest_reference_subset_pixel_mae"] <= 1
