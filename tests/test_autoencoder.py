import numpy as np

from src.models.autoencoder import Autoencoder


def test_autoencoder_scores_outliers_above_benign():
    rng = np.random.RandomState(0)
    latent = rng.normal(size=(4000, 2))
    benign = latent @ rng.normal(size=(2, 6)) * 100  # rank-2 structure the model can learn
    model = Autoencoder(seed=0, threads=1).fit(benign)
    outliers = rng.normal(size=(50, 6)) * 1000  # no rank-2 structure
    assert np.median(model.score(outliers)) > 5 * np.median(model.score(benign[:500]))
