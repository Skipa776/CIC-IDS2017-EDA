import numpy as np

from scripts.run_pipeline import hybrid_alerts


def test_full_share_hybrid_is_the_supervised_model():
    rng = np.random.RandomState(0)
    val = {"lgbm": rng.rand(1000), "autoencoder": rng.rand(1000)}
    test = {"lgbm": rng.rand(500), "autoencoder": rng.rand(500) * 10}  # anomaly scores beyond validation
    benign = np.ones(1000, dtype=bool)
    alone = hybrid_alerts(val, test, benign, "lgbm", "autoencoder", 1.0)
    for b, alerts in alone.items():
        from src.models.evaluate import threshold_on_benign
        assert (alerts == (test["lgbm"] >= threshold_on_benign(val["lgbm"], float(b)))).all()
    assert not hybrid_alerts(val, test, benign, "lgbm", "autoencoder", 0.0)["0.01"][test["autoencoder"] < 0.5].any()
