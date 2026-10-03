import numpy as np

from src.models.cross_dataset import natural_mix


def test_natural_mix_undoes_2018_benign_sampling():
    # 2018 keeps every attack but only 10% of benign, so the natural mix keeps
    # all benign rows and about 10% of attack rows. 2017 is already natural.
    y = np.array([0] * 1000 + [1] * 1000)

    idx = natural_mix(y, "2018")
    assert (y[idx] == 0).sum() == 1000
    assert 50 < (y[idx] == 1).sum() < 150

    assert len(natural_mix(y, "2017")) == len(y)
