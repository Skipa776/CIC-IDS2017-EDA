"""Forward-in-time folds: train on earlier capture days, test on a later one.

Two kinds of validation:
- "day": the day right after training. Its attacks are new to the model, so it
  is the right place to choose settings meant for unseen attacks.
- "holdout": a stratified 20% of the training days' rows. Used when the test
  day repeats an attack from the last training day, so that day must be trained on.
Thresholds always come from the validation set's benign flows.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
from sklearn.model_selection import train_test_split

DAYS_2017 = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
VAL_HOLDOUT = "holdout"


@dataclass(frozen=True)
class Fold:
    name: str
    question: str
    train_days: tuple
    val_days: tuple  # (VAL_HOLDOUT,) means a 20% holdout of the training days
    test_days: tuple
    train_year: str
    test_year: str
    anomaly_only: bool = False  # training days have no attacks


def day_key(meta_source, year):
    """Capture day of a Meta_source value: weekday for 2017, ISO date for 2018."""
    if year == "2017":
        return meta_source.split("-")[0]
    _, dd, mm, yyyy = meta_source.split("-")
    return date(int(yyyy), int(mm), int(dd)).isoformat()


D18 = ["2018-02-14", "2018-02-15", "2018-02-16", "2018-02-20", "2018-02-21",
       "2018-02-22", "2018-02-23", "2018-02-28", "2018-03-01", "2018-03-02"]


def _upto(day):
    return tuple(D18[:D18.index(day) + 1])


FOLDS = [
    Fold("2017_A", "unseen family", ("Monday",), ("Tuesday",), ("Wednesday",), "2017", "2017", anomaly_only=True),
    Fold("2017_B", "unseen family", ("Monday", "Tuesday"), ("Wednesday",), ("Thursday",), "2017", "2017"),
    Fold("2017_C", "unseen family", ("Monday", "Tuesday", "Wednesday"), ("Thursday",), ("Friday",), "2017", "2017"),
    Fold("2018_new_tool_dos", "same family, new tool", _upto("2018-02-15"), (VAL_HOLDOUT,), ("2018-02-16",), "2018", "2018"),
    Fold("2018_new_tool_ddos", "same family, new tool", _upto("2018-02-20"), (VAL_HOLDOUT,), ("2018-02-21",), "2018", "2018"),
    Fold("2018_same_attack_web", "same attack, later date", _upto("2018-02-22"), (VAL_HOLDOUT,), ("2018-02-23",), "2018", "2018"),
    Fold("2018_same_attack_infiltration", "same attack, later date", _upto("2018-02-28"), (VAL_HOLDOUT,), ("2018-03-01",), "2018", "2018"),
    Fold("2018_new_family_bot", "unseen family", _upto("2018-02-28"), ("2018-03-01",), ("2018-03-02",), "2018", "2018"),
    Fold("2017_to_2018", "new network, one year later", tuple(DAYS_2017), (VAL_HOLDOUT,), tuple(D18), "2017", "2018"),
]
FOLDS_BY_NAME = {f.name: f for f in FOLDS}


def partition(fold, train_days_arr, y_train_year, seed, test_days_arr=None):
    """Row indices (train, validation, test) for a fold.

    `train_days_arr` / `y_train_year`: day keys and labels of the training year.
    `test_days_arr`: day keys of the test year when it differs (cross-year folds);
    the returned test indices then index that year's frame.
    """
    idx = np.arange(len(train_days_arr))
    train = idx[np.isin(train_days_arr, fold.train_days)]
    if fold.val_days == (VAL_HOLDOUT,):
        train, val = train_test_split(train, test_size=0.2, stratify=y_train_year[train], random_state=seed)
    else:
        val = idx[np.isin(train_days_arr, fold.val_days)]
    test_arr = train_days_arr if test_days_arr is None else test_days_arr
    test = np.flatnonzero(np.isin(test_arr, fold.test_days))
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        if not len(part):
            raise ValueError(f"{fold.name}: empty {name} partition")
    if test_days_arr is None and (np.intersect1d(train, test).size or np.intersect1d(val, test).size):
        raise AssertionError(f"{fold.name}: test rows overlap training or validation")
    return np.sort(train), np.sort(val), test
