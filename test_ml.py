# test_ml.py

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics


# Shared settings
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"
CSV_PATH = Path("data") / "census.csv"


def _load_small_df(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Load a small deterministic sample to keep tests fast."""
    df = pd.read_csv(CSV_PATH)
    if len(df) > n:
        df = df.sample(n=n, random_state=seed)
    return df


def test_train_and_infer_types_and_shapes():
    """
    Train the model and ensure:
      - it is a RandomForestClassifier,
      - predictions are a numpy array of 0/1 with correct length.
    """
    df = _load_small_df()
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[LABEL]
    )

    X_train, y_train, enc, lb = process_data(
        train, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label=LABEL,
        training=False, encoder=enc, lb=lb
    )

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)

    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == y_test.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_perfect_predictions():
    """
    With perfect predictions, precision/recall/F1 should be 1.0.
    """
    y_true = np.array([0, 1, 1, 0, 1, 0])
    preds = np.array([0, 1, 1, 0, 1, 0])

    p, r, f1 = compute_model_metrics(y_true, preds)

    assert np.isclose(p, 1.0)
    assert np.isclose(r, 1.0)
    assert np.isclose(f1, 1.0)


def test_process_data_consistent_feature_count():
    """
    Encoded feature count must match between training and inference transforms.
    """
    df = _load_small_df()
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[LABEL]
    )

    X_train, y_train, enc, lb = process_data(
        train, categorical_features=CAT_FEATURES, label=LABEL, training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label=LABEL,
        training=False, encoder=enc, lb=lb
    )

    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.ndim == 2 and X_test.ndim == 2
    # Sanity: both y arrays are 1-D
    assert y_train.ndim == 1 and y_test.ndim == 1
