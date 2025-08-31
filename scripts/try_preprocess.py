from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.constants import CATEGORICAL_FEATURES, LABEL

df = pd.read_csv(Path("data/census.csv"))

train, _ = train_test_split(df, test_size=0.2, random_state=42, stratify=df[LABEL])

X, y, encoder, lb = process_data(
    train,
    categorical_features=CATEGORICAL_FEATURES,
    label=LABEL,
    training=True
)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("OHE features:", sum(len(c) for c in encoder.categories_))