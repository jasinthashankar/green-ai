"""
Wine Dataset Pipeline for Green AI Generalization Experiments.

Uses scikit-learn's built-in Wine classification dataset:
- 178 samples, 13 continuous chemical features, 3 cultivar classes.
- Reference: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html

Strict Data Hygiene:
- Stratified 60% Train, 20% Validation, 20% Test partition.
- Feature scaling (StandardScaler) is strictly fitted on X_train only,
  preventing data leakage into validation and test sets.
"""

from typing import Tuple, List
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_wine_dataset(
    scale: bool = True,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str], List[str]]:
    """
    Loads and partitions the Wine dataset with strict data hygiene.

    Parameters
    ----------
    scale : bool, default=True
        Whether to standardize features using StandardScaler fitted on X_train.
    random_state : int, default=42
        Reproducible random seed for train_test_split.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, target_names
    """
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = list(wine.feature_names)
    target_names = list(wine.target_names)

    # 1. Stratified split: 60% train, 40% temp (val + test)
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=random_state, stratify=y
    )

    # 2. Stratified split of temp: 50% val (20% total), 50% test (20% total)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    # 3. Fit scaler strictly on training data
    scaler = StandardScaler()
    if scale:
        scaler.fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)
    else:
        X_train, X_val, X_test = X_train_raw, X_val_raw, X_test_raw

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names, target_names


if __name__ == "__main__":
    X_tr, X_v, X_te, y_tr, y_v, y_te, sc, fnames, tnames = get_wine_dataset()
    print("Wine Dataset loaded successfully:")
    print(f"  Training samples:   {len(X_tr)} (shape: {X_tr.shape})")
    print(f"  Validation samples: {len(X_v)} (shape: {X_v.shape})")
    print(f"  Testing samples:    {len(X_te)} (shape: {X_te.shape})")
    print(f"  Total samples:      {len(X_tr) + len(X_v) + len(X_te)}")
    print(f"  Features ({len(fnames)}): {', '.join(fnames[:4])}...")
    print(f"  Classes ({len(tnames)}):  {tnames}")
