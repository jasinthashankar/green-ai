
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def get_dataset():
    digits = load_digits()
    X, y = digits.data, digits.target

    # Split: 60% training, 20% validation, 20% testing
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset()

    print("Dataset loaded successfully!")
    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Testing samples:", len(X_test))
    print("Number of features:", X_train.shape[1])
