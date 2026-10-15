import joblib
from dataset import get_dataset
from sklearn.metrics import accuracy_score

nn = joblib.load("models/neural_network.pkl")
_, _, X_test, _, _, y_test = get_dataset()
params = sum(w.size for w in nn.coefs_) + sum(b.size for b in nn.intercepts_)
print("hidden layers:", nn.hidden_layer_sizes)
print("parameters:", params)
print("test accuracy:", round(accuracy_score(y_test, nn.predict(X_test)) * 100, 2))