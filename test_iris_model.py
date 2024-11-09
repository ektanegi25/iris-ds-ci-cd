import pytest
from iris_model import load_data, train_model

def test_load_data():
    X, y = load_data()
    assert X.shape[0] > 0, "Dataset should not be empty."
    assert y.shape[0] > 0, "Target variable should not be empty."

def test_train_model():
    X, y = load_data()
    model, accuracy = train_model(X, y)
    assert accuracy > 0.9, "Model accuracy is too low!"

if __name__ == "__main__":
    pytest.main()
