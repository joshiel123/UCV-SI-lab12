from src.perceptron import Perceptron
from src.dataset import training_data

def test_prediction():
    model = Perceptron()
    model.train(training_data)
    assert model.predict([8,1]) == 1

def test_rejection():
    model = Perceptron()
    model.train(training_data)
    assert model.predict([1,0]) == 0
