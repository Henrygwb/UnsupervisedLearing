from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input

class GAGMM(object):
    def __init__(self, hidden_neurons_compress, hidden_neuron_estimate):
        self.compress_hidden = hidden_neurons_compress
        self.extimate_hidden = hidden_neuron_estimate
        self.compressnet = self.build_compress()
        self.extimatenet = self.build_extimate()

    def build_compress(self):
        return net

    def build_estimate(self):
        return net

    def loss(self):
        return l

    def fit(self):
        return 0

    def evaluate(self):
        return 0

    def predict(self):
        return pred