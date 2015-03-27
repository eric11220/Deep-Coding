# dnn
import theano
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Layer:
    def __init__(self, W_init, b_init):
        
        self.W = W_init
        self.b = b_init

    def output(self, X):    
        
        Z = np.dot(self.W, X) + self.b
        return sigmoid(Z)

class DNN:
    def __init__(self, layer_sizes):
        self.layers[]



    def output(self, x):
        pass
    def calculateError(self, x, y):
        pass









