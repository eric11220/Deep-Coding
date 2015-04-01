# dnn
import theano
import pickle
import numpy as np
from loader import Loader

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Layer:
    '''
        initialization weight and bias by normal distribution between 0, 1
    '''
    def __init__(self, in_size, out_size):
        #self.W = np.random.randn(out_size, in_size)
        #self.b = np.random.randn(out_size, 1) 
        #self.W = np.ones(shape=[out_size, in_size])/2
        #self.b = np.ones(shape=[out_size, 1])/2
        self.W = np.random.normal(0, 1, [out_size, in_size])
        self.b = np.random.normal(0, 1, [out_size, 1])

    '''
        X is a d*n matrix 
            feature dimension: d
            data number: n
    '''
    def output(self, X):    
        
        assert(X.ndim == 2)
        Z = np.dot(self.W, X) + self.b
        return sigmoid(Z)

class DNN:
    def __init__(self, layer_sizes):
        self.prev_W_update = []
        self.prev_b_update = []
        self.layers = []
        self.num = 0
        for i in range(0, len(layer_sizes)-1):
            n_input = layer_sizes[i]
            n_output = layer_sizes[i + 1]
            self.layers.append(Layer(n_input, n_output))

    def forwardPass(self, X):
        self.num = X.shape[1]

        output = [X] 
        for layer in self.layers:
            output.append(layer.output(output[-1]))

        return  output

    # TODO: Different from source code
    #       (output)*(1-output)
    def backwordPass(self, Y, output):
        deltas = []
        deltas.append((1 - output[-1])*(output[-1])*(output[-1] - Y))
        #deltas.append(output[-1] - Y)
        #deltas = [output[-1] - Y]
        for layer, output in zip(reversed(self.layers), reversed(output[:-1])):
            deltas.append(layer.W.T.dot(deltas[-1]) * output*(1-output))


        deltas = deltas[:-1]
        deltas.reverse()
        return deltas

    def update(self, outputs, deltas, momentum=0.9, learning_rate=0.01):
        W_update = []
        b_update = []
        for delta, output in zip(deltas, outputs[:-1]):
            W_update.append(delta.dot(output.T)/self.num)
        b_update = [delta.mean(axis=1).reshape(-1,1) for delta in deltas]

        if len(self.prev_W_update) > 0:
            for pdW, pdb, dW, db, layer in zip(self.prev_W_update, self.prev_b_update, W_update, b_update, self.layers):
                delta_W = momentum * pdW - learning_rate * dW
                delta_b = momentum * pdb - learning_rate * db
                layer.W += delta_W
                layer.b += delta_b
        else:
            for dW, db, layer in zip(W_update, b_update, self.layers):
                delta_W = -learning_rate * dW
                delta_b = -learning_rate * db
                layer.W += delta_W
                layer.b += delta_b

        # Store the previous weight updates
        self.prev_W_update = W_update
        self.prev_b_update = b_update

    # transfrom the derived vector to a single label
    def vectorToLabel(self, X):
        X = X.T
        ans = []
        for vector in X:
            vector = np.nan_to_num(vector)
            max_idx = np.argmax(vector)
            label = Loader().n_f_48[max_idx]
            label_for_output = Loader().map_48_39[label]
            ans.append(label_for_output)
        return ans

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(-1,1)

        for layer in self.layers:
            X = layer.output(X)
            #print X
            #print 'SHITTTTTTTTTTTTTTT'
            #raw_input()

        #print 'DONE'

        return self.vectorToLabel(X)
    
    def save(self, path='./dnn.p'):
        pickle.dump(self, open(path, 'wb'))        

    def load(self, path='./dnn.p'):
        return pickle.load(open(path, 'rb'))        

        

