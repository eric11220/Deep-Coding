import sys
import time
import cv2
import numpy as np
import math
import time
import loader
import structure
from dnn import DNN
from loader import Loader
EPOCH = 100
BATCH = 200
NUM_DATA = 1000

def main(argv):
    #X = np.array([[i for i in range(1,100)],[-2*i for i in range(1,100)]])
    #Y = np.array([[2*i for i in range(1,100)],[-4*i for i in range(1,100)]])
    start = time.time()

    nn = DNN([39, 128, 48])
    ld = Loader()
    ld.loadTrain('mfcc', NUM_DATA)

    print 'Done Loading data!!'
    end = time.time()
    print end - start
    raw_input()

    #X = np.array([[0, 1, 0, 1], 
    #              [0, 0, 1, 1]])
    #Y = np.array([0, 1, 1, 0])

    for n in xrange(int(EPOCH)):
        X, Y = ld.loadBatch(BATCH)
        while(len(X) != 0):
            #print X.shape
            output = nn.forwardPass(X)
            deltas = nn.backwordPass(Y, output)
            nn.update(output, deltas, 1)

            X, Y = ld.loadBatch(BATCH)

        end = time.time()
        print end - start

        ld.resetBatchIdx()

    nn.save()

    #for i in [[0, 0], [1, 0], [0, 1], [1, 1]]:
    #    print a.predict(np.array(i))

    #X_plan = np.array([[1],[-1]])
    #Y_plan = nn.predict(X_plan)
    #print Y_plan

# Start point if this script is main program
if __name__ == '__main__':
    main(sys.argv)
