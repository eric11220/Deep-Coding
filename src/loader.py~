import numpy as np
import itertools
class Loader:
    PATH = '/tmp3/mlds_hw1/MLDS_HW1_RELEASE_v1/'
    length = {'mfcc':39, 'fbank':48, 'state':1943}
    trainData = None
    batchIdx = 0
    dim = 0

    def __init__(self):
        pass
    def sanitizeForm(self, form):
        if form != 'mfcc' and form != 'fbank' and form != 'state':
            return False

    def loadFile(self, path, num, form):
        self.num = num
        self.dim = self.length[form]
        self.trainData = np.zeros(shape=(self.dim,0))

        data = {}
        with open(path, 'U') as fptr:
            idx = 0
            for line in fptr:
                name, tmpFeature = line.strip().split(' ', 1)

                idx = idx+1
                if idx > num:
                    break

                features = tmpFeature.split(' ')
                n = np.array(features)[np.newaxis]
                data[name] = np.asarray(n.T)

        return data

    def loadTest(self, form, num):
        if self.sanitizeForm(form) == False:
            return None

        dataPath = self.PATH + form + '/test.ark'
        return self.loadFile(dataPath, num, form)
        
    def loadTrain(self, form, num):
        if self.sanitizeForm(form) == False:
            return None

        dataPath = self.PATH + form + '/train.ark'
        dataDic = self.loadFile(dataPath, num, form)
        for key in dataDic.keys():
            data = dataDic[key]
            self.trainData = np.hstack((self.trainData, data))

    def loadBatch(self, size):
        if self.batchIdx > self.num:
            return False
        else:
            idx = self.batchIdx
            batch = self.trainData[0:self.dim, idx:idx+size]
            self.batchIdx = self.batchIdx+size
            return batch

