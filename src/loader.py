import numpy as np
import itertools
class Loader:

    def __init__(self):
        self.length = {'mfcc':39, 'fbank':48, 'state':1943}  # input lengths 
        self.fbank = {'aa':0,'el':1,'ch':2,'ae':3,'eh':4,'cl':5,'ah':6,'ao':7,'ih':8,'en':9,'ey':10,'aw':11,'ay':12,'ax':13,'er':14,'vcl':15,'ng':16,'iy':17,'sh':18,'th':19,'sil':20,'zh':21,'w':22,'dh':23,'v':24,'ix':25,'y':26,'hh':27,'jh':28,'dx':29,'b':30,'d':31,'g':32,'f':33,'k':34,'m':35,'l':36,'n':37,'uh':38,'p':39,'s':40,'r':41,'t':42,'oy':43,'epi':44,'ow':45,'z':46,'uw':47}   # for constructing ground truth vector based on given label
        self.map_48_39 = {'aa':'aa','ae':'ae','ah':'ah','ao':'aa','aw':'aw','ax':'ah','ay':'ay','b':'b','ch':'ch','cl':'sil','d':'d','dh':'dh','dx':'dx','eh':'eh','el':'l','en':'n','epi':'sil','er':'er','ey':'ey','f':'f','g':'g','hh':'hh','ih':'ih','ix':'ih','iy':'iy','jh':'jh','k':'k','l':'l','m':'m','ng':'ng','n':'n','ow':'ow','oy':'oy','p':'p','r':'r','sh':'sh','sil':'sil','s':'s','th':'th','t':'t','uh':'uh','uw':'uw','vcl':'sil','v':'v','w':'w','y':'y','zh':'sh','z':'z'}
        self.n_f_48 = {0:'aa',1:'el',2:'ch',3:'ae',4:'eh',5:'cl',6:'ah',7:'ao',8:'ih',9:'en',10:'ey',11:'aw',12:'ay',13:'ax',14:'er',15:'vcl',16:'ng',17:'iy',18:'sh',19:'th',20:'sil',21:'zh',22:'w',23:'dh',24:'v',25:'ix',26:'y',27:'hh',28:'jh',29:'dx',30:'b',31:'d',32:'g',33:'f',34:'k',35:'m',36:'l',37:'n',38:'uh',39:'p',40:'s',41:'r',42:'t',43:'oy',44:'epi',45:'ow',46:'z',47:'uw'}

        self.PATH       = '/tmp3/mlds_hw1/MLDS_HW1_RELEASE_v1/' # "root" directory
        self.LBL_PATH   = self.PATH + 'label/train.lab'         # path of ground truth file

        self.trainData          = None
        self.ground_truth_vecs  = None 
        self.batchIdx           = 0
        self.in_dim             = 0
        self.out_dim            = 0
        self.FBANK_DIM          = 48
        self.form               = 'mfcc'

    '''
        Check if the given form is of the 3 legal forms 
    '''
    def sanitizeForm(self, form):
        if form != 'mfcc' and form != 'fbank' and form != 'state':
            print "Format not recognized!\nUsing default format: mfcc..."
            raw_input('Press any key to continue')

    '''
        reset idx for next epoch
    '''
    def resetBatchIdx(self):
        self.batchIdx = 0

    def loadFeature(self, path, num):
        form = self.form
        self.in_dim = self.length[form]

        #labels = {}
        #data = {}
        data = []
        names = []
        with open(path, 'U') as fptr:
            idx = 0
            for line in fptr:
                name, tmpFeature = line.strip().split(' ', 1)

                idx = idx+1
                if idx > num:
                    break

                features = tmpFeature.split(' ')
                features = [float(feature) for feature in features]
                n = np.array(features)

                data.append(n.reshape(-1, 1))
                names.append(name)
        
        return data, names

    def loadLabels(self, path, num):
        labels = []
        with open(self.LBL_PATH, 'U') as fptr:
            idx = 0
            for line in fptr:
                name, lbl = line.strip().split(',')

                idx = idx+1
                if idx > num:
                    break

                labels.append(lbl)

        return labels

    '''
        loading test examples
    '''
    def loadTest(self, num):

        dataPath = self.PATH + self.form + '/train.ark'
        return self.loadFeature(dataPath, num)
        #return self.loadFeature(dataPath, num)
        
    '''
        loading training examples
    '''
    def loadTrain(self, form, num, out_form='fbank'):
        self.num = num
        self.sanitizeForm(form)

        if self.sanitizeForm(out_form) == False:
            out_form = 'mfcc'

        out_dim = self.length[out_form]
        self.out_dim = out_dim

        dataPath = self.PATH + form + '/train.ark'

        #self.trainData, self.trainLabels = self.loadFeature(dataPath, num, form)

        #self.trainData = []
        #self.trainLabels = []
        #dataDic, lblDic = self.loadFeature(dataPath, num)
        self.trainData, names = self.loadFeature(dataPath, num)
        trainLabels = self.loadLabels(dataPath, num)

        self.ground_truth_vecs = []
        #for key in lblDic.keys():
        for name in trainLabels:
            #self.trainData.append(dataDic[key])

            feature_idx = self.getFeatureIdx(out_form, name)
            ground_truth = np.zeros(shape=(out_dim, 1))
            ground_truth[feature_idx] = 1

            self.ground_truth_vecs.append(ground_truth)
            #value = lblDic[key]
            #feature_idx = self.getFeatureIdx(out_form, value)


        #self.trainData = np.zeros(shape=(self.in_dim,0))
        #self.trainLabels = np.zeros(shape=(out_dim,0))

        #for key in dataDic.keys():
        #    data = dataDic[key]
        #    self.trainData = np.hstack((self.trainData, data))
        #
        #for value in lblDic.values():
        #    feature_idx = self.getFeatureIdx(out_form, value)

        #    ground_truth = np.zeros(shape=(out_dim, 1))
        #    ground_truth[feature_idx] = 1

        #    self.trainLabels = np.hstack((self.trainLabels, ground_truth))
        
    def transformData(self, dim, data):
        out = np.zeros(shape=(dim, 0))
        for i in range(0, len(data)):
            out = np.hstack((out, data[i]))
        return out

    '''
        hand out set of training examples of size 'size'
    '''
    def loadBatch(self, size):
        if self.batchIdx >= self.num:
            return [], [] 
        else:
            #batchData   = np.zeros(shape=(self.in_dim,0))
            #batchY      = np.zeros(shape=(self.out_dim,0))

            idx = self.batchIdx
            batchData   = self.transformData(self.in_dim, self.trainData[idx:idx+size])
            batchY      = self.transformData(self.out_dim, self.ground_truth_vecs[idx:idx+size])
            #for i in range(idx, idx+size):
            #    batchData = np.hstack((batchData, self.trainData[i]))
            #    batchY = np.hstack((batchY, self.ground_truth_vecs[i]))

            #batchData = self.trainData[0:self.in_dim, idx:idx+size]
            #batchY = self.trainLabels[0:self.FBANK_DIM, idx:idx+size]
            self.batchIdx = self.batchIdx+size
            return batchData, batchY

    '''
        claculate which idx should be 1 in ground truth vector
    '''
    def getFeatureIdx(self, form, value):
        assert(form == 'fbank' or form =='mfcc' or form=='1943')
        if form == 'mfcc':
            return self.mfcc[value]            
        elif form == 'fbank':
            return self.fbank[value]
            
