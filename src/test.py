from loader import Loader

ld = Loader()
#data = ld.loadTrain('mfcc', 100)
data = ld.loadTest('mfcc', 100)
for key in data.keys():
    print data[key][1:5,0]
