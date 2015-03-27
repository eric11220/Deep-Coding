import sys
import time
import cv2
import numpy as np
import math
import csv
from loader import Loader
'''
import dnn
'''

FEATURE = 'mfcc'
N = 100

def solution(ans):
    with open('submission.csv', 'wb') as submission:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(submission, fieldnames=fieldnames)
        writer.writeheader()
    
        for each_ans in ans:
            writer.writerow({'Id': each_ans[0], 'Prediction': each_ans[1]})
    return

def main(argv):
    print 'Load test data'
    test_data = Loader().loadTest(FEATURE, N)

    '''
    print 'Load model'
    dnn_model = dnn.load_model()

    print 'Feed data to model'
    ans = dnn_model.forward() # [[id1, label1], [id2, label2]]
    '''
    ans = []
    for key in test_data.keys():
        ans.append(key, test_data[key])
    print ans
    print 'Output solution for submission'
    solution(ans)

if __name__ == '__main__':
    main(sys.argv)
