import sys
import time
import cv2
import numpy as np
import math
import csv
from loader import Loader
from dnn import DNN

BATCH = 200
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
    print 'Load test data...'
    test_data, names = Loader().loadTest(N)
    len_data = len(test_data)
    print "Done loading data features\nLength: " + str(len_data)

    print 'Load model...'
    dnn_model = DNN([1,1,1])
    dnn_model = dnn_model.load()

    print 'Feed data to model...'
    idx = 0
    while idx < len_data:
        if idx + BATCH < len_data:
            end = idx + BATCH
        else:
            end = len_data-1
        
        X = Loader().transformData(39, test_data[idx:end])
        idx = idx + BATCH
         
        ans = dnn_model.predict(X) # [[id1, label1], [id2, label2]]
        print ans

    ans = []
    for key in names:
        ans.append([name, test_data[key][0][0]])
    print 'Output solution for submission'
    solution(ans)

if __name__ == '__main__':
    main(sys.argv)
