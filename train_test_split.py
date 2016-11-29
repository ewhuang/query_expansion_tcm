import numpy as np 
import pandas as pd
import random

def partition(lst, n): 
    division = len(lst) / float(n) 
    return [ lst[int(round(division * i)): int(round(division * (i + 1))
        )] for i in xrange(n) ]

records = pd.read_csv('./data/HIS_tuple_word.txt',delimiter='\t',header=None)
numRecords = records.shape[0]
testSize=numRecords*0.1
trainSize=numRecords*0.9
np.random.seed(111)

index_list = range(int(numRecords))
np.random.shuffle(index_list)
partitioned_index_list = partition(index_list, 10)

for run_num in range(10):
    # testIdx = np.random.choice(range(int(numRecords)),size=int(testSize),replace=False)
    testIdx = partitioned_index_list[run_num]
    trainIdx = [i for i in range(int(numRecords)) if i not in testIdx]
    test_tbl = records.iloc[testIdx]
    train_tbl = records.iloc[trainIdx]
    print "Testing set size: ", len(test_tbl)
    print "Training set size: ", len(train_tbl)
    print "Checking that two tables sum up up to original data size: ", len(test_tbl)+len(train_tbl)==numRecords
    test_tbl.to_csv('./data/train_test/test_no_expansion_%d.txt' % run_num,sep='\t',delimiter='\t',header=None,index=False)
    train_tbl.to_csv('./data/train_test/train_no_expansion_%d.txt' % run_num,sep='\t',delimiter='\t',header=None,index=False)
