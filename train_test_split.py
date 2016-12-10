import numpy as np 
import pandas as pd
import random

### This script partitions the patient records into 10 equal test sets and 
### training sets. Writes out the files to ./data/train_test.

np.random.seed(111)
out_folder = './data/train_test'

def partition(lst, n):
    '''
    Partitions a list of values into 10 approximately equal parts.
    '''
    division = len(lst) / float(n) 
    return [lst[int(round(division * i)): int(round(division * (i + 1))
        )] for i in xrange(n)]

def main():
    records = pd.read_csv('./data/clean_HIS_tuple_word.txt', delimiter='\t',
        header=None)

    numRecords = records.shape[0]
    index_list = range(int(numRecords))
    np.random.shuffle(index_list)
    # Parition the list of indices into 10 test sets of indices.
    partitioned_index_list = partition(index_list, 10)

    for run_num in range(10):
        testIdx = partitioned_index_list[run_num]
        # Train indices are all indices not in the test indices.
        trainIdx = [i for i in range(int(numRecords)) if i not in testIdx]
        # Fetch the actual records by slicing.
        test_tbl = records.iloc[testIdx]
        train_tbl = records.iloc[trainIdx]
        assert len(test_tbl) + len(train_tbl) == numRecords
        # Write records out to file.
        test_tbl.to_csv('%s/test_no_expansion_%d.txt' % (out_folder, run_num),
            sep='\t', delimiter='\t', header=None, index=False)
        train_tbl.to_csv('%s/train_no_expansion_%d.txt' % (out_folder, run_num),
            sep='\t', delimiter='\t', header=None, index=False)

if __name__ == '__main__':
    main()
