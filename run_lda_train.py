from glob import glob
import os
for filename in glob('./data/train_test/train_no_expansion_*.txt'):
    print 'Working on {}'.format(filename)
    os.system('python monolingual_lda_baseline.py {}'.format(filename))
