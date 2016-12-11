### Author: Edward Huang

import numpy as np
from scipy.stats import ttest_rel
import sys
import time

### This script evaluates the NDCG's of retrieval systems by performing the
### paired t-test on them.

def read_ndcg_dct(method_type):
    ndcg_dct = {}
    f = open('./results/%s_%s.txt' % (method_type, rank_metric), 'r')
    for line in f:
        ndcg, k = line.split()
        if k not in ndcg_dct:
            ndcg_dct[k] = []
        ndcg_dct[k] += [float(ndcg)]
    f.close()
    return ndcg_dct

def main():
    if len(sys.argv) != 3:
        print ('Usage: python %s no/lda_symptoms/lda_herbs/lda_mixed/bilda_'
            'symptoms/bilda_herbs/bilda_mixed/embedding_symptoms/embedding_'
            'herbs/embedding_mixed/synonym' % sys.argv[0])
        exit()
    global rank_metric
    method_type, rank_metric = sys.argv[1:]
    assert (method_type in ['no', 'lda_symptoms', 'lda_herbs', 'lda_mixed',
        'bilda_symptoms', 'bilda_herbs', 'bilda_mixed', 'embedding_symptoms',
        'embedding_herbs', 'embedding_mixed', 'synonym'])
    assert rank_metric in ['ndcg', 'precision', 'recall']

    baseline_ndcg_dct = read_ndcg_dct('no_expansion')
    lda_ndcg_dct = read_ndcg_dct('%s_expansion' % method_type)
    for k in sorted(baseline_ndcg_dct.keys()):
        baseline_ndcg_list = baseline_ndcg_dct[k]
        lda_ndcg_list = lda_ndcg_dct[k]
        print k
        print 'baseline', np.mean(baseline_ndcg_list), method_type, np.mean(
            lda_ndcg_list)
        print ttest_rel(baseline_ndcg_list, lda_ndcg_list)
        print ''

if __name__ == '__main__':
    start_time = time.time()
    main()
    print "---%f seconds---" % (time.time() - start_time)