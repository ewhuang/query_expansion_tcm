### Author: Edward Huang

import math
import operator
from rank_metrics import ndcg_at_k, precision_at_k
import sys
import time

# This script takes as input a biLDA or baseline, and evaluates based on the
# the queries of the test data. 10 fold CV.
# Run time: 9 minutes.

num_docs = 0
avg_doc_len = 0.0
k_1 = 1.5
b = 0.75
k_list = [10, 50, 100]

def read_input_file(fname):
    '''
    Returns a dictionary of a set of patient records, either training or test.
    Key: (name, dob, visit_date) -> (str, str, str)
    Value: (disease list, symptom list, herb list) -> (list(str), list(str),
            list(str))
    '''
    record_dct = {}
    f = open(fname, 'r')
    for i, line in enumerate(f):
        diseases, name, dob, visit_date, symptoms, herbs = line.split('\t')
        
        # Establish the dictionary key.
        key = (name, dob, visit_date)

        disease_list = diseases.split(':')[:-1]
        symptom_list = list(set(symptoms.split(':')[:-1]))
        herb_list = list(set(herbs.split(':')[:-1]))
        record_dct[key] = (disease_list, symptom_list, herb_list)

    f.close()
    return record_dct

def get_inverted_index(corpus_dct, method_type):
    '''
    Given the corpus dictionary, build the inverted dictionary.
    Key: herb or symptom -> str
    Value: list of patient visits in which the key occurs -> list((str,str,str))
    '''
    global avg_doc_len
    inverted_index = {}
    for key in corpus_dct:
        disease_list, symptom_list, herb_list = corpus_dct[key]

        if 'both' in method_type or 'synonym' in method_type:
            avg_doc_len += len(symptom_list + herb_list)
        else:
            avg_doc_len += len(symptom_list)

        # Update the entry for each symptom and each herb.
        for symptom in symptom_list:
            if symptom not in inverted_index:
                inverted_index[symptom] = []
            inverted_index[symptom] += [key]
        for herb in herb_list:
            if herb not in inverted_index:
                inverted_index[herb] = []
            inverted_index[herb] += [key]
    avg_doc_len /= float(len(corpus_dct))
    return inverted_index

def okapi_bm25(query_list, document, inverted_index):
    '''
    Given a query and a document, compute the Okapi BM25 score. Returns a float.
    '''
    score = 0.0
    doc_len = float(len(document))
    for term in query_list:
        if term not in document:
            continue
        if term in inverted_index:
            n_docs_term = len(inverted_index[term])
        else:
            n_docs_term = 0
        idf = math.log((num_docs - n_docs_term + 0.5) / (n_docs_term + 0.5),
            math.e)
        # 1 because there should be no duplicates of terms.
        tf = (k_1 + 1) / (1 + 2.5 * (1 - b + b * doc_len / avg_doc_len))
        score += tf * idf
    return score

def get_rel_score(query_disease_list, doc_disease_list):
    '''
    This function determines how we compute a relevance score between a query's
    diseases and the document's diseases.
    '''
    size_inter = len(set(query_disease_list).intersection(doc_disease_list))
    # Computing the intersection between the two for gain.
    if rank_metric == 'ndcg':
        return size_inter / float(len(query_disease_list))
    # Otherwise, binary relevance.
    else:
        if size_inter > 0:
            return 1
        else:
            return 0

def evaluate_retrieval(query_dct, corpus_dct, inverted_index, method_type):
    '''
    Given a query dictionary and a corpus dictionary, go through each query and
    determine the precision, recall, and F1 measure for its retrieval with the
    disease labels as relevance measures.
    '''
    global num_docs
    num_docs = len(corpus_dct)

    metric_list = []
    metric_dct = {}

    for query_key in query_dct:
        score_dct = {}

        (query_disease_list, query_symptom_list, query_herb_list
            ) = query_dct[query_key]

        for doc_key in corpus_dct:
            (doc_disease_list, doc_symptom_list, doc_herb_list
                ) = corpus_dct[doc_key]

            # With no query expansion, our document is just the set of symptoms.
            if 'both' in method_type or 'synonym' in method_type:
                document = doc_symptom_list + doc_herb_list
            else:
                document = doc_symptom_list

            score = okapi_bm25(query_symptom_list, document, inverted_index)
            # Compute the relevance judgement.
            relevance = get_rel_score(query_disease_list, doc_disease_list)
            score_dct[(doc_key, relevance)] = score

        sorted_scores = sorted(score_dct.items(), key=operator.itemgetter(1),
            reverse=True)
        # Get the relevance rankings.
        rel_list = [pair[0][1] for pair in sorted_scores]

        # Compute different rank metrics for different values of k.
        for k in k_list:
            if k not in metric_dct:
                metric_dct[k] = []
            if rank_metric == 'ndcg':
                metric_dct[k] += [ndcg_at_k(rel_list, k)]
            elif rank_metric == 'precision':
                metric_dct[k] += [precision_at_k(rel_list, k)]
    return metric_dct

def main():
    if len(sys.argv) != 3:
        print 'Usage: python %s no/lda/bilda/synonym rank_metric' % sys.argv[0]
        exit()
    global rank_metric
    assert sys.argv[1] in ['no', 'lda', 'lda_both', 'bilda', 'bilda_both', 'synonym']
    method_type = '%s_expansion' % sys.argv[1]
    rank_metric = sys.argv[2]
    assert rank_metric in ['ndcg', 'precision', 'recall']

    all_metric_dct = {}
    # range(10) because we are performing 10-fold CV.
    for run_num in range(10):
        test_fname = './data/train_test/test_%s_%d.txt' % (method_type, run_num)
        # We never perform query expansion on the training set.
        train_fname = './data/train_test/train_no_expansion_%d.txt' % run_num
        query_dct = read_input_file(test_fname)
        corpus_dct = read_input_file(train_fname)
        inverted_index = get_inverted_index(corpus_dct, method_type)

        metric_dct = evaluate_retrieval(query_dct, corpus_dct, inverted_index,
            method_type)
        # Compile the metric scores across all runs.
        for k in k_list:
            metric_list = metric_dct[k]
            if k not in all_metric_dct:
                all_metric_dct[k] = []
            all_metric_dct[k] += metric_list

    out = open('./results/%s_%s.txt' % (rank_metric, method_type), 'w')
    for k in k_list:
        metric_list = all_metric_dct[k]
        for metric in metric_list:
            out.write('%f\t%d\n' % (metric, k))
    out.close()

if __name__ == '__main__':
    start_time = time.time()
    main()
    print "---%f seconds---" % (time.time() - start_time)