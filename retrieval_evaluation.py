### Author: Edward Huang

import math
import operator
from rank_metrics import ndcg_at_k
import sys
import time

# This script takes as input a biLDA or baseline, and evaluates based on the
# the queries of the test data. 10 fold CV.
# Run time: 22 minutes.

num_docs = 0
avg_doc_len = 0.0
k_1 = 1.5
b = 0.75
k_list = [10, 50, 100, 200, 500, 1000, 1500, 2000, 5000]

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

def get_inverted_index(corpus_dct):
    '''
    Given the corpus dictionary, build the inverted dictionary.
    Key: herb or symptom -> str
    Value: list of patient visits in which the key occurs -> list((str,str,str))
    '''
    global avg_doc_len
    inverted_index = {}
    for key in corpus_dct:
        disease_list, symptom_list, herb_list = corpus_dct[key]
        # TODO: document length is either num symptoms, or num symptoms + herbs.
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

def okapi_bm25(query_list, doc_list, inverted_index):
    '''
    Given a query and a document, compute the Okapi BM25 score. Returns a float.
    '''
    score = 0.0
    doc_len = float(len(doc_list))
    for term in query_list:
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
    # Currently, computing the intersection between the two.
    return len(set(query_disease_list).intersection(doc_disease_list))

def evaluate_retrieval(query_dct, corpus_dct, inverted_index):
    '''
    Given a query dictionary and a corpus dictionary, go through each query and
    determine the precision, recall, and F1 measure for its retrieval with the
    disease labels as relevance measures.
    '''
    global num_docs
    num_docs = len(corpus_dct)

    ndcg_list = []
    ndcg_dct = {}

    for query_key in query_dct:
        score_dct = {}

        (query_disease_list, query_symptom_list, query_herb_list
            ) = query_dct[query_key]

        for doc_key in corpus_dct:
            (doc_disease_list, doc_symptom_list, doc_herb_list
                ) = corpus_dct[doc_key]
            # TODO: Instead of doc_symptom_list, we can do similarity with
            # doc_symptom_list + doc_herb_list for the expanded queries.
            score = okapi_bm25(query_symptom_list, doc_symptom_list,
                inverted_index)
            # Compute the relevance judgement.
            relevance = get_rel_score(query_disease_list, doc_disease_list)
            score_dct[(doc_key, relevance)] = score
        for k in k_list:
            # Sort the score dictionary. Keep the top k ranks.
            sorted_scores = sorted(score_dct.items(),
                key=operator.itemgetter(1), reverse=True)[:k]
            # Get only the relevance judgements for the top k ranked documents.
            rel_list = [pair[0][1] for pair in sorted_scores]
            if k not in ndcg_dct:
                ndcg_dct[k] = []
            ndcg_dct[k] += [ndcg_at_k(rel_list, k)]
    return ndcg_dct

def main():
    if len(sys.argv) != 2:
        print 'Usage: python %s no/lda/bilda' % sys.argv[0]
        exit()
    global lda_type
    assert sys.argv[1] in ['no', 'lda', 'bilda']
    method_type = '%s_expansion' % sys.argv[1]

    avg_ndcg_dct = {}
    for run_num in range(10):
        test_fname = './data/train_test/test_%s_%d.txt' % (method_type, run_num)
        train_fname = './data/train_test/train_no_expansion_%d.txt' % run_num
        query_dct = read_input_file(test_fname)
        corpus_dct = read_input_file(train_fname)
        inverted_index = get_inverted_index(corpus_dct)

        ndcg_dct = evaluate_retrieval(query_dct, corpus_dct, inverted_index)
        # Average the ndcg across all runs.
        for k in k_list:
            ndcg_list = ndcg_dct[k]
            if k not in avg_ndcg_dct:
                avg_ndcg_dct[k] = []
            avg_ndcg_dct[k] += ndcg_list

    out = open('./results/ndcg_%s.txt' % method_type, 'w')
    for k in k_list:
        ndcg_list = avg_ndcg_dct[k]
        avg_ndcg = sum(ndcg_list) / float(len(ndcg_list))
        out.write('%f\t%d\n' % (avg_ndcg, k))        
    out.close()

if __name__ == '__main__':
    start_time = time.time()
    main()
    print "---%f seconds---" % (time.time() - start_time)