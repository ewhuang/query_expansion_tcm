### Author: Edward Huang

import numpy as np
import operator
import sys
import time

### This script rewrites the test files, except with query expansion performed
### on each query patient's list of symptoms. Query expansion is done by
### word embeddings.
### Run time: 

def get_similarity_code_list():
    '''
    Returns the mappings for the columns and rows in the similarity matrix.
    '''
    similarity_code_list = []
    f = open('./data/herb_symptom_dictionary.txt', 'r')
    for i, line in enumerate(f):
        if i == 0:
            continue
        line = line.strip().split('\t')

        line_length = len(line)
        # Some symptoms don't have good English translations.
        assert line_length == 2 or line_length == 5
        if line_length == 2:
            herb, symptom = line
        elif line_length == 5:
            herb, symptom, english_symptom, db_src, db_src_id = line
        if herb not in similarity_code_list:
            similarity_code_list += [herb]
        if symptom not in similarity_code_list:
            similarity_code_list += [symptom]
    f.close()
    return similarity_code_list

def read_similarity_matrix(similarity_code_list):
    '''
    Returns a dictionary of similarity scores.
    Key: (code_a, code_b) -> tuple(str, str)
    Value: cosine similarity -> float
    '''
    similarity_dct = {}
    f = open('./data/similarity_matrix.txt', 'r')
    for i, line in enumerate(f):
        index_a, index_b, score = line.split()
        # Map the indices to real medical codes.
        code_a = similarity_code_list[int(index_a) - 1]
        code_b = similarity_code_list[int(index_b) - 1]
        if (code_b, code_a) in similarity_dct:
            continue
        score = abs(float(score))
        similarity_dct[(code_a, code_b)] = score
    f.close()
    return similarity_dct

def get_count_dct(code_type, run_num):
    code_count_dct = {}
    f = open('./data/count_dictionaries/%s_count_dct_%d.txt' % (code_type,
        run_num), 'r')
    for line in f:
        code, count = line.split()
        code_count_dct[code] = count
    f.close()
    return code_count_dct

def get_expansion_terms(symptom_list, similarity_dct, similarity_code_list,
    training_code_list):
    '''
    Given a query list, find 10 terms that have the highest similarity scores
    to the symptoms in symptom_list.
    '''
    candidate_term_dct = {}
    for query_symptom in symptom_list:
        # Skip a query if it isn't in the dictionary.
        if query_symptom not in similarity_code_list:
            continue
        for training_code in training_code_list:
            # Skip candidates that are already in the query.
            if training_code in symptom_list:
                continue
            # Skip candidates that aren't in the dictionary.
            if training_code not in similarity_code_list:
                continue

            if (query_symptom, training_code) in similarity_dct:
                score = similarity_dct[(query_symptom, training_code)]
            else:
                score = similarity_dct[(training_code, query_symptom)]
            # Keep only terms that have a score above a threshold.
            if score > 0.9:
                candidate_term_dct[training_code] = score
    # Get the top 10 terms.
    expansion_terms = sorted(candidate_term_dct.items(),
        key=operator.itemgetter(1), reverse=True)[:10]
    # Extract just the terms from the sorted list.
    expansion_terms = [term[0] for term in expansion_terms]
    return expansion_terms

def query_expansion(run_num, similarity_dct, similarity_code_list, mixed):
    '''
    Runs the query expansion.
    '''
    # The list of medical codes in the training set.
    training_code_list = get_count_dct('symptom', run_num).keys()
    if mixed:
        training_code_list += get_count_dct('herb', run_num).keys()
    
    # Process output filename.
    out_fname = './data/train_test/test_embedding_'
    if mixed:
        out_fname += 'mixed_'
    out_fname += 'expansion_%d.txt' % run_num
    print out_fname

    out = open(out_fname, 'w')
    f = open('./data/train_test/test_no_expansion_%d.txt' % run_num, 'r')
    for query in f:
        # Split by tab, fifth element, split by comma, take out trailing comma.
        query = query.split('\t')
        symptom_list = query[4].split(':')[:-1]

        expansion_terms = get_expansion_terms(symptom_list, similarity_dct,
            similarity_code_list, training_code_list)

        # Write expanded query to file
        expanded_query = query[:]
        expanded_query[4] += ':'.join(expansion_terms) + ':'
        
        out.write('\t'.join(expanded_query))
    f.close()
    out.close()

def main():
    if len(sys.argv) not in [1, 2]:
        print 'Usage: python %s mixed<optional>' % sys.argv[0]
        exit()
    mixed = False
    if len(sys.argv) == 2:
        assert sys.argv[1] == 'mixed'
        mixed = True

    # The keys will become the mappings for the similarity matrix.
    similarity_code_list = get_similarity_code_list()
    similarity_dct = read_similarity_matrix(similarity_code_list)

    for run_num in range(10):
        query_expansion(run_num, similarity_dct, similarity_code_list, mixed)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print "---%f seconds---" % (time.time() - start_time)