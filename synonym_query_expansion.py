### Author: Edward Huang

import time

### This script performs query expansion by synonyms according to the herb-
### symptom dictionary.
### Run time: 10 seconds.

def read_code_list(run_num):
    code_list = []
    f = open('./data/code_lists/code_list_%d.txt' % run_num, 'r')
    for line in f:
        code_list += [line.strip()]
    f.close()
    return code_list

def get_medicine_dictionary_file(run_num):
    '''
    Returns the dictionary produced by the file.
    Keys: symptom -> str
    Values: lists of herbs -> list(str)
    '''
    code_list = read_code_list(run_num)
    herb_symptom_dct = {}

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

        if herb not in code_list:
            continue

        if symptom not in herb_symptom_dct:
            herb_symptom_dct[symptom] = []
        herb_symptom_dct[symptom] += [herb]

    f.close()

    return herb_symptom_dct

def get_expansion_terms(symptom_list, herb_symptom_dct):
    '''
    Given a list of symptoms, return the list of herbs that they all map to.
    '''
    expansion_terms = []
    for symptom in symptom_list:
        if symptom in herb_symptom_dct:
            expansion_terms += herb_symptom_dct[symptom]
    return list(set(expansion_terms))

def query_expansion(run_num, herb_symptom_dct):
    '''
    Performs the query expansion on a particular fold of the test sets. Adds
    to each symptom list the herbs that each symptom treats according to the
    herb-symptom dictionary.
    '''
    out = open('./data/train_test/test_synonym_expansion_%d.txt' % run_num, 'w')
    f = open('./data/train_test/test_no_expansion_%d.txt' % run_num, 'r')
    for query in f:
        # Split by tab, fifth element, split by comma, take out trailing comma.
        query = query.split('\t')
        symptom_list = query[4].split(':')[:-1]

        expansion_terms = get_expansion_terms(symptom_list, herb_symptom_dct)

        # Write expanded query to file
        expanded_query = query[:]
        expanded_query[4] += ':'.join(expansion_terms) + ':'
        
        out.write('\t'.join(expanded_query))
    f.close()
    out.close()

def main():
    for run_num in range(10):
        herb_symptom_dct = get_medicine_dictionary_file(run_num)
        query_expansion(run_num, herb_symptom_dct)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print "---%f seconds---" % (time.time() - start_time)