import numpy as np
import sys
import time

### This script rewrites the test files, except with query expansion performed
### on each query patient's list of symptoms.

def read_code_list(run_num):
    code_list = []
    f = open('./data/code_lists/code_list_%d.txt' % run_num, 'r')
    for line in f:
        code_list += [line.strip()]
    f.close()
    return code_list

def get_scaled_topic(symptom_list, word_distr, code_list):
    '''
    Given a symptom list (i.e., query), and the word distributions output by
    some LDA run, we want to recompute the topic probabilities. For each topic,
    we multiply the word probabilities of that topic by the number of query 
    terms that appear in the top 200 words. Add together these topics
    elementwise.
    '''
    n_top_words = 200
    scaled_topic = np.zeros(len(code_list))

    for i, topic_dist in enumerate(word_distr):
        topic_words = np.array(code_list)[np.argsort(topic_dist)][:-(
            n_top_words + 1):-1]
        # Number of query terms in the top n words.
        num_shared_terms = len(set(topic_words).intersection(symptom_list))
        scaled_topic += topic_dist * num_shared_terms

    return scaled_topic

def get_highest_prob_words(symptom_list, scaled_topic, code_list):
    '''
    Given the scaled topic, find the top words to add to the given query. Add
    on twice as many expansion terms as there are symptoms.
    '''
    expansion_terms = []
    num_symptoms = len(symptom_list)

    # Multiply by three, just in case the query terms are the highest prob.
    highest_prob_words = np.array(code_list)[np.argsort(scaled_topic)][:-(
        len(symptom_list) * 3 + 1):-1]
    for candidate in highest_prob_words:
        if candidate not in symptom_list:
            expansion_terms += [candidate]
        if len(expansion_terms) == 2 * num_symptoms:
            break
    return expansion_terms

def query_expansion(run_num):
    '''
    Goes through the basic test queries created by train_test_split.py, and adds
    on the words that most co-occur with the query symptoms. Co-occurrence is
    computed by appearances in the topics output by the different LDA models.
    '''
    code_list = read_code_list(run_num)
    word_distr = np.loadtxt('./results/%s_word_distributions/'
        '%s_word_distribution_%d.txt' % (lda_type, lda_type, run_num))
    
    out = open('./data/train_test/test_lda_expansion_%d.txt' % run_num, 'w')
    f = open('./data/train_test/test_no_expansion_%d.txt' % run_num, 'r')
    for query in f:
        # Split by tab, fifth element, split by comma, take out trailing comma.
        query = query.split('\t')
        symptom_list = query[4].split(':')[:-1]

        scaled_topic = get_scaled_topic(symptom_list, word_distr, code_list)
        expansion_terms = get_highest_prob_words(symptom_list, scaled_topic,
            code_list)

        # Write expanded query to file
        expanded_query = query[:]
        expanded_query[4] += ':'.join(expansion_terms) + ':'
        
        out.write('\t'.join(expanded_query))
    f.close()
    out.close()

def main():
    if len(sys.argv) != 2:
        print 'Usage: python %s lda/bilda' % sys.argv[0]
        exit()
    global lda_type
    lda_type = sys.argv[1]
    assert lda_type in ['lda', 'bilda']

    for run_num in range(10):
        query_expansion(run_num)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print "---%f seconds---" % (time.time() - start_time)