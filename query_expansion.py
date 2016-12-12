import collections
import os 
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
        if len(expansion_terms) == 10:#2 * num_symptoms:
            break
    return expansion_terms
def get_highest_cooccuring_words(symptom_list,run_num,sh_mixed='sympt_only'):
    '''
    Given the symptom list, find the highest co-occuring topics
    then find the highest co-occuring words in those topics. 
    '''
    with open("./data/sequence/pltm_output_topics{}.txt".format(run_num)) as f : 
        lines = f.readlines()
        #Process the MALLET output into a list of list of top topic words 
        herb_top_topic_words =[]
        sympt_top_topic_words =[]
        for l in lines: 
            ls =l.split('\t')
            if len(ls) ==2:
                #Beginning of a new topic
                pass
            else:
        #         print ls[0]
        #         print ls[3]
                word_lst = ls[3].split(" ")
                if int(ls[0])==0:
                    #HERB language
                    herb_top_topic_words.append(word_lst)
                elif int(ls[0])==1:
                    #SYMPT language
                    sympt_top_topic_words.append(word_lst)

    num_symptom = len(symptom_list)
    #Mixing together symptom and herb
    #if sh_mixed: sympt_top_topic_words.extend(herb_top_topic_words)

    sympt_cooccurence =[filter(lambda x: x in symptom_list, topic_lst) for topic_lst in sympt_top_topic_words]
    sympt_cooccurence_count = [len(c) for c in sympt_cooccurence] #cooccurence count for each topic 
    #top-k cooccurence topics index
    sympt_topk_topics  = np.argsort(sympt_cooccurence_count)[::-1][:3*num_symptom+1]
    
    expansion_terms = []
    if (sh_mixed=='mixed'):
        topk_sympt_top_topic_words = np.array(sympt_top_topic_words)[sympt_topk_topics]
        
        flatten_lst = []
        for word_lst in topk_sympt_top_topic_words:
            flatten_lst+=word_lst
        topk_herb_top_topic_words = np.array(herb_top_topic_words)[sympt_topk_topics]

        for word_lst in topk_herb_top_topic_words:
            flatten_lst+=word_lst
        word_counter = collections.Counter(flatten_lst)
        for k,v in word_counter.most_common(10+1):
            if k!='\n':
                expansion_terms.append(k)
    #Compute co-occurence
    elif ( sh_mixed=='sympt' ):
        #find query expansion terms by looking at top-occuring words in those topics 
        topk_sympt_top_topic_words = np.array(sympt_top_topic_words)[sympt_topk_topics]
        
        flatten_lst = []
        for word_lst in topk_sympt_top_topic_words:
            flatten_lst+=word_lst
        word_counter = collections.Counter(flatten_lst)
        expansion_terms = []
        for k,v in word_counter.most_common(10+1):#2*num_symptom+1):
            if k!='\n':
                expansion_terms.append(k)
        
    elif (sh_mixed=='herb'): 
        topk_herb_top_topic_words = np.array(herb_top_topic_words)[sympt_topk_topics]
        # print "topk_herb_top_topic_words: "
        flatten_lst = []
        for word_lst in topk_herb_top_topic_words:
            flatten_lst+=word_lst

        word_counter = collections.Counter(flatten_lst)
        for k,v in word_counter.most_common(10+1):#2*num_symptom+1):
            if k!='\n':
                expansion_terms.append(k)

    return expansion_terms

def query_expansion(run_num,sh_mixed='sympt_only'):
    '''
    Goes through the basic test queries created by train_test_split.py, and adds
    on the words that most co-occur with the query symptoms. Co-occurrence is
    computed by appearances in the topics output by the different LDA models.
    '''
    if lda_type=='lda':
        code_list = read_code_list(run_num)
        word_distr = np.loadtxt('./results/%s_word_distributions/'
            '%s_word_distribution_%d.txt' % (lda_type, lda_type, run_num))
    
    if sh_mixed!='sympt_only':
        out = open('./data/train_test/test_%s_%s_expansion_%d.txt' %(lda_type,sh_mixed,run_num) , 'w')
    else:
        out = open('./data/train_test/test_%s_expansion_%d.txt' %(lda_type,run_num) , 'w')
    f = open('./data/train_test/test_no_expansion_%d.txt' % run_num, 'r')
    for query in f:
        # Split by tab, fifth element, split by comma, take out trailing comma.
        query = query.split('\t') 
        symptom_list = query[4].split(':')[:-1]
        if len(symptom_list)!=0:
            if lda_type =='lda':
                scaled_topic = get_scaled_topic(symptom_list, word_distr, code_list)
                expansion_terms = get_highest_prob_words(symptom_list, scaled_topic,
                    code_list)
            elif lda_type=='bilda':
                expansion_terms = get_highest_cooccuring_words(symptom_list,run_num,sh_mixed=sh_mixed)
            # Write expanded query to file
            expanded_query = query[:]
            expanded_query[4] += ':'.join(expansion_terms) + ':'
            
            out.write('\t'.join(expanded_query))
    f.close()
    out.close()

def main():
    if len(sys.argv) != 3:
        print 'Usage: python %s lda/bilda sh_mixed' % sys.argv[0]
        exit()
    global lda_type
    lda_type = sys.argv[1]
    assert lda_type in ['lda', 'bilda']
    sh_mixed= sys.argv[2]
    # if sys.argv[2]== 'sh_mixed':
    #     sh_mixed = True
    # else:
    #     sh_mixed='sympt_only'


    for run_num in range(10):
        query_expansion(run_num,sh_mixed=sh_mixed)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print "---%f seconds---" % (time.time() - start_time)