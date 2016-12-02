# Given Query (test set data), find top-k co-occurring words for expanding the query.
from numpy import *
from glob import glob
from monolingual_lda_baseline import *
import pandas as pd
# patient_dct = get_patient_dct(filename)
# for test_file in glob('data/train_test/test_no_expansion_*.txt'):
def query_expansion(runIdx):
	test_file  = 'data/train_test/test_no_expansion_{}.txt'.format(runIdx)
	word_distr = np.loadtxt('./results/lda_word_distribution_train_no_expansion_{}.txt'.format(runIdx))
	symptoms_keys=list(pd.read_csv("./data/symptom_count_dct_train_no_expansion_{}.txt".format(runIdx),delimiter='\t',header=None)[0])    
	test_lda_expansion_file =open('data/train_test/test_lda_expansion_{}.txt'.format(runIdx),'w')

	with open(test_file) as tf:
	    queries = tf.readlines()
	    for query in queries:
	        #Queries are symptoms
	#         print query
	        symptoms =  query.split('\t')[4].split(':')
	#             print symptoms_lst     
	        # Populate a matrix of word prob only for the words that's inside the symptoms query
	        query_term_prob = np.zeros_like(word_distr[:,0])
	        sIdx_lst = []
	        for s in symptoms:
	            try:
	#                 print "working"
	                if s!="":
	                    sIdx = symptoms_keys.index(s)
	                    sIdx_lst.append(sIdx)
	                    query_term_prob += word_distr[:,sIdx]
	            except(ValueError):
	#                 print "Ignore: testing symptom not in training keys"
	                pass
	        # Pick the top k topics that contains a lot of the query terms
	        top_k_topic_query_match = argsort(query_term_prob)[::-1][:10]
	        # Find what additional symptoms do these topics have in common, i.e. which words other than sIdx ones have high prob
	        num_expansion_terms=5
	        expansion_Idx_lst = np.argsort(word_distr[top_k_topic_query_match])[::-1][:,num_expansion_terms]
	#         print expansion_Idx_lst
	        expansion_terms = np.array(symptoms_keys)[expansion_Idx_lst]
	        #Write expanded query to file
	        expanded_query = query.split('\t')
	        expanded_query[4] +=":".join(expansion_terms)
	        test_lda_expansion_file.write('\t'.join(expanded_query)+'\n')
	test_lda_expansion_file.close()
for run in range(10):
	query_expansion(run)

