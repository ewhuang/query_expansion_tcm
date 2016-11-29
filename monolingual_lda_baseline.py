#!/usr/bin/python
# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np
import os
import lda
import time
import sys

disease_set = set([])

def generate_folders():
    '''
    Creates the results directory.
    '''
    res_dir = './results/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

def get_patient_dct(filename):
    '''
    Returns dictionary
    Key: (name, date of birth) -> (str, str)
    Value: dictionary, where keys are (name, DOB) pairs and values are tuples
    containing the diseases, diagnosis dates, symptoms, and herbs of each visit.
    '''
    # Keep track of the unique set of diseases so we know n_topics.
    global disease_set
    date_format = '%Y-%m-%d'
    patient_dct = {}
    f = open(filename, 'r')
    for i, line in enumerate(f):
        diseases, name, dob, visit_date, symptoms, herbs = line.split('\t')
        # Always ends with a colon, so the last element of the split will be
        # the empty string.
        disease_list = diseases.split(':')[:-1]
        disease_set = disease_set.union(disease_list)

        visit_date = visit_date.split('，')[1][:len('xxxx-xx-xx')]
        # Format the diagnosis date.
        visit_date = datetime.strptime(visit_date, date_format)

        # Format symptom and herb lists and remove duplicates.
        symptom_list = list(set(symptoms.split(':')[:-1]))
        herb_list = list(set(herbs.split(':')[:-1]))

        # Skip visit records that aren't complete.
        if len(symptom_list) == 0 or len(herb_list) == 0:
            continue

        # Name, date of birth pair uniquely identifies a patient.
        key = (name, dob)
        # Initialize the patient's visit dictionary.
        if key not in patient_dct:
            patient_dct[key] = {}
        patient_dct[key][visit_date] = (disease_list, symptom_list, herb_list)
    f.close()

    return patient_dct

def get_symptom_and_herb_counts(patient_dct):
    '''
    Given the patient dictionary, count the symptom and herb occurrences in
    patients with more than one visit. Writes the counts out to file.
    Returns the list of unique medical codes.
    '''
    herb_count_dct, symptom_count_dct = {}, {}
    for key in patient_dct:
        visit_dct = patient_dct[key]
        if len(visit_dct) == 1:
            continue
        for date in visit_dct:
            disease_list, symptom_list, herb_list = visit_dct[date]

            # Update the counts of each symptom and herb.
            for symptom in symptom_list:
                if symptom not in symptom_count_dct:
                    symptom_count_dct[symptom] = 0
                symptom_count_dct[symptom] += 1
            for herb in herb_list:
                if herb not in herb_count_dct:
                    herb_count_dct[herb] = 0
                herb_count_dct[herb] += 1

    # Write out the unique symptoms and herbs to file.
    herb_out = open('./data/herb_count_dct.txt', 'w')
    for herb in herb_count_dct:
        herb_out.write('%s\t%d\n' % (herb, herb_count_dct[herb]))
    herb_out.close()

    symptom_out = open('./data/symptom_count_dct.txt', 'w')
    for symptom in symptom_count_dct:
        symptom_out.write('%s\t%d\n' % (symptom, symptom_count_dct[symptom]))
    symptom_out.close()

    return list(set(symptom_count_dct.keys()).union(herb_count_dct.keys()))

def get_matrix_from_dct(patient_dct, code_list):
    '''
    Convert the patient dictionary to a document-term matrix.
    '''
    patient_matrix = []
    for key in patient_dct:
        visit_dct = patient_dct[key]
        # Skip patients that only had one visit.
        if len(visit_dct) == 1:
            continue
        for date in sorted(visit_dct.keys()):
            disease_list, symptom_list, herb_list = visit_dct[date]
            curr_code_list = symptom_list + herb_list
            # Create binary vectors for each patient visit.
            curr_row = [1 if c in curr_code_list else 0 for c in code_list]
            patient_matrix += [curr_row]
    return np.array(patient_matrix)

def run_baseline_lda(patient_matrix, code_list):
    model = lda.LDA(n_topics=len(disease_set), n_iter=1500, random_state=1)
    model.fit(patient_matrix)
    topic_word = model.topic_word_
    n_top_words = 20
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(code_list)[np.argsort(topic_dist)][:-(
            n_top_words+1):-1]
        print 'Topic %d: %s' % (i, ','.join(topic_words))
    return topic_word
    
def main():

    if len(sys.argv) == 2:
        filename=sys.argv[1]
    else:
        filename='./data/HIS_tuple_word.txt'
    print filename
    generate_folders()
    patient_dct = get_patient_dct(filename)
    # code_list is the vocabulary list.
    code_list = get_symptom_and_herb_counts(patient_dct)
    patient_matrix = get_matrix_from_dct(patient_dct, code_list)

    # Run LDA.
    topic_word = run_baseline_lda(patient_matrix, code_list)
    np.savetxt('lda_word_distribution.txt',topic_word)

if __name__ == '__main__':
    main()
