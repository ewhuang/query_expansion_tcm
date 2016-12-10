#!/usr/bin/python
# -*- coding: utf-8 -*-

import datetime
import numpy as np
import os
import lda
import time

### This script runs regular LDA on a patient record training set (90% of the
### original data). Writes out the herb counts, symptom counts, code list (for
### mapping symptoms/herbs to integers), and the word distributions for each
### topic. The number of topics will match the number of unique diseases.
### Run time for 5000 iterations: 3.5 hours.

date_format = '%Y-%m-%d'

def generate_folders():
    '''
    Creates the results directory.
    '''
    res_dir = './results/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    count_dir = './data/count_dictionaries/'
    if not os.path.exists(count_dir):
        os.makedirs(count_dir)
    code_list_dir = './data/code_lists/'
    if not os.path.exists(code_list_dir):
        os.makedirs(code_list_dir)
    lda_res_dir = './results/lda_word_distributions/'
    if not os.path.exists(lda_res_dir):
        os.makedirs(lda_res_dir)

def get_patient_dct(filename):
    '''
    Returns dictionary
    Key: (name, date of birth) -> (str, str)
    Value: dictionary, where keys are (name, DOB) pairs and values are tuples
    containing the diseases, diagnosis dates, symptoms, and herbs of each visit.
    '''
    # Keep track of the unique set of diseases so we know n_topics.
    patient_dct, disease_set = {}, set([])

    f = open(filename, 'r')
    for i, line in enumerate(f):
        diseases, name, dob, visit_date, symptoms, herbs = line.split('\t')
        # Always ends with a colon, so the last element of the split will be
        # the empty string.
        disease_list = diseases.split(':')[:-1]
        disease_set = disease_set.union(disease_list)

        visit_date = visit_date.split('，')[1][:len('xxxx-xx-xx')]
        # Format the diagnosis date.
        visit_date = datetime.datetime.strptime(visit_date, date_format)

        # Format symptom and herb lists and remove duplicates.
        symptom_list = list(set(symptoms.split(':')[:-1]))
        herb_list = list(set(herbs.split(':')[:-1]))

        # Name, date of birth pair uniquely identifies a patient.
        key = (name, dob)
        # Initialize the patient's visit dictionary.
        if key not in patient_dct:
            patient_dct[key] = {}
        # If multiple visits in one day, add on a second to the visit.
        while visit_date in patient_dct[key]:
            visit_date += datetime.timedelta(0,1)
        # list(set()) removes duplicates.
        patient_dct[key][visit_date] = (disease_list, symptom_list, herb_list)
    f.close()

    return patient_dct, disease_set

def get_symptom_and_herb_counts(patient_dct, run_num):
    '''
    Given the patient dictionary, count the symptom and herb occurrences in
    patients with more than one visit. Writes the counts out to file.
    Returns the list of unique medical codes.
    '''
    herb_count_dct, symptom_count_dct = {}, {}
    for key in patient_dct:
        visit_dct = patient_dct[key]
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
    herb_out = open('./data/count_dictionaries/herb_count_dct_%s.txt' %
        run_num, 'w')
    for herb in herb_count_dct:
        herb_out.write('%s\t%d\n' % (herb, herb_count_dct[herb]))
    herb_out.close()

    symptom_out = open('./data/count_dictionaries/symptom_count_dct_%s.txt' %
        run_num, 'w')
    for symptom in symptom_count_dct:
        symptom_out.write('%s\t%d\n' % (symptom, symptom_count_dct[symptom]))
    symptom_out.close()

    return list(set(symptom_count_dct.keys()).union(herb_count_dct.keys()))

def write_code_list(code_list, run_num):
    '''
    Writes the code list out to file.
    '''
    out = open('./data/code_lists/code_list_%s.txt' % run_num, 'w')
    for code in code_list:
        out.write('%s\n' % code)
    out.close()

def get_matrix_from_dct(patient_dct, code_list):
    '''
    Convert the patient dictionary to a document-term matrix.
    '''
    patient_matrix = []
    for key in patient_dct:
        visit_dct = patient_dct[key]
        for date in sorted(visit_dct.keys()):
            disease_list, symptom_list, herb_list = visit_dct[date]
            curr_code_list = symptom_list + herb_list
            # Create binary vectors for each patient visit.
            curr_row = [1 if c in curr_code_list else 0 for c in code_list]
            patient_matrix += [curr_row]
    return np.array(patient_matrix)

def run_baseline_lda(patient_matrix, code_list, disease_set):
    model = lda.LDA(n_topics=len(disease_set), n_iter=5000, random_state=1)
    model.fit(patient_matrix)
    topic_word = model.topic_word_
    return topic_word

def main():
    generate_folders()

    for run_num in range(10):
        # Fetch the training patient record dictionary.
        patient_fname = './data/train_test/train_no_expansion_%s.txt' % run_num
        patient_dct, disease_set = get_patient_dct(patient_fname)

        # code_list is the vocabulary list.
        code_list = get_symptom_and_herb_counts(patient_dct, run_num)
        write_code_list(code_list, run_num)

        # Convert the patient dictionary to a matrix for LDA.
        patient_matrix = get_matrix_from_dct(patient_dct, code_list)

        # Run LDA.
        topic_word = run_baseline_lda(patient_matrix, code_list, disease_set)
        np.savetxt('./results/lda_word_distributions/lda_word_distribution_%s'
            '.txt' % run_num, topic_word)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print "---%f seconds---" % (time.time() - start_time)