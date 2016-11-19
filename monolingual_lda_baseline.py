#!/usr/bin/python
# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np
import os
import lda
import time

def get_patient_dct():
    '''
    Returns dictionary
    Key: (name, date of birth) -> (str, str)
    Value: dictionary, where keys are (name, DOB) pairs and values are tuples
    containing the diseases, diagnosis dates, symptoms, and herbs of each visit.
    '''
    date_format = '%Y-%m-%d'
    patient_dct = {}
    f = open('./data/HIS_tuple_word.txt', 'r')
    for i, line in enumerate(f):
        diseases, name, dob, diagnosis_date, symptoms, herbs = line.split('\t')
        # Always ends with a colon, so the last element of the split will be
        # the empty string.
        disease_list = diseases.split(':')[:-1]
        
        diagnosis_date = diagnosis_date.split('，')[1][:len('xxxx-xx-xx')]
        # Format the diagnosis date.
        diagnosis_date = datetime.strptime(diagnosis_date, date_format)

        symptom_list = symptoms.split(':')[:-1]
        herb_list = herbs.split(':')[:-1]
        # Skip visits that don't have a complete record.
        if len(symptom_list) == 0 or len(herb_list) == 0:
            continue

        # Add the listing to the dictionary.
        key = (name, dob)
        # Each unique patient is its own dictionary.
        if key not in patient_dct:
            patient_dct[key] = {}
        # Remove duplicates from our symptom and herb list.
        patient_dct[key][diagnosis_date] = (disease_list, list(set(symptom_list
            )), list(set(herb_list)))
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
            # Convert each symptom/herb to their index in the code list.
            curr_code_list = symptom_list + herb_list
            curr_code_list = [code_list.index(code) for code in curr_code_list]
            curr_row = [0 for i in range(len(code_list))]
            for code in curr_code_list:
                curr_row[code] += 1
            patient_matrix += [curr_row]
    return np.array(patient_matrix)

def generate_folders():
    res_dir = './results/'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

def run_baseline_lda(patient_matrix, code_list):
    # TODO: change n_topics to number of unique diseases.
    model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
    model.fit(patient_matrix)
    topic_word = model.topic_word_
    n_top_words = 20
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(code_list)[np.argsort(topic_dist)][:-(
            n_top_words+1):-1]
        print 'Topic %d: %s' % (i, ','.join(topic_words))

def main():
    generate_folders()
    patient_dct = get_patient_dct()
    # code_list is the vocabulary list.
    code_list = get_symptom_and_herb_counts(patient_dct)
    patient_matrix = get_matrix_from_dct(patient_dct, code_list)

    # Run LDA
    run_baseline_lda(patient_matrix, code_list)

if __name__ == '__main__':
    start_time = time.time()
    main()
    print "---%f seconds---" % (time.time() - start_time)