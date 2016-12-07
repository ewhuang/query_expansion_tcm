from datetime import datetime
import os
import glob
def make_mallet_herb_symptom_files(fold_num):
    disease_set = set([])
    date_format = '%Y-%m-%d'
    f = open('./data/train_test/train_no_expansion_%s.txt'%fold_num, 'r')
    os.chdir('data/sequence/')
    herb_corpus = open('herb%s.txt'%fold_num,'w')
    symptom_corpus = open('symptom%s.txt'%fold_num,'w')
    for i, line in enumerate(f):
    #     if i>10:
    #         break
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
        sdoc = diseases+'\t'+'SYMPT'+'\t'

        for symptom in symptom_list:
            sdoc+=symptom+" "
        sdoc+="\n"
    #     print sdoc
        herb_list = list(set(herbs.split(':')[:-1]))
        hdoc = diseases+'\t'+'HERB'+'\t'
        for herb in herb_list:
            hdoc+=herb+" "
        hdoc+="\n"
    #     print hdoc
        herb_corpus.write(hdoc)
        symptom_corpus.write(sdoc)
    herb_corpus.close()
    symptom_corpus.close()
    # Convert formatted text file to MALLET sequence files 
    os.system("../../../mallet-2.0.8/bin/mallet import-file --input herb{0}.txt  --output herb{0}.sequences ".format(fold_num)+ "--keep-sequence --token-regex '\p{L}+'")
    os.system("../../../mallet-2.0.8/bin/mallet import-file --input symptom{0}.txt  --output symptom{0}.sequences ".format(fold_num)+ "--keep-sequence --token-regex '\p{L}+'")
    os.chdir("../../")
for fold_num in range(10):
    make_mallet_herb_symptom_files(fold_num)
os.chdir("data/sequence/")
for fold_num in range(10):
    print "Working on %s "%fold_num
    os.system("../../../mallet-2.0.8/bin/mallet run cc.mallet.topics.PolylingualTopicModel \
    --output-topic-keys pltm_output_topics{0}.txt --num-top-words 50\
    --language-inputs herb{0}.sequences  symptom{0}.sequences  --num-topics 96 --optimize-interval 10".format(fold_num))