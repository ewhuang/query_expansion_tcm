### Author: Edward Huang

### This script rewrites the input file, removing any patient records that have
### blank symptom or herb lists.

def rewrite_input_file():
    out = open('./data/clean_HIS_tuple_word.txt', 'w')
    f = open('./data/HIS_tuple_word.txt', 'r')
    for i, line in enumerate(f):
        diseases, name, dob, visit_date, symptoms, herbs = line.split('\t')
        if name == 'null' or dob == 'null' or visit_date == 'null':
            continue

        disease_list = diseases.split(':')[:-1]
        symptom_list = list(set(symptoms.split(':')[:-1]))
        herb_list = list(set(herbs.split(':')[:-1]))
        if len(disease_list) == 0 or len(symptom_list) == 0 or len(herb_list
            ) == 0:
            continue
        out.write(line)
    f.close()
    out.close()

def main():
    rewrite_input_file()


if __name__ == '__main__':
    main()