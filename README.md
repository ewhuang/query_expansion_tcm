# tcm-biLDA

### Preprocessing
    
    Splits the HIS_tuple_word.txt data file into 10 different pairs of train/
    test data sets.

    ```bash
    $ python train_test_split.py
    ```

### LDA Baseline

    Runs regular LDA for each of the ten training sets. Writes out the word
    distributions in nxm format. n is the number of unique diseases, and m
    is the number of codes. Code mappings follow ./data/code_lists/

    ```bash
    $ python monolingual_lda_baseline.py
    ```

### Method Evaluations

    ```bash
    $ python retrieval_evaluation.py
    ```