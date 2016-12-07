# tcm-biLDA

### Preprocessing
    
1. Removes blank records.

    ```bash
    $ python remove_blank_queries.py
    ```

2.  Splits the HIS_tuple_word.txt data file into 10 different pairs of train/
    test data sets.

    ```bash
    $ python train_test_split.py
    ```

### LDA Baseline

1.  Runs regular LDA for each of the ten training sets. Writes out the word
    distributions in nxm format. n is the number of unique diseases, and m
    is the number of codes. Code mappings follow ./data/code_lists/

    ```bash
    $ python monolingual_lda_baseline.py
    ```

### Query Expansions

1.  Adds the appropriate query expansion terms to each patient record's list of
    symptoms. The number of expansions terms is equal to twice the number of
    query terms.

    ```bash
    $ python topic_query_expansion.py lda/bilda
    ```

2.  Adds the synonyms to each query based on the herb-symptom dictionary.

    ```bash
    $ python synonym_query_expansion.py
    ```

### Method Evaluations

1.  Evaluates the retrievals using Okapi BM25. Relevant documents are patient
    records that share diseases with the query patient. Currently using NDCG 
    as the rank metric, where gain is calculated by the number of intersecting
    diseases between the query and document.

    ```bash
    $ python retrieval_evaluation.py no/lda/bilda/synonym rank_metric
    ```

    rank_metric in ['ndcg', 'precision', 'recall']

2.  Runs the paired t-tests comparing each of the topic model-based query
    expansion to the baseline without expansion.

    ```bash
    $ python significance_test.py lda/bilda/synonym rank_metric
    ```