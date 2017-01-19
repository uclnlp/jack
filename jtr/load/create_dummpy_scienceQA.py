import json

def create_qa_example():
    support = ["To limit the computational complexity for each dataset the selection algorithm used a population of only 10 incidence cubes. The mutation rate was set to p=0.1 while specific selection technique described in previous section ensured non-decreasing convergence of the GA in the average classification performance. The chromosomes were built along three dimensions capturing features, classifiers and combiners incidence. They have been evaluated by the average misclassification rate obtained for all layers (combiners) separately. To preserve generalisation abilities of the system, the classifiers and hence the combiners were built on the separate training sets and tested on parts of the dataset which have not been used during training. Then the training and testing sets were swapped such that an equivalent of two-fold cross-validation rule has been used for chromosome evaluation. For simplicity the GA was stopped after 100 generations for all datasets, despite the fact that for some cases convergence was achieved earlier. Fig. 4 illustrates the dynamics of the testing performance characteristics during selection process carried out by the GA algorithm. The typical observation is that the algorithm relatively quickly finds the best performing system and then in subsequent generations it keeps improving other solutions in the population. The algorithm showed the capacity to get out of local minima which effectively means discovery of significantly better solution spreading swiftly in many variations during subsequent generations. Fig. 5 depicts the evaluation of a final population of chromosomes for both datasets. For Iris dataset the Min combiner showed the best average performance including the absolute best performing system with only 1.33% misclassification rate. Majority voting showed on average best performance for Liver dataset, including the absolute best performing system with 27.8% error rate. The best systems for both datasets were then further uncovered by illustrating the structure of the classifiers and features selected as shown in Fig. 6. Interestingly, for each selected classifier the algorithm selected at least two features. One classifier for both datasets was excluded. Other than that there is nothing significant about the selection structures shown in Fig. 6. This could only prove that it is very difficult to find the best performing systems as they do not exhibit any visible distinctiveness but are simply lost among large number of system designs embodying huge selection complexity as shown in Eq. (4).",
               "To illustrate the advantages of FCCA, we apply our algorithm to a widely used dataset, Iris dataset [8] for classification. It consists of three target classes: Iris Setosa, Iris Virginica and Iris Versicolor. Each species contains 50 data samples. Each sample has four real-valued features: sepal length, sepal width, petal length and petal width. Before training, data are normalized using (1). After normalization, all features of these samples are between zero and one. By doing this, all features have the same contribution to Iris classification. In our method, two output neurons are needed to represent Iris Setosa and Iris Virginica. Samples from Iris Setosa cause the first output neuron to fire. Samples from Iris Virginica cause the second output neuron to fire. And the samples causing neither output neuron to fire belong to Iris Versicolor."]
    candidates_list = ["classification", "selection"]

    qdict = {
        'question': "What task is the Iris dataset used for?",
        'candidates': [
            {
                'text': cand
            } for cand in candidates_list
            ],
        'answers': [{'text': "classification"}]
    }
    questions = [qdict]
    qset_dict1 = {
        'support': [{'text': supp} for supp in support],
        'questions': questions,
        'id': "inst1"
    }


    support = [
        "In this section, the proposed MDMBSS algorithm is evaluated and is also compared with the previously-proposed MLBSS algorithm under a variety of noise conditions. In order to assess the effectiveness of the proposed algorithm, speech recognition experiments are conducted on three speech databases: FARSDAT [28], TIMIT [29], and a recorded database in a real office environment. The first and second test sets are obtained by artificially adding seven noise types (alarm, brown, multitalker, pink, restaurant, volvo, and white noise) from the NOISEX-92 database [30] to the FARSDAT and TIMIT speech databases, respectively.",
        "Experiments are done in two different operational modes of the Nevisa system: phoneme recognition on FARSDAT and TIMIT databases and isolated command recognition on a distant talking database recorded in a real noisy environment. In each test, one sentence of the test set is used in the optimization phase of the MDMBSS algorithm. After vector α is extracted, speech recognition is performed on the remaining test set sentences using the obtained optimized vector α. For each noise type, the optimization phase is done separately."
        ]

    candidates_list = ["speech recognition", "phoneme recognition", "recognition", "isolated command recognition"]

    qdict = {
        'question': "What task is the TIMIT dataset used for?",
        'candidates': [
            {
                'text': cand
            } for cand in candidates_list
            ],
        'answers': [{'text': "speech recognition"}]
    }
    questions = [qdict]
    qset_dict2 = {
        'support': [{'text': supp} for supp in support],
        'questions': questions,
        'id': "inst2"
    }


    instances = [qset_dict1, qset_dict2]

    corpus_dict = {
        'meta': "scienceQA_snippet.json",
        'instances': instances
    }

    return corpus_dict

def main():
    # some tests:
    # raw_data = load_cbt_file(path=None, part='valid', mode='NE')
    # instances = split_cbt(raw_data)
    # = parse_cbt_example(instances[0])
    corpus = create_qa_example()
    with open("../../jtr/data/scienceQA/snippet_jtrformat.json", 'w') as outfile:
        json.dump(corpus, outfile, indent=2)

    outfile.close()

if __name__ == "__main__":
    main()