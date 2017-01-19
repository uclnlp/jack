import numpy as np


def ranking(true_scores, false_scores):
    """ For a list of score values of true facts (true scores) and a list of
    score values of false facts (false scores), compute the ranking.
    Output: returns the ranks of true facts among false ones.
    """
    all_scores = true_scores + false_scores
    all_scores_sorted = sorted(all_scores)
    # highest scores on top
    all_scores_sorted.reverse()
    true_ranks = []
    for score in true_scores:
        rank = all_scores_sorted.index(score) + 1  # indexing starts at 0, ranks at 1.
        true_ranks.append(rank)

    return list(set(true_ranks))


def RR(true_rank):
    """ Reciprocal Rank"""
    return 1.0 / float(true_rank)


def precision_at_X(true_ranks, X):
    hits_list = [rank in true_ranks for rank in range(1, X + 1)]
    return np.mean(hits_list)


def average_precision(true_ranks):
    precision_values = []
    for X in range(1, max(true_ranks) + 1):
        if X in true_ranks:
            precision_values.append(precision_at_X(true_ranks, X))
    return np.mean(precision_values)


if __name__ == "__main__":
    # testing above functions
    print(RR(4))
    print(precision_at_X([1], 10))
    print(precision_at_X([1, 3], 10))
    print(precision_at_X([1, 3], 2))
    print(average_precision([1]))
    print(average_precision([3]))
    print(average_precision([1, 4]))


