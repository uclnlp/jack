import sys
import json
from jtr.io.read_jtr import jtr_load as jtr_load
import numpy as np
import argparse

def hits_at_k(predictions, golds, k=1):
    """Returns k-hit rate for predictions and questions with ranked answers"""
    gold_answers = get_nested_answers(golds)
    assert len(predictions) == len(gold_answers)
    num_pred_answers = 0
    num_gold_answers = 0
    hits = 0
    for pred,gold in zip(predictions, gold_answers):
        num_gold_answers += len(gold)
        num_pred_answers += k
        for top_k_answer in pred[0:k]:
            found = False
            for gold_answer in gold:
                if top_k_answer['text'] == gold_answer['text']:
                    hits += 1
                    break
    return hits * 100.0 / num_pred_answers
#    print("Accuracy: ({0}/{1}) = {2:3.3f}".format(hits, num_pred_answers, ))

# untested (no models currently produce rank output)
def mean_reciprocal_rank(predictions, golds):
    """Returs mean reciprocal rank for predictions and Qs with ranked As"""
    gold_answers = get_nested_answers(golds)
    assert len(predictions) == len(gold_answers)
    rank_sum = 0
    # pred and gold are lists of answers for the same question
    for pred,gold in zip(predictions, gold_answers):
        best_rank = 0
        # assumes pred predictions are in ranked order
        # could also sort by 'score' field if found
        for rank,value in enumerate(pred):
            for gold_value in gold:
                if value == gold_value and rank < best_rank:
                    best_rank = rank
                    break
        rank_sum += 1/(best_rank+1)
    return rank_sum / len(predictions)

def get_nested_answers(nested_questions):
    """Takes a list of question-sets; returns a list of answers."""
    return [question['answers']
            for qset in nested_questions
            for question in qset['questions']]

def flatten(nlist):
    return [item for sublist in nlist for item in sublist]

# Again, should be replaced with call to library in jtr.io
def read_data(data_filename):
    with open(data_filename) as data_file:
        data = json.load(data_file)
        return data

def eval_jtr_data(test_data_file='../data/scienceQA/scienceQA_cloze_snippet.json', preds_outfile='../test_out.txt'):
    """Read jtr data in and evaluate. Averages predictions for "multiple_flat" support setting.

    Currently only shows accuracy, but can easily be extended with mrr etc.
    test_data, to be loaded with jtr_load(), preds_outfile, produced with TestAllHook
    """

    probs_all = []
    f = open(preds_outfile, "r")
    last_l = ""
    for l in f:
        if not (l.endswith("correct:False\n") or l.endswith("correct:True\n")):
            last_l = l
            continue
        elif not l.startswith("target:"): # not every testing instance is printed on a new line, some are printed over several lines
            l = last_l.strip("\n") + l
            last_l = ""
        l = l.strip("\n").split("\tlogits:[")
        probs_all.append(l[1].split("]\t")[0])

    parser = argparse.ArgumentParser(description='Train and Evaluate a machine reader')
    parser.add_argument('--supports', default='multiple')
    parser.add_argument('--questions', default='single')
    parser.add_argument('--candidates', default='per-instance')
    parser.add_argument('--answers', default='single')
    args = parser.parse_args()

    test_data = jtr_load(open(test_data_file), **vars(args))
    # 'question': questions, 'support': supports, 'answers': answers, 'candidates': candidates

    globi = 0
    corr = 0
    for i, q in enumerate(test_data['question']):
        sups = test_data['support'][i]
        probs = []
        for s in sups: # more than one support
            if globi < len(probs_all):  # in case not all of test data was used
                probs.append(probs_all[globi])
            else:
                break
            globi += 1

        probs, pred = averagePredictions(probs)
        cands = test_data['candidates'][i]
        ans = test_data['answers'][i]
        goldind = cands.index(ans)
        if pred == goldind:
            corr += 1

    acc = float(corr) / float(len(test_data['question']))
    print("Accuracy averaged", acc)


def averagePredictions(probs):
    """Sum probs for all preds; return average props by index, max-index

    Note that different classes can occur multiple times, so that the
    average probability for each class is summed. Finally the max index over
    these sum of average probabilities is return.

    Args:
        probs (list of strings of probabilities for all classes.
    Returns:
        probdict (dict), predicted (int): A dictionary of summed average
                 probabilities , the index of the max summed average
                 probability.
    """
    # sum probabilities, weight by number of predictions
    probdict = {}
    for p in probs:
        probs_s = p.strip(" ").split()
        for i, pp in enumerate(probs_s):
            if i in probdict:
                probdict[i].append(float(pp))
            else:
                probdict[i] = [float(pp)]

    predicted, maxp = 0, 0.0
    for k, v in probdict.items():
        probdict[k] = np.sum(v) / float(len(v))
        if probdict[k] > maxp:
            predicted = k
            maxp = probdict[k]

    return probdict, predicted


def main():
    if len(sys.argv) == 3:
        predictions = read_data(sys.argv[1])
        gold = read_data(sys.argv[2])
        score = hits_at_k(predictions, gold, k=1)
        print('Accuracy: {0:3.3f}'.format(score))
        mrr = mean_reciprocal_rank(predictions, gold)
        print('Mean Reciprocal Rank: {0:3.3f}'.format(mrr))

if __name__ == "__main__": eval_jtr_data() #main()
