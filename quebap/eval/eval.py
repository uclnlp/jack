import sys
import json

def score(predictions, golds, k=1):
    gold_answers = get_gold_answers(golds)
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
    print("Accuracy: ({0}/{1}) = {2:3.3f}".format(hits, num_pred_answers, hits * 100.0 / num_pred_answers))


def get_gold_answers(golds):
    return [question['answers']
            for qset in golds
            for question in qset['questions']]

def flatten(nlist):
    return [item for sublist in nlist for item in sublist]

# Again, should be replaced with call to library in quebap.io
def read_data(data_filename):
    with open(data_filename) as data_file:
        data = json.load(data_file)
        return data

def main():
    if len(sys.argv) == 3:
        predictions = read_data(sys.argv[1])
        gold = read_data(sys.argv[2])
        score(predictions, gold)

if __name__ == "__main__": main()
