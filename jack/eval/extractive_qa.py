import logging
import re
import string
from collections import Counter

logger = logging.getLogger(__name__)


def evaluate(reader, dataset, batch_size):
    answers = reader.process_dataset(dataset, batch_size, silent=False)

    f1 = exact_match = 0
    for pa, (q, ass) in zip(answers, dataset):
        ground_truth = [a.text for a in ass]
        f1 += metric_max_over_ground_truths(f1_score, pa.text, ground_truth)
        exact_match += metric_max_over_ground_truths(exact_match_score, pa.text, ground_truth)

    f1 /= len(answers)
    exact_match /= len(answers)

    return {'F1': f1, 'Exact': exact_match}


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = [0.0]
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
