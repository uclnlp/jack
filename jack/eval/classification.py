from collections import defaultdict


def evaluate(reader, dataset, batch_size):
    answers = reader.process_dataset(dataset, batch_size, silent=False)

    confusion_matrix = defaultdict(lambda: defaultdict(int))

    for (q, a), pa in zip(dataset, answers):
        confusion_matrix[a[0].text][pa.text] += 1

    classes = sorted(confusion_matrix.keys())
    max_class = max(6, len(max(classes, key=len)))

    precision = dict()
    recall = dict()
    f1 = dict()

    confusion_matrix_string = ['\n', ' ' * max_class]
    for c in classes:
        confusion_matrix_string.append('\t')
        confusion_matrix_string.append(c)
        confusion_matrix_string.append(' ' * (max_class - len(c)))
    confusion_matrix_string.append('\n')
    for c1 in classes:
        confusion_matrix_string.append(c1)
        confusion_matrix_string.append(' ' * (max_class - len(c1)))
        for c2 in classes:
            confusion_matrix_string.append('\t')
            ct = str(confusion_matrix[c1][c2])
            confusion_matrix_string.append(ct)
            confusion_matrix_string.append(' ' * (max_class - len(ct)))
        confusion_matrix_string.append('\n')
        precision[c1] = confusion_matrix[c1][c1] / max(1.0, sum(p[c1] for p in confusion_matrix.values()))
        recall[c1] = confusion_matrix[c1][c1] / max(1.0, sum(confusion_matrix[c1].values()))
        f1[c1] = 2 * precision[c1] * recall[c1] / max(1.0, precision[c1] + recall[c1])

    accuracy = sum(confusion_matrix[c][c] for c in classes) / max(
        1.0, sum(sum(vs.values()) for vs in confusion_matrix.values()))

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Confusion Matrix': ''.join(confusion_matrix_string)
    }
