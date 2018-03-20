# -*- coding: utf-8 -*-

from jack.eval import extractive_qa, link_prediction, classification

evaluators = {
    'extractive_qa': extractive_qa.evaluate,
    'link_prediction': link_prediction.evaluate,
    'classification': None
}


def evaluate_reader(reader, dataset, batch_size):
    from jack.readers.implementations import extractive_qa_readers, classification_readers, link_prediction_readers
    reader_name = reader.shared_resources.config.get('reader')
    if reader_name in extractive_qa_readers:
        return extractive_qa.evaluate(reader, dataset, batch_size)
    elif reader_name in link_prediction_readers:
        return link_prediction.evaluate(reader, dataset, batch_size)
    elif reader_name in classification_readers:
        return classification.evaluate(reader, dataset, batch_size)


def pretty_print_results(d, prefix=''):
    for k, v in sorted(d.items(), key=lambda x: x[0]):
        if isinstance(v, dict):
            print(prefix + k + ":")
            pretty_print_results(v, prefix + '\t')
        elif '\n' in str(v):
            print(prefix + k + ":")
            print(str(v).replace('\n', '\n' + prefix + '\t'))
        else:
            print(prefix + k + ":", str(v))
