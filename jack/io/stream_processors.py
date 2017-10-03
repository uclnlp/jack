from jack.util.hdf5_processing.pipeline import DatasetStreamer
from jack.util.hdf5_processing.processors import RemoveLineOnJsonValueCondition, DictKey2ListMapper, JsonLoaderProcessors

def get_snli_stream_processor():
    s = DatasetStreamer()
    s.add_stream_processor(JsonLoaderProcessors())
    s.add_stream_processor(RemoveLineOnJsonValueCondition('gold_label', lambda label: label == '-'))
    s.add_stream_processor(DictKey2ListMapper(['sentence1', 'sentence2', 'gold_label']))
    return s

dataset2stream_processor = {}
dataset2stream_processor['snli'] = get_snli_stream_processor()

