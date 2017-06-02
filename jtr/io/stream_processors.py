from jtr.util.hdf5_processing.pipeline import DatasetStreamer
from jtr.util.hdf5_processing.processors import RemoveLineOnJsonValueCondition, DictKey2ListMapper, JsonLoaderProcessors

def get_snli_stream_processor():
    s = DatasetStreamer()
    s.add_stream_processor(JsonLoaderProcessors())
    s.add_stream_processor(RemoveLineOnJsonValueCondition('gold_label', lambda label: label == '-'))
    s.add_stream_processor(DictKey2ListMapper(['sentence1', 'sentence2', 'gold_label']))
    return s

reader2stream_processor = {}
reader2stream_processor['cbilstm_snli_streaming_reader'] = get_snli_stream_processor()

