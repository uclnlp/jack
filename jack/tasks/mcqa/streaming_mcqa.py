from jack.core import *
from jack.data_structures import *

from jack.util.hdf5_processing.pipeline import Pipeline, DatasetStreamer
from jack.util.hdf5_processing.processors import AddToVocab, SaveLengthsToState, ConvertTokenToIdx, StreamToHDF5, Tokenizer, NaiveNCharTokenizer
from jack.util.hdf5_processing.processors import JsonLoaderProcessors, DictKey2ListMapper, RemoveLineOnJsonValueCondition, ToLower
from jack.util.hdf5_processing.batching import StreamBatcher

import nltk

class StreamingSingleSupportFixedClassInputs(InputModule):
    def __init__(self, shared_vocab_config):
        self.shared_resources = shared_vocab_config

    @property
    def training_ports(self) -> List[TensorPort]:
        return [Ports.Target.target_index]

    @property
    def output_ports(self) -> List[TensorPort]:
        """Defines the outputs of the InputModule

        1. Word embedding index tensor of questions of mini-batchs
        2. Word embedding index tensor of support of mini-batchs
        3. Max timestep length of mini-batches for support tensor
        4. Max timestep length of mini-batches for question tensor
        5. Labels
        """
        return [Ports.Input.multiple_support,
                Ports.Input.question, Ports.Input.support_length,
                Ports.Input.question_length, Ports.Target.target_index, Ports.Input.sample_id]

    def __call__(self, qa_settings: List[QASetting]) \
            -> Mapping[TensorPort, np.ndarray]:
        pass

    def setup_from_data(self, data: Iterable[Tuple[QASetting, List[Answer]]], dataset_name=None, identifier=None) -> SharedResources:
        # tokenize and convert to hdf5
        # 1. Setup pipeline to save lengths and generate vocabulary
        tokenizer = nltk.tokenize.WordPunctTokenizer()
        p = Pipeline(dataset_name, delete_all_previous_data=identifier=='train')
        if identifier == 'train':
            p.add_sent_processor(ToLower())
            p.add_sent_processor(Tokenizer(tokenizer.tokenize))
            p.add_token_processor(AddToVocab())
            p.execute(data)
            p.save_vocabs()
        else:
            p.load_vocabs()

        # 2. Process the data further to stream it to hdf5
        p.clear_processors()

        # save the lengths of the data
        p.add_sent_processor(Tokenizer(tokenizer.tokenize))
        p.add_post_processor(SaveLengthsToState())
        p.execute(data)

        # convert to indicies and stream to HDF5
        p.clear_processors()
        p.add_sent_processor(ToLower())
        p.add_sent_processor(Tokenizer(tokenizer.tokenize))
        p.add_post_processor(ConvertTokenToIdx())
        p.add_post_processor(StreamToHDF5(identifier))
        p.execute(data)

        if identifier == 'train':
            self.shared_resources.config['answer_size'] = p.state['vocab']['general'].num_labels
            self.shared_resources.vocab = p.state['vocab']['general']
        return self.shared_resources

    def batch_generator(self, dataset: Iterable[Tuple[QASetting, List[Answer]]], is_eval: bool, dataset_name=None,
                        identifier=None) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        batch_size = self.shared_resources.config['batch_size']
        batcher = StreamBatcher(dataset_name, identifier, batch_size)
        self.batcher = batcher

        def gen():
            for str2var in batcher:
                feed_dict = {
                    Ports.Input.multiple_support: str2var['support'].reshape(batch_size, 1, -1),
                    Ports.Input.question: str2var["input"],
                    Ports.Target.target_index:  str2var["target"],
                    Ports.Input.question_length : str2var['input_length'],
                    Ports.Input.support_length : str2var['support_length'].reshape(batch_size, 1),
                    Ports.Input.sample_id : str2var['index']
                }

                yield feed_dict

        return GeneratorWithRestart(gen)
