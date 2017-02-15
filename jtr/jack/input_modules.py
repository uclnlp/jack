from jtr.pipelines import pipeline
from jtr.preprocess.batch import get_batches
from jtr.preprocess.map import numpify
from jtr.preprocess.vocab import Vocab
from jtr.jack.preprocessing import preprocess_with_pipeline
from jtr.jack.data_structures import Ports

class QuestionOneSupportGlobalCandiatesInputModule(InputModule):
    def __init__(self, shared_vocab_config):
        self.shared_vocab_config = shared_vocab_config


    @property
    def output_ports(self) -> Dict[TensorPort]:
        """Defines the outputs of the InputModule

        1. Word embedding index tensor of questions of mini-batchs
        2. Word embedding index tensor of support of mini-batchs
        3. Max timestep length of mini-batches for support tensor
        4. Max timestep length of mini-batches for question tensor
        5. Labels
        """
        S = Ports.Input.single_support
        Q = Ports.Input.question
        S_len = Ports.Input. support_length
        Q_len = Ports.Input.question_length
        y = Port.Input.atomic_candidates

        return {S.name : S,
                Q.name : Q,
                S_len.name : S_len
                Q_len.name : Q_len.name,
                y.name : y}


    def training_ports(self) -> List[TensorPort]:
        return [Ports.Targets.target_index]

    def __call__(self, qa_settings: List[QASetting])
                    -> Mapping[TensorPort, np.ndarray]:
        corpus = preprocess_with_pipeline(data, test_time)

        x_dict = {
            Ports.Input.single_support : corpus["support"],
            Ports.Input.question : corpus["question"],
            Ports.Input.question_length : corpus['question_lengths'],
            Ports.Input.support_length : corpus['support_lengths'],
            Ports.Input.atomic_candiates : corpus['candidates']
        }

        return numpify(x_dict)


    def setup_from_data(self, data: List[Tuple[QASetting, List[Answer]]])
                            -> SharedResources:
        corpus = preprocess_with_pipeline(data, test_time)
        self.shared_vocab_config
                    .config['num_candiates'] = len(corpus['candidates'])
        pass


    def dataset_generator(self, dataset: List[Tuple[QASetting, List[Answer]]],
                          is_eval: bool) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        corpus = preprocess_with_pipeline(data, test_time, negsamples=1)
        xy_dict = {
            Ports.Input.single_support: corpus["support"],
            Ports.Input.question: corpus["question"],
            Ports.Input.atomic_candidates :  corpus["candidates"],
            Ports.Input.question_length : corpus['question_lengths'],
            Ports.Input.support_length : corpus['support_lengths']
        }
        if not is_eval:
            xy_dict[Ports.Targets.target_index] =  corpus["answers"]

        return get_batches(xy_dict)

    @abstractmethod
    def setup(self):
        pass
