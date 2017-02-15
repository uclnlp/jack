import tensorflow as tf
from jtr.jack import *
from jtr.jack.data_structures import *
from jtr.pipelines import pipeline
from jtr.preprocess.batch import get_batches
from jtr.preprocess.map import numpify, deep_map, dynamic_subsample, notokenize
from jtr.preprocess.vocab import Vocab

from typing import List, Sequence




#class SimpleKBPPorts:
#    question_embedding = TensorPort(tf.float32, [None, None],
#                                    "question_embedding",
#                                    "embedding for a batch of questions",
#                                    "[num_questions, emb_dim]")


class ModelFInputModule(InputModule):
    def __init__(self, shared_resources):
        self.vocab = shared_resources.vocab
        self.config = shared_resources.config
        self.shared_resources = shared_resources

    def setup_from_data(self, data: List[Tuple[QASetting, List[Answer]]]) -> SharedResources:
        self.preprocess(data)
        return self.shared_resources

    def setup(self, shared_resources: SharedResources):
        pass

    @property
    def training_ports(self) -> List[TensorPort]:
        return [Ports.Targets.target_index]

    def preprocess(self, data, test_time=False):
        corpus = {"support": [], "question": [], "candidates": []}
        if not test_time:
            corpus["answers"] = []
        for xy in data:
            if test_time:
                x = xy
                y = None
            else:
                x, y = xy
            corpus["support"].append(x.support)
            corpus["question"].append(x.question)
            corpus["candidates"].append(x.atomic_candidates)
            assert len(y) == 1
            if not test_time:
                corpus["answers"].append(y[0].text)
        corpus =deep_map(corpus, notokenize, ['question'])
        corpus = deep_map(corpus, self.vocab, ['question'])
        corpus = deep_map(corpus, self.vocab, ['candidates'], cache_fun=True)
        if not test_time:
            corpus = deep_map(corpus, self.vocab, ['answers'])
            corpus=dynamic_subsample(corpus,'candidates','answers',how_many=1)
        return corpus

    def dataset_generator(self, dataset: List[Tuple[QASetting, List[Answer]]],
                          is_eval: bool) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        corpus = self.preprocess(dataset)
        xy_dict = {
            Ports.Input.multiple_support: corpus["support"],
            Ports.Input.question: corpus["question"],
            Ports.Input.atomic_candidates: corpus["candidates"],
            Ports.Targets.target_index: corpus["answers"]
        }
        return get_batches(xy_dict)

    def __call__(self, qa_settings: List[QASetting]) -> Mapping[TensorPort, np.ndarray]:
        corpus = self.preprocess(qa_settings, test_time=True)
        x_dict = {
            Ports.Input.multiple_support: corpus["support"],
            Ports.Input.question: corpus["question"],
            Ports.Input.atomic_candidates: corpus["candidates"]
        }
        return numpify(x_dict)

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Input.multiple_support,
                Ports.Input.question, Ports.Input.atomic_candidates]


class ModelFModelModule(SimpleModelModule):
    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.candidate_scores, FlatPorts.Misc.embedded_question]

    @property
    def training_output_ports(self) -> List[TensorPort]:
        return [Ports.loss]

    @property
    def training_input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.candidate_scores, Ports.Targets.target_index, FlatPorts.Misc.embedded_question]

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Input.question, Ports.Input.atomic_candidates]

    def create_training_output(self,
                               shared_resources: SharedVocabAndConfig,
                               candidate_scores: tf.Tensor,
                               target_index: tf.Tensor,
                               question_embedding: tf.Tensor) -> Sequence[tf.Tensor]:
        with tf.variable_scope("modelf",reuse=True):
            embeddings = tf.get_variable("embeddings")
        embedded_answer = tf.expand_dims(tf.gather(embeddings, target_index),-1)  # [batch_size, repr_dim, 1]
        answer_score = tf.squeeze(tf.batch_matmul(question_embedding,embedded_answer),axis=[1,2])  # [batch_size]
        loss = tf.reduce_mean(tf.nn.softplus(tf.reduce_sum(candidate_scores,1)-2*answer_score))
        return loss,

    def create_output(self,
                      shared_resources: SharedVocabAndConfig,
                      question: tf.Tensor,
                      atomic_candidates: tf.Tensor) -> Sequence[tf.Tensor]:
        repr_dim = shared_resources.config["repr_dim"]
        with tf.variable_scope("modelf"):
            embeddings = tf.get_variable(
                "embeddings", [len(self.shared_resources.vocab), repr_dim],
                trainable=True, dtype="float32")

            embedded_question = tf.gather(embeddings, question)  # [batch_size, 1, repr_dim]
            embedded_candidates = tf.gather(embeddings, atomic_candidates)  # [batch_size, num_candidates, repr_dim]
            
            scores = tf.batch_matmul(embedded_candidates,embedded_question,adj_y=True)
            
            squeezed = tf.squeeze(scores, 2)
            return squeezed, embedded_question



class ModelFOutputModule(OutputModule):
    def setup(self):
        pass

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.candidate_scores]

    def __call__(self, inputs: List[QASetting], candidate_scores: np.ndarray) -> List[Answer]:
        # len(inputs) == batch size
        # candidate_scores: [batch_size, max_num_candidates]
        winning_indices = np.argmax(candidate_scores, axis=1)
        result = []
        for index_in_batch, question in enumerate(inputs):
            winning_index = winning_indices[index_in_batch]
            score = candidate_scores[index_in_batch, winning_index]
            result.append(AnswerWithDefault(question.atomic_candidates[winning_index], score=score))
        return result



