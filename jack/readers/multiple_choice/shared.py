# -*- coding: utf-8 -*-

from abc import ABCMeta
from typing import Any

from jack.core import *
from jack.data_structures import *
from jack.preprocessing import preprocess_with_pipeline
from jack.util.batch import get_batches
from jack.util.map import numpify
from jack.util.pipelines import pipeline, transpose_dict_of_lists


class SingleSupportFixedClassForward(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward_pass(self, shared_resources,
                     Q_embedding_matrix, Q_ids, Q_lengths,
                     S_embedding_matrix, S_ids, S_lengths,
                     num_classes):
        '''Takes a single support and question and produces logits'''
        raise NotImplementedError


class AbstractSingleSupportFixedClassModel(TFModelModule, SingleSupportFixedClassForward):
    def __init__(self, shared_resources, question_embedding_matrix=None, support_embedding_matrix=None):
        self.shared_resources = shared_resources
        self.vocab = self.shared_resources.vocab
        self.config = self.shared_resources.config
        self.question_embedding_matrix = question_embedding_matrix
        self.support_embedding_matrix = support_embedding_matrix
        super(AbstractSingleSupportFixedClassModel, self).__init__(shared_resources)

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Input.multiple_support,
                Ports.Input.question, Ports.Input.support_length,
                Ports.Input.question_length]

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits,
                Ports.Prediction.candidate_index]

    @property
    def training_input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits,
                Ports.Target.target_index]

    @property
    def training_output_ports(self) -> List[TensorPort]:
        return [Ports.loss]

    def create_output(self, shared_resources: SharedResources,
                      support: tf.Tensor,
                      question: tf.Tensor,
                      support_length: tf.Tensor,
                      question_length: tf.Tensor) -> Sequence[tf.Tensor]:
        question_ids, support_ids = question, support
        if self.question_embedding_matrix is None:
            vocab_size = len(shared_resources.vocab)
            input_size = shared_resources.config['repr_dim_input']
            self.question_embedding_matrix = tf.get_variable(
                "emb_Q", [vocab_size, input_size],
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=True, dtype="float32")
            self.support_embedding_matrix = tf.get_variable(
                "emb_S", [vocab_size, input_size],
                initializer=tf.contrib.layers.xavier_initializer(),
                trainable=True, dtype="float32")

        logits = self.forward_pass(shared_resources,
                                   question_ids, question_length,
                                   support_ids, support_length,
                                   shared_resources.config['answer_size'])

        predictions = tf.argmax(logits, 1, name='prediction')

        return [logits, predictions]

    def create_training_output(self, shared_resources: SharedResources,
                               logits: tf.Tensor, labels: tf.Tensor) -> Sequence[tf.Tensor]:
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels),
                              name='predictor_loss')
        return [loss]


class SimpleMCInputModule(OnlineInputModule[Mapping[str, Any]]):
    def __init__(self, shared_resources):
        self.vocab = shared_resources.vocab
        self.config = shared_resources.config
        self.shared_resources = shared_resources

    def setup_from_data(self, data: Iterable[Tuple[QASetting, List[Answer]]]):

        # Run preprocessing once for all data in order to populate the vocabulary.
        questions, answers = zip(*data)
        self.preprocess(questions, answers)

    @property
    def training_ports(self) -> List[TensorPort]:
        return [Ports.Target.candidate_1hot]

    def preprocess(self, questions: List[QASetting],
                   answers: Optional[List[List[Answer]]] = None,
                   is_eval: bool = False) \
            -> List[Mapping[str, Any]]:

        output_keys = ["support", "question", "candidates"]
        corpus = {
            "support": [q.support for q in questions],
            "question": [q.question for q in questions],
            "candidates": [q.atomic_candidates for q in questions]
        }

        if answers:
            assert len(answers) == len(questions)
            assert all(len(a) == 1 for a in answers)

            corpus["answers"] = [[a[0].text] for a in answers]
            output_keys += ["targets"]

        corpus, _, _, _ = pipeline(corpus, self.vocab, sepvocab=False,
                                   test_time=(answers is None))

        return transpose_dict_of_lists(corpus, output_keys)

    def create_batch(self, annotations: List[Mapping[str, Any]],
                     is_eval: bool, with_answers: bool) \
            -> Mapping[TensorPort, np.ndarray]:

        x_dict = {
            Ports.Input.multiple_support: [a["support"] for a in annotations],
            Ports.Input.question: [a["question"] for a in annotations],
            Ports.Input.atomic_candidates: [a["candidates"] for a in annotations]
        }

        if not is_eval:

            x_dict.update({
                Ports.Target.candidate_1hot: [a["targets"] for a in annotations]
            })

        return numpify(x_dict)

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Input.multiple_support,
                Ports.Input.question, Ports.Input.atomic_candidates]


class MultiSupportFixedClassInputs(InputModule):
    def __init__(self, shared_resources):
        self.shared_resources = shared_resources

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
        corpus, _, _, _ = \
            preprocess_with_pipeline(qa_settings,
                                     self.shared_resources.vocab,
                                     self.shared_resources.answer_vocab,
                                     sepvocab=True,
                                     test_time=True)

        xy_dict = {
            Ports.Input.multiple_support: corpus["support"],
            Ports.Input.question: corpus["question"],
            Ports.Input.question_length: corpus['question_lengths'],
            Ports.Input.support_length: corpus['support_lengths'],
            Ports.Input.sample_id: corpus['ids']
        }

        return numpify(xy_dict)


    def setup_from_data(self, data: Iterable[Tuple[QASetting, List[Answer]]]):
        sepvocab=True
        corpus, train_vocab, train_answer_vocab, train_candidate_vocab = \
                preprocess_with_pipeline(data, self.shared_resources.vocab,
                        None, sepvocab=sepvocab)
        train_vocab.freeze()
        train_answer_vocab.freeze()
        train_candidate_vocab.freeze()
        self.shared_resources.config['answer_size'] = len(train_answer_vocab)
        self.shared_resources.vocab = train_vocab
        if sepvocab:
            self.shared_resources.answer_vocab = train_answer_vocab
        else:
            self.shared_resources.answer_vocab = train_vocab

    def batch_generator(self, dataset: Iterable[Tuple[QASetting, List[Answer]]], batch_size: int, is_eval: bool) \
            -> List[Mapping[TensorPort, np.ndarray]]:
        corpus, _, _, _ = \
                preprocess_with_pipeline(dataset,
                        self.shared_resources.vocab,
                        self.shared_resources.answer_vocab,
                        sepvocab=True)

        xy_dict = {
            Ports.Input.multiple_support: corpus["support"],
            Ports.Input.question: corpus["question"],
            Ports.Target.target_index:  [a[0] for a in corpus["answers"]],
            Ports.Input.question_length: corpus['question_lengths'],
            Ports.Input.support_length: corpus['support_lengths'],
            Ports.Input.sample_id: corpus['ids']
        }

        return get_batches(xy_dict, batch_size)


class SimpleMCModelModule(TFModelModule):

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Input.multiple_support, Ports.Input.question, Ports.Input.atomic_candidates]

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits]

    @property
    def training_input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits, Ports.Target.candidate_1hot]

    @property
    def training_output_ports(self) -> List[TensorPort]:
        return [Ports.loss]

    def create_training_output(self,
                               shared_resources: SharedResources,
                               logits: tf.Tensor,
                               candidate_labels: tf.Tensor) -> Sequence[tf.Tensor]:
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                labels=candidate_labels)
        return loss,

    def create_output(self,
                      shared_resources: SharedResources,
                      multiple_support: tf.Tensor,
                      question: tf.Tensor,
                      atomic_candidates: tf.Tensor) -> Sequence[tf.Tensor]:
        emb_dim = shared_resources.config["repr_dim"]
        with tf.variable_scope("simplce_mcqa"):
            # varscope.reuse_variables()
            embeddings = tf.get_variable(
                "embeddings", [len(self.shared_resources.vocab), emb_dim],
                trainable=True, dtype="float32")

            embedded_supports = tf.reduce_sum(tf.gather(embeddings, multiple_support), (1, 2))  # [batch_size, emb_dim]
            embedded_question = tf.reduce_sum(tf.gather(embeddings, question), (1,))  # [batch_size, emb_dim]
            embedded_supports_and_question = embedded_supports + embedded_question
            embedded_candidates = tf.gather(embeddings, atomic_candidates)  # [batch_size, num_candidates, emb_dim]

            scores = tf.matmul(embedded_candidates, tf.expand_dims(embedded_supports_and_question, -1))

            squeezed = tf.squeeze(scores, 2)
            return squeezed,


class SimpleMCOutputModule(OutputModule):
    def __init__(self):
        self.setup()

    def setup(self):
        pass

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits]

    def __call__(self, inputs: List[QASetting], logits: np.ndarray) -> List[Answer]:
        # len(inputs) == batch size
        # logits: [batch_size, max_num_candidates]
        winning_indices = np.argmax(logits, axis=1)
        result = []
        for index_in_batch, question in enumerate(inputs):
            winning_index = winning_indices[index_in_batch]
            score = logits[index_in_batch, winning_index]
            ans = Answer(question.atomic_candidates[winning_index], score=score)
            result.append(ans)
        return result


class MisclassificationOutputModule(OutputModule):

    def __init__(self, interval, limit=100):
        self.lower, self.upper = interval
        self.limit = limit
        self.i = 0
        self.setup()

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits,
                Ports.Prediction.candidate_index,
                Ports.Target.target_index,
                Ports.Input.sample_id]

    def __call__(self, inputs: List[QASetting],
                 logits,
                 candidate_idx,
                 labels,
                 sample_ids) -> List[Answer]:
        if self.i >= self.limit:
            return

        class2idx = {}
        idx2class = {}

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x, 1).reshape(-1, 1))
            return e_x / e_x.sum(1).reshape(-1, 1)

        logits = softmax(logits)
        num_classes = logits.shape[1]
        for i, (right_idx, predicted_idx) in enumerate(zip(labels, candidate_idx)):
            data_idx = sample_ids[i]
            qa, answer = inputs[data_idx]
            answer = answer[0]
            if answer.text not in class2idx:
                class2idx[answer.text] = right_idx
                idx2class[right_idx] = answer.text
            if len(class2idx) < num_classes: continue
            if self.i >= self.limit: continue
            if right_idx == predicted_idx: continue
            score = logits[i][right_idx]
            if self.lower < score < self.upper:
                self.i += 1
                logger.info('Question: {0}'.format(qa.question))
                logger.info('Support: {0}'.format(qa.support[0]))
                logger.info('Answer: {0}'.format(answer.text))
                logger.info('Predicted class: {0}'.format(idx2class[predicted_idx]))

                predictions_str = str([(idx2class[b], a) for a,b in zip(logits[i], range(num_classes))])
                logger.info('Predictions: {0}'.format(predictions_str))

    def setup(self):
        pass

    def store(self, path):
        pass

    def load(self, path):
        pass
