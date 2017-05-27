# -*- coding: utf-8 -*-

from jtr.core import *
from jtr.data_structures import *
from jtr.preprocessing import preprocess_with_pipeline
from jtr.tf_fun import rnn, simple

from jtr.tasks.mcqa.abstract_multiplechoice import AbstractSingleSupportFixedClassModel
from jtr.util.batch import get_batches
from jtr.util.map import numpify
from jtr.util.pipelines import pipeline

import logging

logger = logging.getLogger(__name__)


class SimpleMCInputModule(InputModule):
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
        return [Ports.Target.candidate_1hot]

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
                corpus["answers"].append([y[0].text])
        corpus, _, _, _ = pipeline(corpus, self.vocab, sepvocab=False,
                                   test_time=test_time)
        return corpus

    def dataset_generator(self, dataset: List[Tuple[QASetting, List[Answer]]],
                          is_eval: bool) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        corpus = self.preprocess(dataset)
        xy_dict = {
            Ports.Input.multiple_support: corpus["support"],
            Ports.Input.question: corpus["question"],
            Ports.Input.atomic_candidates: corpus["candidates"],
            Ports.Target.candidate_1hot: corpus["targets"]
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

class StreamingSingleSupportFixedClassInputs(InputModule):
    def __init__(self, shared_vocab_config):
        self.shared_vocab_config = shared_vocab_config
        self.setup_from_data(self.shared_vocab_config.train_data)

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

    def setup_from_data(self, data: List[Tuple[QASetting, List[Answer]]]) -> SharedResources:
        raise Exception("Can only be setup from files!")

    def setup_from_file(self, path):
        pass

    def dataset_generator(self, dataset: List[Tuple[QASetting, List[Answer]]],
                          is_eval: bool) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        corpus, _, _, _ = \
                preprocess_with_pipeline(dataset,
                        self.shared_vocab_config.vocab, self.answer_vocab, use_single_support=True, sepvocab=True)

        xy_dict = {
            Ports.Input.multiple_support: corpus["support"],
            Ports.Input.question: corpus["question"],
            Ports.Target.target_index:  corpus["answers"],
            Ports.Input.question_length : corpus['question_lengths'],
            Ports.Input.support_length : corpus['support_lengths'],
            Ports.Input.sample_id : corpus['ids']
        }

        return get_batches(xy_dict)

    def setup(self):
        pass


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
        pass

    def setup_from_data(self, data: List[Tuple[QASetting, List[Answer]]],
                        sepvocab=True) -> SharedResources:
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

    def dataset_generator(self, dataset: List[Tuple[QASetting, List[Answer]]],
                          is_eval: bool, dataset_identifier= None) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        corpus, _, _, _ = \
                preprocess_with_pipeline(dataset,
                        self.shared_resources.vocab,
                        self.shared_resources.answer_vocab,
                        use_single_support=True, sepvocab=True)

        xy_dict = {
            Ports.Input.multiple_support: corpus["support"],
            Ports.Input.question: corpus["question"],
            Ports.Target.target_index:  corpus["answers"],
            Ports.Input.question_length: corpus['question_lengths'],
            Ports.Input.support_length: corpus['support_lengths'],
            Ports.Input.sample_id: corpus['ids']
        }

        return get_batches(xy_dict)

    def setup(self):
        pass


class SimpleMCModelModule(SimpleModelModule):

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
            result.append(Answer(question.atomic_candidates[winning_index], score=score))
        return result


class PairOfBiLSTMOverSupportAndQuestionModel(AbstractSingleSupportFixedClassModel):
    def forward_pass(self, shared_resources,
                     Q_ids, Q_lengths,
                     S_ids,  S_lengths,
                     num_classes):
        # final states_fw_bw dimensions:
        # [[[batch, output dim], [batch, output_dim]]
        S_ids = tf.squeeze(S_ids, 1)
        S_lengths = tf.squeeze(S_lengths, 1)

        Q_seq = tf.nn.embedding_lookup(self.question_embedding_matrix, Q_ids)
        S_seq = tf.nn.embedding_lookup(self.support_embedding_matrix, S_ids)

        all_states_fw_bw, final_states_fw_bw = rnn.pair_of_bidirectional_LSTMs(
            Q_seq, Q_lengths, S_seq, S_lengths, shared_resources.config['repr_dim'],
            drop_keep_prob=1.0 - shared_resources.config['dropout'],
            conditional_encoding=True)
        # ->  [batch, 2*output_dim]
        final_states = tf.concat([final_states_fw_bw[0][1], final_states_fw_bw[1][1]],axis=1)
        # [batch, 2*output_dim] -> [batch, num_classes]
        outputs = simple.fully_connected_projection(final_states, num_classes)
        return outputs


class DecomposableAttentionModel(AbstractSingleSupportFixedClassModel):
    def forward_pass(self, shared_resources,
                     question, question_length,
                     support, support_length,
                     num_classes):
        # final states_fw_bw dimensions:
        # [[[batch, output dim], [batch, output_dim]]
        support = tf.squeeze(support, 1)
        support_length = tf.squeeze(support_length, 1)

        question_embedding = tf.nn.embedding_lookup(self.question_embedding_matrix, question)
        support_embedding = tf.nn.embedding_lookup(self.support_embedding_matrix, support)

        model_kwargs = {
            'sequence1': question_embedding,
            'sequence1_length': question_length,
            'sequence2': support_embedding,
            'sequence2_length': support_length,
            'representation_size': 200,
            'dropout_keep_prob': 1.0 - shared_resources.config.get('dropout', 0),
            'use_masking': True,
            'prepend_null_token': True
        }

        from jtr.tasks.mcqa.dam import FeedForwardDAMP
        model = FeedForwardDAMP(**model_kwargs)
        logits = model()
        return logits


class ESIMModel(AbstractSingleSupportFixedClassModel):
    def forward_pass(self, shared_resources,
                     question, question_length,
                     support, support_length,
                     num_classes):
        # final states_fw_bw dimensions:
        # [[[batch, output dim], [batch, output_dim]]
        support = tf.squeeze(support, 1)
        support_length = tf.squeeze(support_length, 1)

        question_embedding = tf.nn.embedding_lookup(self.question_embedding_matrix, question)
        support_embedding = tf.nn.embedding_lookup(self.support_embedding_matrix, support)

        model_kwargs = {
            'sequence1': question_embedding,
            'sequence1_length': question_length,
            'sequence2': support_embedding,
            'sequence2_length': support_length,
            'representation_size': shared_resources.config.get('repr_dim', 300),
            'dropout_keep_prob': 1.0 - shared_resources.config.get('dropout', 0),
            'use_masking': True
        }

        from jtr.tasks.mcqa.esim import ESIM
        model = ESIM(**model_kwargs)
        logits = model()
        return logits


class EmptyOutputModule(OutputModule):

    def __init__(self):
        self.setup()

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits,
                Ports.Prediction.candidate_index,
                Ports.Target.target_index]

    def __call__(self, inputs: List[QASetting], *tensor_inputs: np.ndarray) -> List[Answer]:
        return tensor_inputs

    def setup(self):
        pass

    def store(self, path):
        pass

    def load(self, path):
        pass


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
