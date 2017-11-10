# -*- coding: utf-8 -*-

from jtr.jack.core import *
from jtr.jack.data_structures import *
from jtr.jack.tf_fun import rnn, simple

from jtr.pipelines import pipeline, jtr_map_to_targets
from jtr.preprocess.batch import get_batches
from jtr.preprocess.map import numpify
from jtr.preprocess.vocabulary import Vocab
from jtr.jack.preprocessing import preprocess_with_pipeline
from jtr.jack.tasks.mcqa.abstract_multiplechoice import AbstractSingleSupportFixedClassModel

from jtr.preprocess.map import tokenize, notokenize, lower, deep_map, deep_seq_map, dynamic_subsample

from typing import List, Tuple, Mapping
import tensorflow as tf
import numpy as np


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
        return [Ports.Targets.candidate_labels]

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
            Ports.Targets.candidate_labels: corpus["targets"]
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


class SingleSupportFixedClassInputs(InputModule):
    def __init__(self, shared_resources):
        self.vocab = shared_resources.vocab
        self.config = shared_resources.config
        self.shared_resources = shared_resources

    @property
    def training_ports(self) -> List[TensorPort]:
        return [Ports.Input.candidates1d]

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Input.single_support,
                Ports.Input.question, Ports.Input.support_length,
                Ports.Input.question_length, Ports.Targets.candidate_idx,
                Ports.Input.sample_id, Ports.Input.keep_prob]

    def __call__(self, qa_settings: List[QASetting]) \
            -> Mapping[TensorPort, np.ndarray]:
        pass

    @staticmethod
    def preprocess(data, lowercase=False, test_time=False, add_lengths=True):
        corpus = {"support": [], "question": [], "candidates": [], "ids": []}
        if not test_time:
            corpus["answers"] = []
        #read data
        for i, xy in enumerate(data):
            if test_time:
                x = xy
                y = None
            else:
                x, y = xy
            corpus["support"].append((x.support)[0])
            corpus["ids"].append(i)
            corpus["question"].append(x.question)
            corpus["candidates"].append(x.atomic_candidates)
            assert len(y) == 1
            if not test_time:
                corpus["answers"].append(y[0].text)
        #preprocessing
        corpus = deep_map(corpus, tokenize, ['question', 'support'])
        if lowercase:
            corpus = deep_seq_map(corpus, lower, ['question', 'support'])
        corpus = deep_seq_map(corpus, lambda xs: ["<SOS>"] + xs + ["<EOS>"], ['question', 'support'])

        #add length
        if add_lengths:
            corpus = deep_seq_map(corpus, lambda xs: len(xs), keys=['question', 'support'],
                                  fun_name='lengths', expand=True)
        return corpus


    def setup_from_data(self, data: List[Tuple[QASetting, List[Answer]]]) -> SharedResources:
        #set config params
        self.setup()
        #preprocess data
        corpus = self.preprocess(data, self.lowercase, test_time=False, add_lengths=False)
        #populate and build vocab
        self.vocab.add(corpus["question"], corpus["support"])
        self.vocab.build(min_freq=self.min_freq, max_size=self.max_size)
        #create answer vocab
        self.answer_vocab = Vocab(unk=None)
        self.answer_vocab.add(corpus["candidates"])
        self.answer_vocab.build()
        logger.debug('answer_vocab: ' + str(self.answer_vocab.sym2id))
        self.config['answer_size'] = len(self.answer_vocab)


    def dataset_generator(self, dataset: List[Tuple[QASetting, List[Answer]]],
                          is_eval: bool, test_time=False) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        corpus = self.preprocess(dataset, self.lowercase, test_time=test_time, add_lengths=True)
        corpus = self.vocab.encode(corpus, keys=['question', 'support'])
        corpus = self.answer_vocab.encode(corpus, keys=['candidates', 'answers'])

        xy_dict = {
            Ports.Input.single_support: corpus['support'],
            Ports.Input.question: corpus['question'],
            Ports.Targets.candidate_idx:  corpus['answers'],
            Ports.Input.question_length: corpus['question_lengths'],
            Ports.Input.support_length: corpus['support_lengths'],
            Ports.Input.sample_id: corpus['ids']
        }

        if is_eval:
            keep_prob_dict = {Ports.Input.keep_prob: 1}
            return get_batches(xy_dict, batch_size=self.eval_batch_size, exact_epoch=True, update=keep_prob_dict)
        else:
            keep_prob_dict = {Ports.Input.keep_prob: 1.-self.dropout}
            return get_batches(xy_dict, batch_size=self.batch_size, update=keep_prob_dict)


    def setup(self):
        self.batch_size = self.config.get("batch_size", 1)
        self.eval_batch_size = self.config.get("eval_batch_size", 256)
        self.dropout = self.config.get("dropout", 0)
        self.lowercase = self.config.get("lowercase", False)
        self.min_freq = self.config.get("vocab_min_freq", 1)
        self.max_size = self.config.get("vocab_max_size", sys.maxsize)
        self.init_embeddings = self.config.get("init_embeddings", 'uniform')
        self.normalize_embeddings = self.config.get("normalize_embeddings", False)


class SimpleMCModelModule(SimpleModelModule):

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.candidate_scores]

    @property
    def training_output_ports(self) -> List[TensorPort]:
        return [Ports.loss]

    @property
    def training_input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.candidate_scores, Ports.Targets.candidate_labels]

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Input.multiple_support, Ports.Input.question, Ports.Input.atomic_candidates]

    def create_training_output(self,
                               shared_resources: SharedVocabAndConfig,
                               candidate_scores: tf.Tensor,
                               candidate_labels: tf.Tensor) -> Sequence[tf.Tensor]:
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=candidate_scores, labels=candidate_labels)
        return loss,

    def create_output(self,
                      shared_resources: SharedVocabAndConfig,
                      multiple_support: tf.Tensor,
                      question: tf.Tensor,
                      atomic_candidates: tf.Tensor) -> Sequence[tf.Tensor]:
        emb_dim = shared_resources.config["repr_dim"]
        with tf.variable_scope("simple_mcqa"):
            # varscope.reuse_variables()
            embeddings = tf.get_variable(
                "embeddings", [len(self.shared_resources.vocab), emb_dim],
                trainable=True, dtype="float32")

            embedded_supports = tf.reduce_sum(tf.gather(embeddings, multiple_support), (1, 2))  # [batch_size, emb_dim]
            embedded_question = tf.reduce_sum(tf.gather(embeddings, question), (1,))  # [batch_size, emb_dim]
            embedded_supports_and_question = embedded_supports + embedded_question
            embedded_candidates = tf.gather(embeddings, atomic_candidates)  # [batch_size, num_candidates, emb_dim]

            scores = tf.matmul(embedded_candidates,
                                     tf.expand_dims(embedded_supports_and_question, -1))

            squeezed = tf.squeeze(scores, 2)
            return squeezed,


class SimpleMCOutputModule(OutputModule):
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


class PairOfBiLSTMOverQuestionAndSupportModel(AbstractSingleSupportFixedClassModel):
    def forward_pass(self, shared_resources, embeddings,
                     Q, S, Q_lengths, S_lengths,
                     num_classes, keep_prob=1):
        # final states_fw_bw dimensions:
        # [[[batch, output dim], [batch, output_dim]]

        Q_seq = tf.nn.embedding_lookup(embeddings, Q)
        S_seq = tf.nn.embedding_lookup(embeddings, S)

        #### OLD signature: ### all_states_fw_bw, final_states_fw_bw = rnn.pair_of_bidirectional_LSTMs(
        lstm1_states, lstm2_states = rnn.pair_of_bidirectional_LSTMs(
                Q_seq, Q_lengths, S_seq, S_lengths,
                shared_resources.config['repr_dim'], drop_keep_prob=keep_prob,
                conditional_encoding=True)
        final_states_lstm2 = lstm2_states['final-states']

        # ->  [batch, 2*output_dim]
        final_states = tf.concat([final_states_lstm2[0][1],
                                  final_states_lstm2[1][1]], axis=1)

        # [batch, 2*output_dim] -> [batch, num_classes]
        outputs = simple.fully_connected_projection(
            final_states, num_classes, name='output_projection')

        return outputs


class EmptyOutputModule(OutputModule):

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.candidate_scores,
                Ports.Prediction.candidate_idx,
                Ports.Targets.candidate_idx]

    def __call__(self, inputs: List[QASetting], *tensor_inputs: np.ndarray) -> List[Answer]:
        return tensor_inputs

    def setup(self):
        pass

    def store(self, path):
        pass

    def load(self, path):
        pass

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, 1).reshape(-1,1))
    return e_x / e_x.sum(1).reshape(-1,1)

class MisclassificationOutputModule(OutputModule):

    def __init__(self, interval, limit=100):
        self.lower, self.upper = interval
        self.limit = limit
        self.i = 0

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.candidate_scores,
                Ports.Prediction.candidate_idx,
                Ports.Targets.candidate_idx,
                Ports.Input.sample_id]

    def __call__(self, inputs: List[QASetting],
            candidate_scores,
            candidate_idx,
            labels,
            sample_ids) -> List[Answer]:
        if self.i >= self.limit: return


        class2idx = {}
        idx2class = {}

        candidate_scores = softmax(candidate_scores)
        num_classes = candidate_scores.shape[1]
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
            score = candidate_scores[i][right_idx]
            if score < self.upper and score > self.lower:
                self.i += 1
                print('#'*75)
                print('Question: {0}'.format(qa.question))
                print('Support: {0}'.format(qa.support[0]))
                print('Answer: {0}'.format(answer.text))
                print('-'*75)
                print('Predicted class: {0}'.format(
                    idx2class[predicted_idx]))
                print('Predictions: {0}'.format(
                    [(idx2class[b], a) for a,b in zip(candidate_scores[i],range(num_classes))]))
                print('#'*75 + '\n')

    def setup(self):
        pass

    def store(self, path):
        pass

    def load(self, path):
        pass

