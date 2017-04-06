# -*- coding: utf-8 -*-

from jtr.jack.core import *
from jtr.jack.data_structures import *
from jtr.jack.tf_fun import rnn, simple

from jtr.pipelines import pipeline
from jtr.preprocess.batch import get_batches
from jtr.preprocess.map import numpify
from jtr.preprocess.vocab import Vocab
from jtr.jack.preprocessing import preprocess_with_pipeline
from jtr.jack.tasks.mcqa.abstract_multiplechoice import AbstractSingleSupportFixedClassModel

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
    def __init__(self, shared_vocab_config):
        self.shared_vocab_config = shared_vocab_config

    @property
    def training_ports(self) -> List[TensorPort]:
        return [Ports.Input.candidates1d]

    @property
    def output_ports(self) -> List[TensorPort]:
        """Defines the outputs of the InputModule

        1. Word embedding index tensor of questions of mini-batchs
        2. Word embedding index tensor of support of mini-batchs
        3. Max timestep length of mini-batches for support tensor
        4. Max timestep length of mini-batches for question tensor
        5. Labels
        """
        return [Ports.Input.single_support,
                Ports.Input.question, Ports.Input.support_length,
                Ports.Input.question_length, Ports.Targets.candidate_idx, Ports.Input.sample_id]

    def __call__(self, qa_settings: List[QASetting]) \
            -> Mapping[TensorPort, np.ndarray]:
        pass

    def setup_from_data(self, data: List[Tuple[QASetting, List[Answer]]]) -> SharedResources:
        corpus, train_vocab, train_answer_vocab, train_candidate_vocab = \
                preprocess_with_pipeline(data, self.shared_vocab_config.vocab,
                        None, sepvocab=True)
        train_vocab.freeze()
        train_answer_vocab.freeze()
        train_candidate_vocab.freeze()
        self.shared_vocab_config.config['answer_size'] = len(train_answer_vocab)
        self.shared_vocab_config.vocab = train_vocab
        self.answer_vocab = train_answer_vocab


    def dataset_generator(self, dataset: List[Tuple[QASetting, List[Answer]]],
                          is_eval: bool) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        corpus, _, _, _ = \
                preprocess_with_pipeline(dataset,
                        self.shared_vocab_config.vocab, self.answer_vocab, use_single_support=True, sepvocab=True)

        xy_dict = {
            Ports.Input.single_support: corpus["support"],
            Ports.Input.question: corpus["question"],
            Ports.Targets.candidate_idx:  corpus["answers"],
            Ports.Input.question_length : corpus['question_lengths'],
            Ports.Input.support_length : corpus['support_lengths'],
            Ports.Input.sample_id : corpus['ids']
        }

        return get_batches(xy_dict)

    def setup(self):
        pass


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
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=candidate_scores,
                labels=candidate_labels)
        return loss,

    def create_output(self,
                      shared_resources: SharedVocabAndConfig,
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


class PairOfBiLSTMOverSupportAndQuestionModel(AbstractSingleSupportFixedClassModel):
    def forward_pass(self, shared_resources, nvocab,
                     Q, S, Q_lengths, S_lengths,
                     num_classes):
        # final states_fw_bw dimensions:
        # [[[batch, output dim], [batch, output_dim]]

        Q_seq = nvocab(Q)
        S_seq = nvocab(S)

        all_states_fw_bw, final_states_fw_bw = rnn.pair_of_bidirectional_LSTMs(
                Q_seq, Q_lengths, S_seq, S_lengths,
                shared_resources.config['repr_dim'], drop_keep_prob =
                1.0-shared_resources.config['dropout'],
                conditional_encoding=True)

        # ->  [batch, 2*output_dim]
        final_states = tf.concat([final_states_fw_bw[0][1],
                                 final_states_fw_bw[1][1]],axis=1)

        # [batch, 2*output_dim] -> [batch, num_classes]
        outputs = simple.fully_connected_projection(final_states,
                                                         num_classes)

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

