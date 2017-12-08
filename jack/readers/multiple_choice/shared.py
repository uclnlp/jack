# -*- coding: utf-8 -*-

from abc import ABCMeta
from typing import NamedTuple

import progressbar

from jack.core import *
from jack.core.data_structures import *
from jack.core.tensorflow import TFModelModule
from jack.readers.multiple_choice import util
from jack.util import preprocessing
from jack.util.map import numpify

logger = logging.getLogger(__name__)


class SingleSupportFixedClassForward(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward_pass(self, shared_resources, embedded_question, embedded_support, num_classes, tensors):
        """Takes a single support and question and produces logits"""
        raise NotImplementedError


class AbstractSingleSupportMCModel(TFModelModule, SingleSupportFixedClassForward):
    def __init__(self, shared_resources):
        self.shared_resources = shared_resources
        self.vocab = self.shared_resources.vocab
        self.config = self.shared_resources.config
        super(AbstractSingleSupportMCModel, self).__init__(shared_resources)

    @property
    def input_ports(self) -> List[TensorPort]:
        if self.shared_resources.config.get("vocab_from_embeddings", False):
            return [Ports.Input.emb_support, Ports.Input.emb_question,
                    Ports.Input.support_length, Ports.Input.question_length,
                    # character information
                    Ports.Input.word_chars, Ports.Input.word_char_length,
                    Ports.Input.question_words, Ports.Input.support_words,
                    Ports.is_eval]
        else:
            return [Ports.Input.support, Ports.Input.question,
                    Ports.Input.support_length, Ports.Input.question_length,
                    # character information
                    Ports.Input.word_chars, Ports.Input.word_char_length,
                    Ports.Input.question_words, Ports.Input.support_words,
                    Ports.is_eval]

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

    def create_output(self, shared_resources: SharedResources, input_tensors) -> Mapping[TensorPort, tf.Tensor]:
        vocab_from_embeddings = self.shared_resources.config.get("vocab_from_embeddings", False)
        support_port = Ports.Input.emb_support if vocab_from_embeddings else Ports.Input.support
        question_port = Ports.Input.emb_question if vocab_from_embeddings else Ports.Input.question
        support = input_tensors[support_port]
        question = input_tensors[question_port]
        input_size = shared_resources.config['repr_dim_input']
        tensors = TensorPortTensors(input_tensors)
        if not shared_resources.config.get("vocab_from_embeddings", False):
            if hasattr(shared_resources, 'embeddings'):
                e = tf.constant(shared_resources.embeddings, tf.float32)
            else:
                vocab_size = len(shared_resources.vocab)
                e = tf.get_variable("embeddings", [vocab_size, input_size],
                                    initializer=tf.random_normal_initializer(0.0, 0.1),
                                    trainable=True, dtype="float32")

            embedded_question = tf.nn.embedding_lookup(e, question)
            embedded_support = tf.nn.embedding_lookup(e, support)
        else:
            embedded_question = question
            embedded_support = support

        embedded_question.set_shape([None, None, input_size])
        embedded_support.set_shape([None, None, input_size])

        logits = self.forward_pass(shared_resources, embedded_question, embedded_support,
                                   shared_resources.config['answer_size'], tensors)

        predictions = tf.argmax(logits, 1, name='prediction')

        return {
            Ports.Prediction.logits: logits,
            Ports.Prediction.candidate_index: predictions
        }

    def create_training_output(self, shared_resources: SharedResources, input_tensors):
        tensors = TensorPortTensors(input_tensors)
        return {
            Ports.loss: tf.losses.sparse_softmax_cross_entropy(logits=tensors.logits, labels=tensors.target_index)
        }


MCAnnotation = NamedTuple('MCAnnotation', [
    ('question_tokens', List[str]),
    ('question_ids', List[int]),
    ('question_length', int),
    ('support_tokens', List[List[str]]),
    ('support_ids', List[List[int]]),
    ('support_length', List[int]),
    ('answer', Optional[int]),
    ('id', Optional[int]),
])


class MultipleChoiceSingleSupportInputModule(OnlineInputModule[MCAnnotation]):
    def __init__(self, shared_resources):
        self.shared_resources = shared_resources

    @property
    def training_ports(self) -> List[TensorPort]:
        return [Ports.Target.target_index]

    @property
    def output_ports(self) -> List[TensorPort]:
        """Defines the outputs of the InputModule"""
        if self.shared_resources.config.get("vocab_from_embeddings", False):
            return [Ports.Input.emb_support,
                    Ports.Input.emb_question, Ports.Input.support_length,
                    Ports.Input.question_length, Ports.Input.sample_id,
                    # character information
                    Ports.Input.word_chars, Ports.Input.word_char_length,
                    Ports.Input.question_words, Ports.Input.support_words,
                    Ports.is_eval]
        else:
            return [Ports.Input.support,
                    Ports.Input.question, Ports.Input.support_length,
                    Ports.Input.question_length, Ports.Input.sample_id,
                    # character information
                    Ports.Input.word_chars, Ports.Input.word_char_length,
                    Ports.Input.question_words, Ports.Input.support_words,
                    Ports.is_eval]

    def preprocess(self, questions: List[QASetting],
                   answers: Optional[List[List[Answer]]] = None,
                   is_eval: bool = False) -> List[MCAnnotation]:
        if answers is None:
            answers = [None] * len(questions)
        preprocessed = []
        if len(questions) > 1000:
            bar = progressbar.ProgressBar(
                max_value=len(questions),
                widgets=[' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') '])
            for i, (q, a) in bar(enumerate(zip(questions, answers))):
                preprocessed.append(self.preprocess_instance(i, q, a))
        else:
            for i, (q, a) in enumerate(zip(questions, answers)):
                preprocessed.append(self.preprocess_instance(i, q, a))

        return preprocessed

    def preprocess_instance(self, idd: int, question: QASetting,
                            answers: Optional[List[Answer]] = None) -> MCAnnotation:
        has_answers = answers is not None

        q_tokenized, q_ids, q_length, _, _ = preprocessing.nlp_preprocess(
            question.question, self.shared_resources.vocab,
            lowercase=self.shared_resources.config.get('lowercase', True))
        s_tokenized, s_ids, s_length, _, _ = preprocessing.nlp_preprocess(
            question.support[0], self.shared_resources.vocab,
            lowercase=self.shared_resources.config.get('lowercase', True))

        return MCAnnotation(
            question_tokens=q_tokenized,
            question_ids=q_ids,
            question_length=q_length,
            support_tokens=s_tokenized,
            support_ids=s_ids,
            support_length=s_length,
            answer=self.shared_resources.answer_vocab(answers[0].text) if has_answers else 0,
            id=idd
        )

    def create_batch(self, annotations: List[MCAnnotation],
                     is_eval: bool, with_answers: bool) -> Mapping[TensorPort, np.ndarray]:
        # also add character information
        word_chars, word_lengths, tokens, vocab, rev_vocab = \
            preprocessing.unique_words_with_chars(
                [a.question_tokens for a in annotations] + [a.support_tokens for a in annotations],
                self.shared_resources.char_vocab)
        question_words, support_words = tokens[:len(annotations)], tokens[len(annotations):]

        q_lengths = [a.question_length for a in annotations]
        s_lengths = [a.support_length for a in annotations]
        xy_dict = {
            Ports.Input.question_length: q_lengths,
            Ports.Input.support_length: s_lengths,
            Ports.Input.sample_id: [a.id for a in annotations],
            Ports.Input.word_chars: word_chars,
            Ports.Input.word_char_length: word_lengths,
            Ports.Input.question_words: question_words,
            Ports.Input.support_words: support_words,
            Ports.is_eval: is_eval
        }

        if self.shared_resources.config.get("vocab_from_embeddings", False):
            emb_support = np.zeros([len(annotations), max(s_lengths), self.emb_matrix.shape[1]])
            emb_question = np.zeros([len(annotations), max(q_lengths), self.emb_matrix.shape[1]])
            for i, a in enumerate(annotations):
                for j, k in enumerate(a.support_ids):
                    emb_support[i, j] = self._get_emb(k)
                for j, k in enumerate(a.question_ids):
                    emb_question[i, j] = self._get_emb(k)

            xy_dict[Ports.Input.emb_support] = emb_support
            xy_dict[Ports.Input.emb_question] = emb_question
        else:
            xy_dict[Ports.Input.support] = [a.support_ids for a in annotations]
            xy_dict[Ports.Input.question] = [a.question_ids for a in annotations]

        if with_answers:
            xy_dict[Ports.Target.target_index] = [a.answer for a in annotations]
        return numpify(xy_dict)

    def _get_emb(self, idx):
        if idx < self.emb_matrix.shape[0]:
            return self.emb_matrix[idx]
        else:
            return self.default_vec

    def setup(self):
        vocab = self.shared_resources.vocab
        if vocab.emb is not None:
            self.emb_matrix = vocab.emb.lookup
            self.default_vec = np.zeros([vocab.emb_length])

    def setup_from_data(self, data: Iterable[Tuple[QASetting, List[Answer]]]):
        vocab = self.shared_resources.vocab
        if not vocab.frozen:
            preprocessing.fill_vocab(
                (q for q, _ in data), vocab, lowercase=self.shared_resources.config.get('lowercase', True))
            vocab.freeze()
            if vocab.emb is not None:
                self.shared_resources.embeddings = np.zeros([len(vocab), vocab.emb_length])
                for w, i in self.shared_resources.vocab.sym2id.items():
                    e = vocab.emb.get(w)
                    if e is not None:
                        self.shared_resources.embeddings[i] = e

        if not hasattr(self.shared_resources, 'answer_vocab') or not self.shared_resources.answer_vocab.frozen:
            self.shared_resources.answer_vocab = util.create_answer_vocab(answers=(a for _, ass in data for a in ass))
            self.shared_resources.answer_vocab.freeze()
        self.shared_resources.config['answer_size'] = len(self.shared_resources.answer_vocab)
        self.shared_resources.char_vocab = preprocessing.char_vocab_from_vocab(self.shared_resources.vocab)


class SimpleMCOutputModule(OutputModule):
    def __init__(self, shared_resources=None):
        self._shared_resources = shared_resources

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
            if self._shared_resources is not None and hasattr(self._shared_resources, 'answer_vocab'):
                ans = Answer(self._shared_resources.answer_vocab.id2sym[winning_index], score=score)
            else:
                ans = Answer(question.atomic_candidates[winning_index], score=score)
            result.append(ans)
        return result
