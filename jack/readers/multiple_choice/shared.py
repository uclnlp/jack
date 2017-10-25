# -*- coding: utf-8 -*-

from abc import ABCMeta

from jack.core import *
from jack.core.data_structures import *
from jack.readers.multiple_choice import util
from jack.util import preprocessing
from jack.util.map import numpify


class SingleSupportFixedClassForward(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward_pass(self, shared_resources, embedded_question, q_lengths, embedded_support, s_lengths, num_classes):
        '''Takes a single support and question and produces logits'''
        raise NotImplementedError


class AbstractSingleSupportFixedClassModel(TFModelModule, SingleSupportFixedClassForward):
    def __init__(self, shared_resources):
        self.shared_resources = shared_resources
        self.vocab = self.shared_resources.vocab
        self.config = self.shared_resources.config
        super(AbstractSingleSupportFixedClassModel, self).__init__(shared_resources)

    @property
    def input_ports(self) -> List[TensorPort]:
        if self.shared_resources.config.get("vocab_from_embeddings", False):
            return [Ports.Input.embedded_support, Ports.Input.embedded_question,
                    Ports.Input.support_length, Ports.Input.question_length]
        else:
            return [Ports.Input.support, Ports.Input.question,
                    Ports.Input.support_length, Ports.Input.question_length]

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
        input_size = shared_resources.config['repr_dim_input']
        if not self.shared_resources.config["vocab_from_embeddings"]:
            if hasattr(self.shared_resources, 'embeddings'):
                e = tf.constant(self.shared_resources.embeddings, tf.float32)
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

        logits = self.forward_pass(shared_resources,
                                   embedded_question, question_length,
                                   embedded_support, support_length,
                                   shared_resources.config['answer_size'])

        predictions = tf.argmax(logits, 1, name='prediction')

        return [logits, predictions]

    def create_training_output(self, shared_resources: SharedResources, logits: tf.Tensor, labels: tf.Tensor):
        return tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels),


class SingleSupportFixedClassInputs(OnlineInputModule[Mapping[str, any]]):
    def __init__(self, shared_resources):
        self.shared_resources = shared_resources

    @property
    def training_ports(self) -> List[TensorPort]:
        return [Ports.Target.target_index]

    @property
    def output_ports(self) -> List[TensorPort]:
        """Defines the outputs of the InputModule"""
        if self.shared_resources.config["vocab_from_embeddings"]:
            return [Ports.Input.embedded_support,
                    Ports.Input.embedded_question, Ports.Input.support_length,
                    Ports.Input.question_length, Ports.Input.sample_id]
        else:
            return [Ports.Input.support,
                    Ports.Input.question, Ports.Input.support_length,
                    Ports.Input.question_length, Ports.Input.sample_id]

    def preprocess(self, questions: List[QASetting], answers: Optional[List[List[Answer]]] = None,
                   is_eval: bool = False) -> List[Mapping[str, any]]:
        preprocessed = list()
        for i, qa in enumerate(questions):
            _, token_ids, length, _, _ = preprocessing.nlp_preprocess(
                qa.question, self.shared_resources.vocab, lowercase=self.shared_resources.config.get('lowercase', True))
            _, s_token_ids, s_length, _, _ = preprocessing.nlp_preprocess(
                qa.support[0], self.shared_resources.vocab,
                lowercase=self.shared_resources.config.get('lowercase', True))

            preprocessed.append({
                'supports': s_token_ids,
                'question': token_ids,
                'support_lengths': s_length,
                'question_lengths': length,
                'ids': i,
            })
            if answers is not None:
                preprocessed[-1]["answers"] = self.shared_resources.answer_vocab(answers[i][0].text)

        return preprocessed

    def create_batch(self, annotations: List[Mapping[str, any]],
                     is_eval: bool, with_answers: bool) -> Mapping[TensorPort, np.ndarray]:
        q_lengths = [a["question_lengths"] for a in annotations]
        s_lengths = [a["support_lengths"] for a in annotations]
        if self.shared_resources.config.get("vocab_from_embeddings", False):
            emb_support = np.zeros([len(annotations), max(s_lengths), self.emb_matrix.shape[1]])
            emb_question = np.zeros([len(annotations), max(q_lengths), self.emb_matrix.shape[1]])
            for i, a in enumerate(annotations):
                for j, k in enumerate(a["supports"]):
                    emb_support[i, j] = self._get_emb(k)
                for j, k in enumerate(a["question"]):
                    emb_question[i, j] = self._get_emb(k)

            xy_dict = {
                Ports.Input.embedded_support: emb_support,
                Ports.Input.embedded_question: emb_question,
                Ports.Input.question_length: q_lengths,
                Ports.Input.support_length: s_lengths,
                Ports.Input.sample_id: [a['ids'] for a in annotations]
            }
        else:
            xy_dict = {
                Ports.Input.support: [a["supports"] for a in annotations],
                Ports.Input.question: [a["question"] for a in annotations],
                Ports.Input.question_length: [a["question_lengths"] for a in annotations],
                Ports.Input.support_length: [a['support_lengths'] for a in annotations],
                Ports.Input.sample_id: [a['ids'] for a in annotations]
            }

        if "answers" in annotations[0]:
            xy_dict[Ports.Target.target_index] = [a["answers"] for a in annotations]
        return numpify(xy_dict, keys=[Ports.Input.support, Ports.Input.question,
                                      Ports.Input.question_length, Ports.Input.support_length])

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
            if len(class2idx) < num_classes:
                continue
            if self.i >= self.limit:
                continue
            if right_idx == predicted_idx:
                continue
            score = logits[i][right_idx]
            if self.lower < score < self.upper:
                self.i += 1
                logger.info('Question: {0}'.format(qa.question))
                logger.info('Support: {0}'.format(qa.support[0]))
                logger.info('Answer: {0}'.format(answer.text))
                logger.info('Predicted class: {0}'.format(idx2class[predicted_idx]))

                predictions_str = str([(idx2class[b], a) for a, b in zip(logits[i], range(num_classes))])
                logger.info('Predictions: {0}'.format(predictions_str))

    def setup(self):
        pass

    def store(self, path):
        pass

    def load(self, path):
        pass
