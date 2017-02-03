"""
This file contains FastQA specific modules and ports
"""

import random
from collections import defaultdict

import jtr
from jtr.jack import *
from jtr.jack.fun import model_module_factory, no_shared_resources
from jtr.jack.tasks.xqa.shared import XqaPorts
from jtr.jack.tasks.xqa.util import token_to_char_offsets
from jtr.jack.tf_fun.xqa import xqa_min_crossentropy_loss
from jtr.preprocess.batch import GeneratorWithRestart
from jtr.preprocess.map import deep_map, numpify


class FastQAPorts:
    """
    It is good practice define all ports needed for a single model jointly, to get an overview
    """

    # We feed embeddings directly
    emb_question = FlatPorts.Misc.embedded_question
    question_length = FlatPorts.Input.question_length
    emb_support = FlatPorts.Misc.embedded_support
    support_length = FlatPorts.Input.support_length

    # but also ids, for char-based embeddings
    unique_word_chars = TensorPort(tf.int32, [None, None], "question_chars",
                                   "Represents questions using symbol vectors",
                                   "[U, max_num_chars]")
    unique_word_char_length = TensorPort(tf.int32, [None], "question_char_length",
                                         "Represents questions using symbol vectors",
                                         "[U]")
    question_words2unique = TensorPort(tf.int32, [None, None], "question_words2unique",
                                       "Represents support using symbol vectors",
                                       "[batch_size, max_num_question_tokens]")
    support_words2unique = TensorPort(tf.int32, [None, None], "support_words2unique",
                                      "Represents support using symbol vectors",
                                      "[batch_size, max_num_support_tokens, max]")

    keep_prob = Ports.Input.keep_prob
    is_eval = Ports.Input.is_eval

    # This feature is model specific and thus, not part of the conventional Ports
    word_in_question = TensorPort(tf.float32, [None, None], "word_in_question_feature",
                                  "Represents a 1/0 feature for all context tokens denoting"
                                  " whether it is part of the question or not",
                                  "[Q, support_length]")

    correct_start_training = TensorPortWithDefault(np.array([0], np.int64), tf.int64, [None], "correct_start_training",
                                                   "Represents the correct start of the span which is given to the"
                                                   "model during training for use to predicting end.",
                                                   "[A]")

    answer2question_training = TensorPortWithDefault([0], tf.int32, [None], "answer2question_training",
                                                     "Represents mapping to question idx per answer, which is used "
                                                     "together with correct_start_training during training.",
                                                     "[A]")


    # output ports
    start_scores = FlatPorts.Prediction.start_scores
    end_scores = FlatPorts.Prediction.end_scores
    span_prediction = FlatPorts.Prediction.answer_span
    token_char_offsets = XqaPorts.token_char_offsets

    # ports used during training
    answer2question = FlatPorts.Input.answer2question
    answer_span = FlatPorts.Target.answer_span


# FastQA model module factory method, like fastqa.model.fastqa_model
fastqa_with_min_crossentropy_loss =\
    model_module_factory(input_ports=[FastQAPorts.emb_question, FastQAPorts.question_length,
                                      FastQAPorts.emb_support, FastQAPorts.support_length,
                                      # char embedding inputs
                                      FastQAPorts.unique_word_chars, FastQAPorts.unique_word_char_length,
                                      FastQAPorts.question_words2unique, FastQAPorts.support_words2unique,
                                      # feature input
                                      FastQAPorts.word_in_question,
                                      # optional input, provided only during training
                                      FastQAPorts.correct_start_training, FastQAPorts.answer2question_training,
                                      FastQAPorts.keep_prob, FastQAPorts.is_eval],
                         output_ports=[FastQAPorts.start_scores, FastQAPorts.end_scores, FastQAPorts.span_prediction],
                         training_input_ports=[FastQAPorts.start_scores, FastQAPorts.end_scores,
                                               FastQAPorts.answer_span, FastQAPorts.answer2question],
                         training_output_ports=[Ports.loss],
                         training_function=no_shared_resources(xqa_min_crossentropy_loss))


class FastQAInputModule(InputModule):

    def __init__(self, shared_vocab_config):
        assert isinstance(shared_vocab_config, SharedVocabAndConfig), \
            "shared_resources for FastQAInputModule must be an instance of SharedVocabAndConfig"
        self.shared_vocab_config = shared_vocab_config

    def _get_emb(self, idx):
        if idx < self.emb_matrix.shape[0]:
            return self.emb_matrix[idx]
        else:
            return self.default_vec

    @property
    def output_ports(self) -> List[TensorPort]:
        return [FastQAPorts.emb_question, FastQAPorts.question_length,
                FastQAPorts.emb_support, FastQAPorts.support_length,
                # char
                FastQAPorts.unique_word_chars, FastQAPorts.unique_word_char_length,
                FastQAPorts.question_words2unique, FastQAPorts.support_words2unique,
                # features
                FastQAPorts.word_in_question,
                # optional, only during training
                FastQAPorts.correct_start_training, FastQAPorts.answer2question_training,
                FastQAPorts.keep_prob, FastQAPorts.is_eval,
                # for output module
                FastQAPorts.token_char_offsets]

    @property
    def training_ports(self) -> List[TensorPort]:
        return [FastQAPorts.answer_span, FastQAPorts.answer2question]

    def setup_from_data(self, data: List[Tuple[Question, List[Answer]]]) -> SharedResources:
        # create character vocab + word lengths + char ids per word
        vocab = self.shared_vocab_config.vocab
        char_vocab = dict()
        char_vocab["PAD"] = 0
        for i in range(max(vocab.id2sym.keys())+1):
            w = vocab.id2sym.get(i)
            if w is not None:
                for c in w:
                    if c not in char_vocab:
                        char_vocab[c] = len(char_vocab)
        self.shared_vocab_config.config["char_vocab"] = char_vocab
        # Assumes that vocab and embeddings are given during creation
        self.setup(self.shared_vocab_config)
        return self.shared_vocab_config

    def setup(self, shared_vocab_config: SharedResources):
        assert isinstance(shared_vocab_config, SharedVocabAndConfig), \
            "shared_resources for FastQAInputModule must be an instance of SharedVocabAndConfig"
        self.shared_vocab_config = shared_vocab_config
        vocab = shared_vocab_config.vocab
        config = shared_vocab_config.config
        self.batch_size = config.get("batch_size", 1)
        self.dropout = config.get("dropout", 1)
        self._rng = random.Random(config.get("seed", 123))
        self.emb_matrix = vocab.emb.lookup
        self.default_vec = np.zeros([vocab.emb_length])
        self.char_vocab = self.shared_vocab_config.config["char_vocab"]

    def dataset_generator(self, dataset: List[Tuple[Question, List[Answer]]], is_eval: bool) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        corpus = {"support": [], "support_lengths": [], "question": [], "question_lengths": []}

        for input, answers in dataset:
            corpus["support"].append(" ".join(input.support))
            corpus["question"].append(input.question)

        corpus = deep_map(corpus, jtr.preprocess.map.tokenize, ['question', 'support'])
        word_in_question = []

        token_offsets = []
        answer_spans = []
        for i, (q, s) in enumerate(zip(corpus["question"], corpus["support"])):
            input, answers = dataset[i]
            corpus["support_lengths"].append(len(s))
            corpus["question_lengths"].append(len(q))

            # char to token offsets
            support = " ".join(input.support)
            offsets = token_to_char_offsets(support, s)
            token_offsets.append(offsets)
            spans = []
            for a in answers:
                start = 0
                while start < len(offsets) and offsets[start] < a.span[0]:
                    start += 1

                if start == len(offsets):
                    continue

                end = start
                while end+1 < len(offsets) and offsets[end+1] < a.span[1]:
                    end += 1
                if (start, end) not in spans:
                    spans.append((start, end))
            answer_spans.append(spans)

            # word in question feature
            wiq = []
            for token in s:
                wiq.append(float(token in q))
            word_in_question.append(wiq)

        emb_supports = np.zeros([self.batch_size, max(corpus["support_lengths"]), self.emb_matrix.shape[1]])
        emb_questions = np.zeros([self.batch_size, max(corpus["question_lengths"]), self.emb_matrix.shape[1]])

        corpus_ids = deep_map(corpus, self.shared_vocab_config.vocab, ['question', 'support'])

        def batch_generator():
            todo = list(range(len(corpus_ids["question"])))
            self._rng.shuffle(todo)
            while todo:
                question = list()
                support = list()
                support_lengths = list()
                question_lengths = list()
                wiq = list()
                spans = list()
                span2question = []
                offsets = []
                unique_words_set = dict()
                unique_words = list()
                unique_word_lengths = list()
                question2unique = list()
                support2unique = list()

                # we have to create batches here and cannot precompute them because of the batch-specific wiq feature
                for i, j in enumerate(todo[:self.batch_size]):
                    q2u = list()
                    for w in corpus["question"][j]:
                        if w not in unique_words_set:
                            unique_word_lengths.append(len(w))
                            unique_words.append([self.char_vocab.get(c, 0) for c in w])
                            unique_words_set[w] = len(unique_words_set)
                        q2u.append(unique_words_set[w])
                    question2unique.append(q2u)
                    s2u = list()
                    for w in corpus["support"][j]:
                        if w not in unique_words_set:
                            unique_word_lengths.append(len(w))
                            unique_words.append([self.char_vocab.get(c, 0) for c in w])
                            unique_words_set[w] = len(unique_words_set)
                        s2u.append(unique_words_set[w])
                    support2unique.append(s2u)

                    question.append(corpus_ids["question"][j])
                    support.append(corpus_ids["support"][j])
                    for k in range(len(support[-1])):
                        emb_supports[i, k] = self._get_emb(support[-1][k])
                    for k in range(len(question[-1])):
                        emb_supports[i, k] = self._get_emb(question[-1][k])
                    support_lengths.append(corpus["support_lengths"][j])
                    question_lengths.append(corpus["question_lengths"][j])
                    spans.extend(answer_spans[j])
                    span2question.extend(i for _ in answer_spans[j])
                    wiq.append(word_in_question[j])
                    offsets.append(token_offsets[j])

                batch_size = len(question_lengths)

                output = {
                    FastQAPorts.unique_word_chars: unique_words,
                    FastQAPorts.unique_word_char_length: unique_word_lengths,
                    FastQAPorts.question_words2unique: question2unique,
                    FastQAPorts.support_words2unique: support2unique,
                    FastQAPorts.emb_support: emb_supports[:batch_size, :max(support_lengths), :],
                    FastQAPorts.support_length: support_lengths,
                    FastQAPorts.emb_question: emb_questions[:batch_size, :max(question_lengths), :],
                    FastQAPorts.question_length: question_lengths,
                    FastQAPorts.word_in_question: wiq,
                    FastQAPorts.answer_span: spans,
                    FastQAPorts.correct_start_training: [] if is_eval else [s[0] for s in spans],
                    FastQAPorts.answer2question: span2question if is_eval else list(range(len(span2question))),
                    FastQAPorts.answer2question_training: [] if is_eval else span2question,
                    FastQAPorts.keep_prob: 0.0 if is_eval else 1 - self.dropout,
                    FastQAPorts.is_eval: is_eval,
                    FastQAPorts.token_char_offsets: offsets
                }

                # we can only numpify in here, because bucketing is not possible prior
                batch = numpify(output, keys=[FastQAPorts.unique_word_chars,
                                              FastQAPorts.question_words2unique, FastQAPorts.support_words2unique,
                                              FastQAPorts.word_in_question, FastQAPorts.token_char_offsets])
                todo = todo[self.batch_size:]
                yield batch

        return GeneratorWithRestart(batch_generator)

    def __call__(self, inputs: List[Question]) -> Mapping[TensorPort, np.ndarray]:
        corpus = {"support": [], "support_lengths": [], "question": [], "question_lengths": []}
        for input in inputs:
            corpus["support"].append(" ".join(input.support))
            corpus["question"].append(input.question)

        corpus = deep_map(corpus, jtr.preprocess.map.tokenize, ['question', 'support'])
        word_in_question = []
        token_offsets = []
        unique_words_set = dict()
        unique_words = list()
        unique_word_lengths = list()
        question2unique = list()
        support2unique = list()

        for q, s, input in zip(corpus["question"], corpus["support"], inputs):
            corpus["support_lengths"].append(len(s))
            corpus["question_lengths"].append(len(q))
            q2u = list()
            for w in q:
                if w not in unique_words_set:
                    unique_word_lengths.append(len(w))
                    unique_words.append([self.char_vocab.get(c, 0) for c in w])
                    unique_words_set[w] = len(unique_words_set)
                q2u.append(unique_words_set[w])
            question2unique.append(q2u)
            s2u = list()
            for w in s:
                if w not in unique_words:
                    unique_word_lengths.append(len(w))
                    unique_words.append([self.char_vocab.get(c, 0) for c in w])
                    unique_words_set[w] = len(unique_words_set)
                s2u.append(unique_words_set[w])
            support2unique.append(s2u)

            # char to token offsets
            offsets = token_to_char_offsets(input.support[0], s)
            token_offsets.append(offsets)

            # word in question feature
            wiq = []
            for token in s:
                wiq.append(float(token in q))
            word_in_question.append(wiq)

        batch_size = len(inputs)
        emb_supports = np.zeros([batch_size, max(corpus["support_lengths"]), self.emb_matrix.shape[1]])
        emb_questions = np.zeros([batch_size, max(corpus["question_lengths"]), self.emb_matrix.shape[1]])

        corpus_ids = deep_map(corpus, self.shared_vocab_config.vocab, ['question', 'support'])

        for i, q in enumerate(corpus_ids["question"]):
            for k, v in enumerate(corpus_ids["support"][i]):
                emb_supports[i, k] = self._get_emb(v)
            for k, v in enumerate(q):
                emb_questions[i, k] = self._get_emb(v)

        output = {
            FastQAPorts.unique_word_chars: unique_words,
            FastQAPorts.unique_word_char_length: unique_word_lengths,
            FastQAPorts.question_words2unique: question2unique,
            FastQAPorts.support_words2unique: support2unique,
            FastQAPorts.emb_support: emb_supports,
            FastQAPorts.support_length: corpus["support_lengths"],
            FastQAPorts.emb_question: emb_questions,
            FastQAPorts.question_length: corpus["question_lengths"],
            FastQAPorts.word_in_question: word_in_question,
            FastQAPorts.token_char_offsets: token_offsets
        }

        output = numpify(output, keys=[FastQAPorts.unique_word_chars, FastQAPorts.question_words2unique,
                                       FastQAPorts.support_words2unique,FastQAPorts.word_in_question,
                                       FastQAPorts.token_char_offsets])

        return output