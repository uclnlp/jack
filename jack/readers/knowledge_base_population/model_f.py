# -*- coding: utf-8 -*-

from random import shuffle
from typing import Any

from jack.core import *
from jack.util.map import numpify, deep_map
from jack.util.preprocessing import transpose_dict_of_lists


class ShuffleList:
    def __init__(self, drawlist, qa):
        assert len(drawlist) > 0
        self.qa = qa
        self.drawlist = drawlist
        shuffle(self.drawlist)
        self.iter = self.drawlist.__iter__()

    def next(self, q):
        try:
            avoided = False
            trial, max_trial = 0, 50
            samp = None
            while not avoided and trial < max_trial:
                samp = next(self.iter)
                trial += 1
                avoided = False if samp in self.qa[q] else True
            return samp
        except:
            shuffle(self.drawlist)
            self.iter = self.drawlist.__iter__()
            return next(self.iter)


def posnegsample(corpus, question_key, answer_key, candidate_key, sl):
    question_dataset = corpus[question_key]
    candidate_dataset = corpus[candidate_key]
    answer_dataset = corpus[answer_key]
    new_candidates = []
    assert (len(candidate_dataset) == len(answer_dataset))
    for i in range(0, len(candidate_dataset)):
        question = question_dataset[i][0]
        answers = [answer_dataset[i]] if not hasattr(answer_dataset[i], '__len__') else answer_dataset[i]
        posneg = [] + answers
        avoided = False
        trial, max_trial = 0, 50
        samp = None
        while (not avoided and trial < max_trial):
            samp = sl.next(question)
            trial += 1
            avoided = False if samp in answers else True
        posneg.append(samp)
        new_candidates.append(posneg)
    result = {}
    result.update(corpus)
    result[candidate_key] = new_candidates
    return result


class ModelFInputModule(OnlineInputModule[Mapping[str, Any]]):
    def __init__(self, shared_resources):
        self.shared_resources = shared_resources

    def setup_from_data(self, data: Iterable[Tuple[QASetting, List[Answer]]]):
        questions, answers = zip(*data)
        self.preprocess(questions, answers)
        self.shared_resources.vocab.freeze()
        return self.shared_resources

    @property
    def training_ports(self) -> List[TensorPort]:
        return [Ports.Target.target_index]

    def preprocess(self, questions: List[QASetting],
                   answers: Optional[List[List[Answer]]] = None,
                   is_eval: bool = False) \
            -> List[Mapping[str, Any]]:

        has_answers = answers is not None
        answers = answers or [None] * len(questions)

        corpus = {"question": [], "candidates": [], "answers": []}
        for x, y in zip(questions, answers):

            corpus["question"].append(x.question)
            corpus["candidates"].append(x.atomic_candidates)

            if y is not None:
                assert len(y) == 1
                corpus["answers"].append([y[0].text])
        corpus = deep_map(corpus, lambda x: [x], ['question'])
        corpus = deep_map(corpus, self.shared_resources.vocab, ['question'])
        corpus = deep_map(corpus, self.shared_resources.vocab, ['candidates'], cache_fun=True)

        if has_answers:
            corpus = deep_map(corpus, self.shared_resources.vocab, ['answers'])
            qanswers = {}
            for i, q in enumerate(corpus['question']):
                q0 = q[0]
                if q0 not in qanswers:
                    qanswers[q0] = set()
                a = corpus["answers"][i][0]
                qanswers[q0].add(a)
            if not is_eval:
                sl = ShuffleList(corpus["candidates"][0], qanswers)
                corpus = posnegsample(corpus, 'question', 'answers', 'candidates', sl)
                # corpus = dynamic_subsample(corpus,'candidates','answers',how_many=1)

        return transpose_dict_of_lists(corpus,
                                       ["question", "candidates"] +
                                       (["answers"] if has_answers else []))

    def create_batch(self, annotations: List[Mapping[str, Any]],
                     is_eval: bool, with_answers: bool) -> Mapping[TensorPort, np.ndarray]:

        output = {
            Ports.Input.question: [a["question"] for a in annotations],
            Ports.Input.atomic_candidates: [a["candidates"] for a in annotations]
        }

        if with_answers:
            output.update({
                Ports.Target.target_index: [a["answers"][0] for a in annotations]
            })
        return numpify(output)

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Input.question, Ports.Input.atomic_candidates]


class ModelFModelModule(TFModelModule):
    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Input.question, Ports.Input.atomic_candidates]

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits, FlatPorts.Misc.embedded_question]

    @property
    def training_input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits, FlatPorts.Misc.embedded_question, Ports.Target.target_index]

    @property
    def training_output_ports(self) -> List[TensorPort]:
        return [Ports.loss]

    def create_training_output(self, shared_resources, logits, embedded_question, target_index) -> Sequence[tf.Tensor]:
        with tf.variable_scope("modelf", reuse=True):
            embeddings = tf.get_variable("embeddings", trainable=True, dtype="float32")
            embedded_answer = tf.expand_dims(tf.nn.sigmoid(tf.gather(embeddings, target_index)),
                                             1)  # [batch_size, 1, repr_dim]
            answer_score = tf.reduce_sum(embedded_question * embedded_answer, 2)  # [batch_size, 1]
            loss = tf.reduce_sum(tf.nn.softplus(logits - answer_score))
        return loss,

    def create_output(self, shared_resources, question, atomic_candidates) -> Sequence[tf.Tensor]:
        repr_dim = shared_resources.config["repr_dim"]
        with tf.variable_scope("modelf"):
            embeddings = tf.get_variable(
                "embeddings",
                trainable=True, dtype="float32",
                initializer=tf.random_uniform([len(shared_resources.vocab), repr_dim], -.1, .1))
            # [batch_size, 1, repr_dim]
            embedded_question = tf.gather(embeddings, question)
            # [batch_size, num_candidates, repr_dim]
            embedded_candidates = tf.nn.sigmoid(tf.gather(embeddings, atomic_candidates))
            # [batch_size, num_candidates]
            logits = tf.reduce_sum(tf.multiply(embedded_candidates, embedded_question), 2)

        return logits, embedded_question


class ModelFOutputModule(OutputModule):
    def __init__(self):
        self.setup()

    def setup(self):
        pass

    @property
    def input_ports(self) -> Sequence[TensorPort]:
        return [Ports.Prediction.logits]

    def __call__(self, inputs: Sequence[QASetting], logits: np.ndarray) -> Sequence[Answer]:
        # len(inputs) == batch size
        # logits: [batch_size, max_num_candidates]
        winning_indices = np.argmax(logits, axis=1)
        result = []
        for index_in_batch, question in enumerate(inputs):
            winning_index = winning_indices[index_in_batch]
            score = logits[index_in_batch, winning_index]
            result.append(Answer(question.atomic_candidates[winning_index], score=score))
        return result
