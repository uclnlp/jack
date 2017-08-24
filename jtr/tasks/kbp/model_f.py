# -*- coding: utf-8 -*-

from random import shuffle
from typing import Any

from jtr.core import *
from jtr.util.map import numpify, deep_map, notokenize
from jtr.util.pipelines import transpose_dict_of_lists


class ShuffleList:
    def __init__(self,drawlist,qa):
        assert len(drawlist) > 0
        self.qa = qa
        self.drawlist = drawlist
        shuffle(self.drawlist)
        self.iter = self.drawlist.__iter__()
    def next(self,q):
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


def posnegsample(corpus, question_key, answer_key, candidate_key,sl):
    question_dataset = corpus[question_key]
    candidate_dataset = corpus[candidate_key]
    answer_dataset = corpus[answer_key]
    new_candidates = []
    assert (len(candidate_dataset) == len(answer_dataset))
    for i in range(0, len(candidate_dataset)):
        question = question_dataset[i][0]
        candidates = candidate_dataset[i]
        answers = [answer_dataset[i]] if not hasattr(answer_dataset[i],'__len__') else answer_dataset[i]
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

    def setup_from_data(self, data: Iterable[Tuple[QASetting, List[Answer]]], dataset_name=None, identifier=None):
        questions, answers = zip(*data)
        self.preprocess(questions, answers)
        self.shared_resources.vocab.freeze()
        return self.shared_resources

    @property
    def training_ports(self) -> List[TensorPort]:
        return [Ports.Target.target_index]

    @property
    def batch_size(self):
        return self.shared_resources.config['batch_size']

    def preprocess(self, questions: List[QASetting],
                   answers: Optional[List[List[Answer]]] = None,
                   is_eval: bool = False) \
            -> List[Mapping[str, Any]]:

        assert answers is not None

        corpus = { "question": [], "candidates": [], "answers":[]}
        for x, y in zip(questions, answers):
            corpus["question"].append(x.question)
            corpus["candidates"].append(x.atomic_candidates)
            assert len(y) == 1
            corpus["answers"].append(y[0].text)
        corpus = deep_map(corpus, notokenize, ['question'])
        corpus = deep_map(corpus, self.shared_resources.vocab, ['question'])
        corpus = deep_map(corpus, self.shared_resources.vocab, ['candidates'], cache_fun=True)
        corpus = deep_map(corpus, self.shared_resources.vocab, ['answers'])
        qanswers = {}
        for i,q in enumerate(corpus['question']):
            q0=q[0]
            if q0 not in qanswers:
                qanswers[q0] = set()
            a = corpus["answers"][i]
            qanswers[q0].add(a)
        if not is_eval:
            sl = ShuffleList(corpus["candidates"][0], qanswers)
            corpus = posnegsample(corpus, 'question', 'answers', 'candidates', sl)
            #corpus = dynamic_subsample(corpus,'candidates','answers',how_many=1)

        return transpose_dict_of_lists(corpus, ["question", "candidates", "answers"])

    def create_batch(self, annotations: List[Mapping[str, Any]],
                     is_eval: bool, with_answers: bool) \
            -> Mapping[TensorPort, np.ndarray]:

        output = {
            Ports.Input.question: [a["question"] for a in annotations],
            Ports.Input.atomic_candidates: [a["candidates"] for a in annotations],
            Ports.Target.target_index: [a["answers"] for a in annotations]
        }
        return numpify(output)

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Input.question, Ports.Input.atomic_candidates, Ports.Target.target_index]


class ModelFModelModule(SimpleModelModule):
    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Input.question, Ports.Input.atomic_candidates, Ports.Target.target_index]

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits, Ports.loss]

    @property
    def training_input_ports(self) -> List[TensorPort]:
        return [Ports.loss]

    @property
    def training_output_ports(self) -> List[TensorPort]:
        return [Ports.loss]

    def create_training_output(self,
                               shared_resources: SharedResources,
                               loss: tf.Tensor) -> Sequence[tf.Tensor]:
        return loss,

    def create_output(self,
                      shared_resources: SharedResources,
                      question: tf.Tensor,
                      atomic_candidates: tf.Tensor,
                      target_index: tf.Tensor) -> Sequence[tf.Tensor]:
        repr_dim = shared_resources.config["repr_dim"]
        with tf.variable_scope("modelf"):
            embeddings = tf.get_variable(
                "embeddings",
                trainable=True, dtype="float32",
                initializer=tf.random_uniform([len(shared_resources.vocab), repr_dim],-.1,.1))

            embedded_question = tf.gather(embeddings, question)  # [batch_size, 1, repr_dim]
            embedded_candidates = tf.nn.sigmoid(tf.gather(embeddings, atomic_candidates))  # [batch_size, num_candidates, repr_dim]
            embedded_answer = tf.expand_dims(tf.nn.sigmoid(tf.gather(embeddings, target_index)),1)  # [batch_size, 1, repr_dim]
            #embedded_candidates = tf.gather(embeddings, atomic_candidates)  # [batch_size, num_candidates, repr_dim]
            #embedded_answer = tf.expand_dims(tf.gather(embeddings, target_index),1)  # [batch_size, 1, repr_dim]
            logits = tf.reduce_sum(tf.multiply(embedded_candidates,embedded_question),2) # [batch_size, num_candidates]
            answer_score = tf.reduce_sum(tf.multiply(embedded_question,embedded_answer),2)  # [batch_size, 1]
            loss = tf.reduce_sum(tf.nn.softplus(logits-answer_score))

            return logits, loss


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


class KBPReader(JTReader):
    """
    A Reader reads inputs consisting of questions, supports and possibly candidates, and produces answers.
    It consists of three layers: input to tensor (input_module), tensor to tensor (model_module), and tensor to answer
    (output_model). These layers are called in-turn on a given input (list).
    """

    def train(self, optim,
              training_set: Iterable[Tuple[QASetting, Answer]],
              max_epochs=10, hooks=[],
              l2=0.0, clip=None, clip_op=tf.clip_by_value):
        """
        This method trains the reader (and changes its state).
        Args:
            test_set: test set
            dev_set: dev set
            training_set: the training instances.
            **train_params: parameters to be sent to the training function `jtr.train.train`.

        Returns: None

        """
        assert self.is_train, "Reader has to be created for with is_train=True for training."

        logging.info("Setting up data and model...")
        # First setup shared resources, e.g., vocabulary. This depends on the input module.
        self.setup_from_data(training_set)

        loss = self.model_module.tensors[Ports.loss]

        if l2 != 0.0:
            loss += \
                tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * l2

        if clip is not None:
            gradients = optim.compute_gradients(loss)
            if clip_op == tf.clip_by_value:
                gradients = [(tf.clip_by_value(grad, clip[0], clip[1]), var)
                             for grad, var in gradients]
            elif clip_op == tf.clip_by_norm:
                gradients = [(tf.clip_by_norm(grad, clip), var)
                             for grad, var in gradients]
            min_op = optim.apply_gradients(gradients)
        else:
            min_op = optim.minimize(loss)

        # initialize non model variables like learning rate, optim vars ...
        self.session.run([v.initializer for v in tf.global_variables() if v not in self.model_module.variables])

        logging.info("Start training...")
        for i in range(1, max_epochs + 1):
            batches = self.input_module.batch_generator(training_set, is_eval=False)
            for j, batch in enumerate(batches):
                feed_dict = self.model_module.convert_to_feed_dict(batch)
                _, current_loss = self.session.run([min_op, loss], feed_dict=feed_dict)

                for hook in hooks:
                    hook.at_iteration_end(i, current_loss)

            # calling post-epoch hooks
            for hook in hooks:
                hook.at_epoch_end(i)
