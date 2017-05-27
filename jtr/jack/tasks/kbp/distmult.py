# -*- coding: utf-8 -*-

from jtr.jack.core import *
from jtr.jack.data_structures import *
from jtr.preprocess.map import numpify

from jtr.preprocess.batch import get_batches
from jtr.pipelines import pipeline

from typing import List, Sequence


class DistMultInputModule(InputModule):
    def __init__(self, shared_resources):
        self.shared_resources = shared_resources

    def setup_from_data(self, data: List[Tuple[QASetting, List[Answer]]]) -> SharedResources:
        print('XXX')

        self.vocab = self.shared_resources.vocab
        self.triples = [x[0].question.split() for x in data]

        self.entity_set = {s for [s, _, _] in self.triples} | {o for [_, _, o] in self.triples}
        self.predicate_set = {p for [_, p, _] in self.triples}

        self.entity_to_index = {entity: index for index, entity in enumerate(self.entity_set)}
        self.predicate_to_index = {predicate: index for index, predicate in enumerate(self.predicate_set)}

        self.shared_resources.config['entity_to_index'] = self.entity_to_index
        self.shared_resources.config['predicate_to_index'] = self.predicate_to_index

        self.shared_resources.vocab.freeze()
        return self.shared_resources

    def setup(self):
        pass

    @property
    def training_ports(self) -> List[TensorPort]:
        return [Ports.Target.target_index]

    def preprocess(self, data, test_time=False):
        corpus = {'support': [], 'question': [], 'candidates': [], 'answers': []}
        for x, y in data:
            corpus["question"].append(x.question)
            assert len(y) == 1
        corpus, _, _, _ = pipeline(corpus, self.vocab, sepvocab=False,
                                   test_time=test_time)
        return corpus

    def dataset_generator(self, dataset: List[Tuple[QASetting, List[Answer]]],
                          is_eval: bool,
                          test_time: bool) -> Iterable[Mapping[TensorPort, np.ndarray]]:
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
        xy_dict = {
            Ports.Input.multiple_support: corpus["support"],
            Ports.Input.question: corpus["question"],
            Ports.Input.atomic_candidates: corpus["candidates"]
        }
        return get_batches(xy_dict)

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Input.question]


class DistMultModelModule(SimpleModelModule):
    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits, Ports.loss]

    @property
    def training_output_ports(self) -> List[TensorPort]:
        return [Ports.loss]

    @property
    def training_input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits, Ports.Target.target_index]

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Input.question]

    def create_training_output(self,
                               shared_resources: SharedResources,
                               logits: tf.Tensor,
                               target_index: tf.Tensor) -> Sequence[tf.Tensor]:
        return [self.loss]

    def create_output(self, shared_resources: SharedResources, question: tf.Tensor) -> Sequence[tf.Tensor]:
        print('YYY')

        with tf.variable_scope('distmult'):
            self.embedding_size = shared_resources.config['repr_dim']

            self.entity_to_index = shared_resources.config['entity_to_index']
            self.predicate_to_index = shared_resources.config['predicate_to_index']

            nb_entities = len(self.entity_to_index)
            nb_predicates = len(self.predicate_to_index)

            self.entity_embeddings = tf.get_variable('entity_embeddings',
                                                     [nb_entities, self.embedding_size],
                                                     initializer=tf.contrib.layers.xavier_initializer(),
                                                     dtype='float32')
            self.predicate_embeddings = tf.get_variable('predicate_embeddings',
                                                        [nb_predicates, self.embedding_size],
                                                        initializer=tf.contrib.layers.xavier_initializer(),
                                                        dtype='float32')

            logits = self.forward_pass(shared_resources, question)

            # TODO XXX
            labels = tf.ones_like(logits)
            self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        return [self.loss, logits]

    def forward_pass(self, shared_resources, question):
        subject_idx = question[:, 0]
        predicate_idx = question[:, 1]
        object_idx = question[:, 2]

        subject_emb = tf.nn.embedding_lookup(self.entity_embeddings, subject_idx)
        predicate_emb = tf.nn.embedding_lookup(self.predicate_embeddings, predicate_idx)
        object_emb = tf.nn.embedding_lookup(self.entity_embeddings, object_idx)
        return tf.reduce_sum(subject_emb * predicate_emb * object_emb, axis=1)


class DistMultOutputModule(OutputModule):
    def setup(self):
        pass

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits]

    def __call__(self, inputs: Sequence[QASetting],
                 *tensor_inputs: np.ndarray) -> Sequence[Answer]:
        return None


class KBPReader(JTReader):
    """
    A Reader reads inputs consisting of questions, supports and possibly candidates, and produces answers.
    It consists of three layers: input to tensor (input_module), tensor to tensor (model_module), and tensor to answer
    (output_model). These layers are called in-turn on a given input (list).
    """

    def train(self, optimizer,
              training_set: List[Tuple[QASetting, Answer]],
              max_epochs=10, hooks=[],
              l2=0.0, clip=None, clip_op=tf.clip_by_value):
        assert self.is_train, "Reader has to be created for with is_train=True for training."

        logger.info("Setting up data and model...")
        # First setup shared resources, e.g., vocabulary. This depends on the input module.
        self.setup_from_data(training_set)

        batches = self.input_module.dataset_generator(training_set, is_eval=False, test_time=False)
        print(batches)

        loss = self.model_module.tensors[Ports.loss]

        if l2:
            loss += tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * l2

        if clip:
            gradients = optimizer.compute_gradients(loss)
            if clip_op == tf.clip_by_value:
                gradients = [(tf.clip_by_value(grad, clip[0], clip[1]), var)
                             for grad, var in gradients]
            elif clip_op == tf.clip_by_norm:
                gradients = [(tf.clip_by_norm(grad, clip), var)
                             for grad, var in gradients]
            min_op = optimizer.apply_gradients(gradients)
        else:
            min_op = optimizer.minimize(loss)

        # initialize non model variables like learning rate, optim vars ...
        self.sess.run([v.initializer for v in tf.global_variables() if v not in self.model_module.variables])

        logger.info("Start training...")
        for i in range(1, max_epochs + 1):
            for j, batch in enumerate(batches):
                feed_dict = self.model_module.convert_to_feed_dict(batch)
                _, current_loss = self.sess.run([min_op, loss], feed_dict=feed_dict)

