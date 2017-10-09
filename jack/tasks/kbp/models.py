# -*- coding: utf-8 -*-
from jack.core import *
from jack.data_structures import *
from jack.tasks.kbp.shared import KBPPorts
from jack.util.map import numpify


class KnowledgeGraphEmbeddingInputModule(OnlineInputModule[List[List[int]]]):
    def __init__(self, shared_resources):
        self.shared_resources = shared_resources

    def setup_from_data(self, data: Iterable[Tuple[QASetting, List[Answer]]]):
        self.triples = [x[0].question.split() for x in data]

        self.entity_set = {s for [s, _, _] in self.triples} | {o for [_, _, o] in self.triples}
        self.predicate_set = {p for [_, p, _] in self.triples}

        self.entity_to_index = {entity: index for index, entity in enumerate(self.entity_set)}
        self.predicate_to_index = {predicate: index for index, predicate in enumerate(self.predicate_set)}

        self.shared_resources.config['entity_to_index'] = self.entity_to_index
        self.shared_resources.config['predicate_to_index'] = self.predicate_to_index
        return self.shared_resources

    @property
    def training_ports(self) -> List[TensorPort]:
        return []

    def preprocess(self, questions: List[QASetting],
                   answers: Optional[List[List[Answer]]] = None,
                   is_eval: bool = False) -> List[List[int]]:
        """Converts questions to triples."""
        triples = []
        for qa_setting in questions:
            s, p, o = qa_setting.question.split()
            s_idx, o_idx = self.entity_to_index[s], self.entity_to_index[o]
            p_idx = self.predicate_to_index[p]
            triples.append([s_idx, p_idx, o_idx])

        return triples

    def create_batch(self, triples: List[List[int]],
                     is_eval: bool, with_answers: bool) -> Mapping[TensorPort, np.ndarray]:
        batch_size = len(triples)

        xy_dict = {
            Ports.Input.multiple_support: [0] * batch_size,
            Ports.Input.question: triples,
            Ports.Input.atomic_candidates: [0] * batch_size
        }
        return numpify(xy_dict)

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Input.question]


class KnowledgeGraphEmbeddingModelModule(TFModelModule):
    def __init__(self, *args, model_name='DistMult', **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Input.question]

    @property
    def output_ports(self) -> List[TensorPort]:
        return [KBPPorts.triple_logits]

    @property
    def training_input_ports(self) -> List[TensorPort]:
        return [Ports.Input.question, KBPPorts.triple_logits]

    @property
    def training_output_ports(self) -> List[TensorPort]:
        return [Ports.loss, Ports.Prediction.logits]

    def create_training_output(self, shared_resources: SharedResources,
                               question: tf.Tensor, logits: tf.Tensor) -> Sequence[tf.Tensor]:
        positive_labels = tf.ones_like(logits)
        nb_entities = len(self.entity_to_index)

        random_subject_indices = tf.random_uniform(shape=(tf.shape(question)[0], 1),
                                                   minval=0, maxval=nb_entities, dtype=tf.int32)
        random_object_indices = tf.random_uniform(shape=(tf.shape(question)[0], 1),
                                                  minval=0, maxval=nb_entities, dtype=tf.int32)

        # question_corrupted_subjects[:, 0].assign(random_indices)
        question_corrupted_subjects = tf.concat(values=[random_subject_indices, question[:, 1:]], axis=1)
        question_corrupted_objects = tf.concat(values=[question[:, :2], random_object_indices], axis=1)

        negative_subject_logits = self.forward_pass(shared_resources, question_corrupted_subjects)
        negative_object_logits = self.forward_pass(shared_resources, question_corrupted_objects)

        logits = tf.concat(values=[logits, negative_subject_logits, negative_object_logits], axis=0)

        negative_labels = tf.zeros_like(positive_labels)
        labels = tf.concat(values=[positive_labels, negative_labels, negative_labels], axis=0)

        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(losses, axis=0)
        return loss, logits

    def create_output(self, shared_resources: SharedResources, question: tf.Tensor) -> Sequence[tf.Tensor]:
        with tf.variable_scope('knowledge_graph_embedding'):
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

        return logits,

    def forward_pass(self, shared_resources, question):
        subject_idx = question[:, 0]
        predicate_idx = question[:, 1]
        object_idx = question[:, 2]

        subject_emb = tf.nn.embedding_lookup(self.entity_embeddings, subject_idx, max_norm=1.0)
        predicate_emb = tf.nn.embedding_lookup(self.predicate_embeddings, predicate_idx)
        object_emb = tf.nn.embedding_lookup(self.entity_embeddings, object_idx, max_norm=1.0)

        from jack.tasks.kbp import scores
        assert self.model_name is not None

        model_class = scores.get_function(self.model_name)
        model = model_class(
            subject_embeddings=subject_emb,
            predicate_embeddings=predicate_emb,
            object_embeddings=object_emb)

        return model()


class KnowledgeGraphEmbeddingOutputModule(OutputModule):
    def setup(self):
        pass

    @property
    def input_ports(self) -> List[TensorPort]:
        return [KBPPorts.triple_logits]

    def __call__(self, inputs: Sequence[QASetting], logits: np.ndarray) -> Sequence[Answer]:
        # len(inputs) == batch size
        # logits: [batch_size, max_num_candidates]
        results = []
        for index_in_batch, question in enumerate(inputs):
            score = logits[index_in_batch]
            results.append(Answer(None, score=score))
        return results
