# -*- coding: utf-8 -*-
from jack.core import *
from jack.core.data_structures import *
from jack.core.tensorflow import TFModelModule
from jack.util.map import numpify


class KnowledgeGraphEmbeddingInputModule(OnlineInputModule[List[List[int]]]):
    def __init__(self, shared_resources):
        self._kbp_rng = random.Random(123)
        super(KnowledgeGraphEmbeddingInputModule, self).__init__(shared_resources)

    def setup_from_data(self, data: Iterable[Tuple[QASetting, List[Answer]]]):
        triples = [tuple(x[0].question.split()) for x in data]

        entity_set = {s for [s, _, _] in triples} | {o for [_, _, o] in triples}
        predicate_set = {p for [_, p, _] in triples}

        entity_to_index = {entity: index for index, entity in enumerate(entity_set)}
        predicate_to_index = {predicate: index for index, predicate in enumerate(predicate_set)}

        self.shared_resources.entity_to_index = entity_to_index
        self.shared_resources.predicate_to_index = predicate_to_index

    def preprocess(self, questions: List[QASetting], answers: Optional[List[List[Answer]]] = None,
                   is_eval: bool = False) -> List[List[int]]:
        """Converts questions to triples."""
        triples = []
        for qa_setting in questions:
            s, p, o = qa_setting.question.split()
            s_idx = self.shared_resources.entity_to_index.get(s, len(self.shared_resources.entity_to_index))
            o_idx = self.shared_resources.entity_to_index.get(o, len(self.shared_resources.entity_to_index))
            p_idx = self.shared_resources.predicate_to_index[p]
            triples.append([s_idx, p_idx, o_idx])

        return triples

    def create_batch(self, triples: List[List[int]],
                     is_eval: bool, with_answers: bool) -> Mapping[TensorPort, np.ndarray]:
        triples = list(triples)
        if with_answers:
            target = [1] * len(triples)
        if with_answers:
            for i in range(len(triples)):
                s, p, o = triples[i]
                for _ in range(self.shared_resources.config.get('num_negative', 1)):
                    random_subject_index = self._kbp_rng.randint(0, len(self.shared_resources.entity_to_index) - 1)
                    random_object_index = self._kbp_rng.randint(0, len(self.shared_resources.entity_to_index) - 1)
                    triples.append([random_subject_index, p, o])
                    triples.append([s, p, random_object_index])
                    target.append(0)
                    target.append(0)

        xy_dict = {Ports.Input.question: triples}
        if with_answers:
            xy_dict[Ports.Target.target_index] = target
        return numpify(xy_dict)

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Input.question]

    @property
    def training_ports(self) -> List[TensorPort]:
        return [Ports.Target.target_index]


class KnowledgeGraphEmbeddingModelModule(TFModelModule):
    def __init__(self, *args, model_name='DistMult', **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Input.question]

    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits]

    @property
    def training_input_ports(self) -> List[TensorPort]:
        return [Ports.Target.target_index, Ports.Prediction.logits]

    @property
    def training_output_ports(self) -> List[TensorPort]:
        return [Ports.loss]

    def create_training_output(self, shared_resources: SharedResources, input_tensors) \
            -> Mapping[TensorPort, tf.Tensor]:
        tensors = TensorPortTensors(input_tensors)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=tensors.logits,
                                                         labels=tf.to_float(tensors.target_index))
        loss = tf.reduce_mean(losses, axis=0)
        return {Ports.loss: loss}

    def create_output(self, shared_resources: SharedResources, input_tensors) -> Mapping[TensorPort, tf.Tensor]:
        tensors = TensorPortTensors(input_tensors)
        with tf.variable_scope('knowledge_graph_embedding'):
            embedding_size = shared_resources.config['repr_dim']

            nb_entities = len(shared_resources.entity_to_index) + 1  # + 1 to allow for unknown entities
            nb_predicates = len(shared_resources.predicate_to_index)

            entity_embeddings = tf.get_variable('entity_embeddings',
                                                [nb_entities, embedding_size],
                                                dtype='float32')
            predicate_embeddings = tf.get_variable('predicate_embeddings',
                                                   [nb_predicates, embedding_size],
                                                   initializer=tf.contrib.layers.xavier_initializer(),
                                                   dtype='float32')

            subject_idx = tensors.question[:, 0]
            predicate_idx = tensors.question[:, 1]
            object_idx = tensors.question[:, 2]

            subject_emb = tf.nn.embedding_lookup(entity_embeddings, subject_idx, max_norm=1.0)
            predicate_emb = tf.nn.embedding_lookup(predicate_embeddings, predicate_idx)
            object_emb = tf.nn.embedding_lookup(entity_embeddings, object_idx, max_norm=1.0)

            from jack.readers.link_prediction import scores
            assert self.model_name is not None

            model_class = scores.get_function(self.model_name)
            model = model_class(
                subject_embeddings=subject_emb,
                predicate_embeddings=predicate_emb,
                object_embeddings=object_emb)

            logits = model()

        return {
            Ports.Prediction.logits: logits
        }


class KnowledgeGraphEmbeddingOutputModule(OutputModule):
    def setup(self):
        pass

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.logits]

    def __call__(self, inputs: Sequence[QASetting], logits: np.ndarray) -> Sequence[Answer]:
        # len(inputs) == batch size
        # logits: [batch_size, max_num_candidates]
        results = []
        for index_in_batch, question in enumerate(inputs):
            score = logits[index_in_batch]
            results.append(Answer(question.question, score=score))
        return results
