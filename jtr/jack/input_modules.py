
class QuestionOneSupportGlobalCandiatesInputModule(InputModule):
    def __init__(self, shared_vocab_config):
        assert isinstance(shared_vocab_config, SharedVocabAndConfig), \
            "shared_resources for FastQAInputModule must be an instance of SharedVocabAndConfig"
        self.shared_vocab_config = shared_vocab_config


    def output_ports(self) -> List[TensorPort]:
        """Defines the outputs of the InputModule

        1. Word embedding index tensor of questions of mini-batchs
        2. Word embedding index tensor of support of mini-batchs
        3. Max timestep length of mini-batches
        3. The target class tensor
        """

        # The dimensions are: Time, batch, emb_dim
        self.Q = TensorPort(tf.float32, [None. None, None], 'Q_emb_idx')
        self.S = TensorPort(tf.float32, [None. None, None], 'S_emb_idx')
        # The dimensions are: Time, batch, 1
        self.y = TensorPort(tf.int32, [None. None], 'label')

        # The dimensions are: batch, 1
        self.Q_lengths = TensorPort(tf.int32, [None], 'Q_lengths')
        self.S_lengths = TensorPort(tf.int32, [None], 'S_lengths')

        return [self.Q, self.S, self.y, self.Q_lengths, self.S_lengths]


    def training_ports(self) -> List[TensorPort]:
        # TODO: Bad interface, fix this with composition
        pass

    def __call__(self, qa_settings: List[QASetting])
                    -> Mapping[TensorPort, np.ndarray]:
        """
        Converts a list of inputs into a single batch of tensors, consisting with the `output_ports` of this
        module.
        Args:
            qa_settings: a list of instances (question, support, optional candidates)

        Returns:
            A mapping from ports to tensors.

        """
        Q_tokenized, Q_idx, Q_lengths, S_tokenized, S_idx, S_lengths, \
        _, _, _ = self.prepare_data(
                        qa_settings, self.shared_vocab_config.vocab,
                        with_answers=False)

        unique_words, unique_word_lengths, question2unique, support2unique = self.unique_words(q_tokenized, s_tokenized)

        batch_size = len(qa_settings)
        emb_supports = np.zeros(
                [batch_size, max(s_lengths), super().emb_shapes.shape[1]])
        emb_questions = np.zeros(
                [batch_size, max(q_lengths), super().emb_shapes[1]])

        for i, q in enumerate(q_idx):
            for k, v in enumerate(s_idx[i]):
                emb_supports[i, k] = super().get_emb(v)
            for k, v in enumerate(q):
                emb_questions[i, k] = super().get_emb(v)

        output = {
                self.Q : emb_questions,
                self.S : emb_supports,
                self.y : 'TODO: This is missing'
                self.Q_lengths : Q_lengths,
                self.S_lengths : S_lengths,
        }

        output = numpify(output, keys=[self.Q, self.S, self.y, self.Q_lengths,
            self.S_lengths])

        return output

    @abstractmethod
    def dataset_generator(self,
                          dataset: List[Tuple[QASetting, List[Answer]]],
                          is_eval: bool) -> Iterable[Mapping[TensorPort, np.ndarray]]:
        """
        Given a training set of input-answer pairs, this method produces an iterable/generator
        that when iterated over returns a sequence of batches. These batches map ports to tensors
        just as `__call__` does, but provides additional bindings for the `training_ports` ports in
        case `is_eval` is `False`.
        Args:
            dataset: a set of pairs of input and answer.
            is_eval: is this dataset generated for evaluation only (not training).

        Returns: An iterable/generator that, on each pass through the data, produces a list of batches.
        """
        pass

    @abstractmethod
    def setup_from_data(self, data: List[Tuple[QASetting, List[Answer]]]) -> SharedResources:
        """
        Sets up the module based on input data. This usually involves setting up vocabularies and other
        resources.
        Args:
            data: a set of pairs of input and answer.

        Returns: vocab
        """
        pass

    @abstractmethod
    def setup(self):
        """
        Args:
            shared_resources:
        """
        pass
