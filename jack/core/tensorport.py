"""Tensor ports are used to define 'module signatures'

Modules are loosly coupled with each other via the use of tensor ports. They simply define what kind of tensors are
produced at the input and/or output of each module, thus defining a kind of signature. This allows for maximum
flexibility when (re-)using modules in different combinations.
"""
import logging
from typing import Mapping, Sequence, Any

import numpy as np
import tensorflow as tf

try:
    import torch
except ImportError as e:
    pass

logger = logging.getLogger(__name__)

_allowed_torch = {np.dtype(np.int8), np.dtype(np.int16), np.dtype(np.int32), np.dtype(np.int64),
                  np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64)}


class TensorPort:
    """A TensorPort defines an input or output tensor for a modules.

    A port defines at least a shape, name, and its data type.
    """

    def __init__(self, dtype, shape, name, doc_string=None, shape_string=None):
        """Create a new TensorPort.

        Args:
            dtype: the (numpy) data type of the port.
            shape: the shape of the tensor.
            name: the name of this port (should be a valid TF name)
            doc_string: a documentation string associated with this port
            shape_string: a string of the form [size_1,size_2,size_3] where size_i is a text describing the
                size of the tensor's dimension i (such as "number of batches").
        """
        self.dtype = np.dtype(dtype)
        self.shape = shape
        self.name = name
        self.__doc__ = doc_string
        self.shape_string = shape_string

    def create_tf_placeholder(self):
        """Convenience method that produces a placeholder of the type and shape defined by the port.

        Returns: a placeholder of same type, shape and name.
        """
        return tf.placeholder(tf.as_dtype(self.dtype), self.shape, self.name)

    def create_torch_variable(self, value, gpu=False):
        """Convenience method that produces a tensor given the value of the defined type.

        Returns: a torch tensor of same type.
        """
        if isinstance(value, torch.autograd.Variable):
            if gpu:
                value = value.cuda()
            return value
        if not torch.is_tensor(value):
            if not isinstance(value, np.ndarray):
                value = np.array(value, dtype=self.dtype)
            else:
                value = value.astype(self.dtype)
            if value.size == 0:
                return value

            if self.dtype in _allowed_torch:
                value = torch.autograd.Variable(torch.from_numpy(value))
        else:
            value = torch.autograd.Variable(value)
        if gpu and isinstance(value, torch.autograd.Variable):
            value = value.cuda()
        return value

    @staticmethod
    def torch_to_numpy(value):
        """Convenience method that produces a tensor given the value of the defined type.

        Returns: a torch tensor of same type.
        """
        if isinstance(value, torch.autograd.Variable):
            value = value.data
        if torch.is_tensor(value):
            return value.cpu().numpy()
        elif isinstance(value, np.ndarray):
            return value
        else:
            return np.ndarray(value)

    def get_description(self):
        """Returns a multi-line description string of the TensorPort."""

        return "Tensorport '%s'" % self.name + "\n" + \
               "  dtype: " + str(self.dtype) + "\n" + \
               "  shape: " + str(self.shape) + "\n" + \
               "  doc_string: " + str(self.__doc__) + "\n" + \
               "  shape_string: " + str(self.shape_string)

    def __gt__(self, port):
        return self.name > port.name

    def __repr__(self):
        return "<TensorPort (%s)>" % self.name

    @staticmethod
    def to_mapping(ports: Sequence['TensorPort'], tensors: Sequence[tf.Tensor]):
        """
        Create a dictionary of ports to tensors based on ordered port and tensor sequences.
        Args:
            ports: list of ports
            tensors: list of tensors (same length as ports)

        Returns: mapping from the i-th port to the i-th tensor in the lists.

        """
        return dict(zip(ports, tensors))


class TensorPortWithDefault(TensorPort):
    """
    TensorPort that also defines a default value.
    """

    def __init__(self, default_value, shape, name, doc_string=None, shape_string=None):
        """Default value must be a numpy array."""
        self.default_value = default_value
        super().__init__(default_value.dtype, shape, name, doc_string=doc_string, shape_string=shape_string)

    def create_tf_placeholder(self):
        """Creates a TF placeholder_with_default.

        Convenience method that produces a constant of the type, value and shape defined by the port.
        Returns: a constant tensor of same type, shape and name. It can nevertheless be fed with external values
        as if it was a placeholder.
        """
        ph = tf.placeholder_with_default(self.default_value, self.shape, self.name)
        if ph.dtype != tf.as_dtype(self.dtype):
            logger.warning(
                "Placeholder {} with default of type {} created for TensorPort with type {}!".format(self.name,
                                                                                                     ph.dtype,
                                                                                                     self.dtype))
        return ph

    def create_torch_variable(self, value, gpu=False):
        if value is None:
            value = self.default_value
        return super(TensorPortWithDefault, self).create_torch_variable(value, gpu)


class Ports:
    """Defines sopme common ports for reusability and as examples. Readers can of course define their own.

    This class groups input ports. Different modules can refer to these ports
    to define their input or output, respectively.
    """

    loss = TensorPort(np.float32, [None], "loss",
                      "Represents loss on each instance in the batch",
                      "[batch_size]")
    keep_prob = TensorPortWithDefault(np.array(1.0, np.float32), [], "keep_prob",
                                      "scalar representing keep probability when using dropout",
                                      "[]")
    is_eval = TensorPortWithDefault(np.array(True), [], "is_eval",
                                    "boolean that determines whether input is eval or training.",
                                    "[]")

    class Input:
        question = TensorPort(np.int32, [None, None], "question",
                              "Represents questions using symbol vectors",
                              "[batch_size, max_num_question_tokens]")

        support = TensorPort(np.int32, [None, None], "support",
                             "Represents instances with single support documents",
                             "[batch_size, max_num_tokens]")

        multiple_support = TensorPort(np.int32, [None, None, None], "multiple_support",
                                      ("Represents instances with multiple support documents",
                                       " or single instances with extra dimension set to 1"),
                                      "[batch_size, max_num_support, max_num_tokens]")

        atomic_candidates = TensorPort(np.int32, [None, None], "atomic_candidates",
                                       ("Represents candidate choices using single symbols. ",
                                        "This could be a list of entities from global entities ",
                                        "for example atomic_candidates = [e1, e7, e83] from ",
                                        "global_entities = [e1, e2, e3, ..., eN-1, eN"),
                                       "[batch_size, num_candidates]")

        sample_id = TensorPort(np.int32, [None], "sample_id",
                               "Maps this sample to the index in the input data",
                               "[batch_size]")

        muliple_support_length = TensorPort(np.int32, [None, None], "muliple_support_length",
                                            "Represents length of supports in each support in batch",
                                            "[batch_size, num_supports]")

        support_length = TensorPort(np.int32, [None], "support_length",
                                    "Represents length of supports in batch",
                                    "[batch_size]")

        question_length = TensorPort(np.int32, [None], "question_length",
                                     "Represents length of questions in batch",
                                     "[batch_size]")

        emb_support = TensorPort(np.float32, [None, None, None], "emb_support",
                                      "Represents the embedded support",
                                      "[S, max_num_tokens, N]")

        emb_question = TensorPort(np.float32, [None, None, None], "emb_question",
                                       "Represents the embedded question",
                                       "[Q, max_num_question_tokens, N]")

        # character based information
        word_chars = TensorPort(np.int32, [None, None], "word_chars",
                                "Represents questions using symbol vectors",
                                "[U, max_num_chars]")
        word_char_length = TensorPort(np.int32, [None], "word_char_length",
                                      "Represents questions using symbol vectors",
                                      "[U]")
        question_batch_words = TensorPort(np.int32, [None, None], "question_batch_words",
                                          "Represents question using in-batch vocabulary.",
                                          "[batch_size, max_num_question_tokens]")
        support_batch_words = TensorPort(np.int32, [None, None], "support_batch_words",
                                         "Represents support using in-batch vocabulary",
                                         "[batch_size, max_num_support_tokens]")

        # Number of questions in batch is Q, number of supports is S, number of answers is A, number of candidates is C.
        # Typical input ports such as support, candidates, answers are defined together with individual mapping ports.
        # This allows for more flexibility when numbers can vary between questions.

        support2question = TensorPort(np.int32, [None], "support2question",
                                      "Represents mapping to question idx per support",
                                      "[S]")
        candidate2question = TensorPort(np.int32, [None], "candidate2question",
                                        "Represents mapping to question idx per candidate",
                                        "[C]")
        answer2support = TensorPortWithDefault(np.array([0], np.int32), [None], "answer2support",
                                               "Represents mapping to support idx per answer", "[A]")
        atomic_candidates1D = TensorPort(np.int32, [None], "candidates1D",
                                         "Represents candidate choices using single symbols",
                                         "[C]")

        seq_candidates = TensorPort(np.int32, [None, None], "seq_candidates",
                                    "Represents candidate choices using single symbols",
                                    "[C, max_num_tokens]")

        # MISC intermediate ports that might come in handy
        # -embeddings
        embedded_seq_candidates = TensorPort(np.float32, [None, None, None], "embedded_seq_candidates_flat",
                                             "Represents the embedded sequential candidates",
                                             "[C, max_num_tokens, N]")

        embedded_candidates = TensorPort(np.float32, [None, None], "embedded_candidates_flat",
                                         "Represents the embedded candidates",
                                         "[C, N]")

    class Prediction:
        logits = TensorPort(np.float32, [None, None], "logits",
                            "Represents output scores for each candidate",
                            "[C, num_candidates]")

        candidate_index = TensorPort(np.float32, [None], "candidate_idx",
                                     "Represents answer as a single index",
                                     "[C]")

        candidate_scores = TensorPort(np.float32, [None], "candidate_scores_flat",
                                      "Represents output scores for each candidate",
                                      "[C]")

        candidate_idx = TensorPort(np.float32, [None], "candidate_predictions_flat",
                                   "Represents groundtruth candidate labels, usually 1 or 0",
                                   "[C]")

        # extractive QA
        start_scores = TensorPort(np.float32, [None, None], "start_scores",
                                  "Represents start scores for each support sequence",
                                  "[S, max_num_tokens]")

        end_scores = TensorPort(np.float32, [None, None], "end_scores",
                                "Represents end scores for each support sequence",
                                "[S, max_num_tokens]")

        answer_span = TensorPort(np.int32, [None, 3], "answer_span",
                                 "Represents answer as a (doc_idx, start, end) span", "[A, 3]")

        # generative QA
        generative_symbol_scores = TensorPort(np.int32, [None, None, None], "symbol_scores",
                                              "Represents symbol scores for each possible "
                                              "sequential answer given during training",
                                              "[A, max_num_tokens, vocab_len]")

        generative_symbols = TensorPort(np.int32, [None, None], "symbol_prediction",
                                        "Represents symbol sequence for each possible "
                                        "answer target_indexpredicted by the model",
                                        "[A, max_num_tokens]")

    class Target:
        candidate_1hot = TensorPort(np.float32, [None, None], "candidate_targets",
                                    "Represents target (0/1) values for each candidate",
                                    "[batch_size, num_candidates]")

        target_index = TensorPort(np.int32, [None], "target_index",
                                  ("Represents symbol id of target candidate. ",
                                   "This can either be an index into a full list of candidates,",
                                   " which is fixed, or an index into a partial list of ",
                                   "candidates, for example a list of potential entities ",
                                   "from a list of many candidates"),
                                  "[batch_size]")

        answer_span = TensorPort(np.int32, [None, 2], "answer_span_target",
                                 "Represents answer as a (start, end) span", "[A, 2]")

        seq_answer = TensorPort(np.int32, [None, None], "answer_seq_target",
                                "Represents answer as a sequence of symbols",
                                "[A, max_num_tokens]")

        symbols = TensorPort(np.int32, [None, None], "symbol_targets",
                             "Represents symbols for each possible target answer sequence",
                             "[A, max_num_tokens]")


class TensorPortTensors:
    """
    This class wraps around mappings from tensor ports to tensors and makes the tensors available by
    by `x.foo` instead of `x['foo']` calls.
    """

    def __init__(self, mapping: Mapping[TensorPort, Any]):
        """
        Create a wrapping based on the passed in mapping/dictionary.
        Args:
            mapping: Mapping from ports to tensors.
        """
        self.name_to_tensor = ({key.name: value for key, value in mapping.items()})

    def __getattr__(self, item):
        """
        Returns the tensor belonging to the port with the given name.
        Args:
            item: the tensor port name.

        Returns: the tensor associated with the tensor port of the given name.
        """
        return self.name_to_tensor[item]
