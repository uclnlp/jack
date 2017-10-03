import re

from jack.core import *
from jack.io.embeddings import Vocabulary, Embeddings
from jack.tasks.xqa.shared import XQAPorts, XQAOutputModule
from jack.tasks.xqa.util import prepare_data, stack_and_pad
from jack.tf_fun.rnn import birnn_with_projection
from jack.util import tfutil
from jack.util.map import numpify
from jack.tasks.xqa.util import tokenize

_tokenize_pattern = re.compile('\w+|[^\w\s]')

"""

In this tutorial, we focus on the minimal steps required to implement a new model from scratch using Jack.

We will implement a simple Bi-LSTM baseline for extractive question answering.
The architecture is as follows:
- Words of question and support are embedded using random embeddings (not trained)
- Both word and question are encoded using a bi-directional LSTM
- The question is summarized by averaging its token representations
- A feedforward NN scores each of the support tokens to be the start of the answer
- A feedforward NN scores each of the support tokens to be the end of the answer

In order to implement a Jack reader, we define three modules:
- **Input Module**: Responsible for mapping `QASetting`s to numpy array assoicated with `TensorPort`s
- **Model Module**: Defines the TensorFlow graph
- **Output Module**: Converting the network output to the output of the system. In our case, this involves extracting the answer string from the context. We will use the existing `XQAOutputModule`.


"""


"""

## Ports

All communication between input, model and output modules happens via `TensorPort`s.
Normally, you should try to reuse ports wherever possible to be able to reuse modules as well.
If you need a new port, however, it is also straight-forward to define one.
For this tutorial, we will define most ports here.

"""

embedded_question = TensorPort(tf.float32, [None, None, None], "embedded_question_flat",
                               "Represents the embedded question",
                               "[Q, max_num_question_tokens, N]")

question_length = TensorPort(tf.int32, [None], "question_length_flat",
                             "Represents length of questions in batch",
                             "[Q]")

embedded_support = TensorPort(tf.float32, [None, None, None], "embedded_support_flat",
                              "Represents the embedded support",
                              "[S, max_num_tokens, N]")

support_length = TensorPort(tf.int32, [None], "support_length_flat",
                            "Represents length of support in batch",
                            "[S]")

answer_span = TensorPort(tf.int32, [None, 2], "answer_span_target_flat",
                         "Represents answer as a (start, end) span", "[A, 2]")

"""
In order to reuse the `XQAOutputModule`, we'll use existing ports defined in `XQAPorts` for the `char_token_offset` and the predictions.
We'll also use the `Ports.loss` port, because the the JTR training code expects this port as output of the model module.

"""

print(XQAPorts.token_char_offsets.get_description())
print(XQAPorts.start_scores.get_description())
print(XQAPorts.end_scores.get_description())
print(XQAPorts.span_prediction.get_description())

print(Ports.loss.get_description())



"""
## Input Module

The input module is responsible for converting `QASetting` instances to numpy
arrays, which are mapped to `TensorPort`s. Essentially, we are building a
feed dict used for training and inference.

You could implement the `InputModule` interface, but in many cases it'll be
easier to inherit from `OnlineInputModule`. Doing this, we need to:
- Define the output `TensorPort`s of our input module
- Implement the preprocessing (e.g. tokenization, mapping to embedding vectors, ...). The result of this step is one *annotation* per instance, e.g. a `dict`.
- Implement batching. Given a list of annotations, you need to define how to build the feed dict.

"""


class MyInputModule(OnlineInputModule):

    def __init__(self, shared_resources):
        """The module is initialized with a `shared_resources`.

        For the purpose of this tutorial, we will only use the `vocab` property
        which provides the embeddings. You could also pass arbitrary
        configuration parameters in the `shared_resources.config` dict.
        """
        self.vocab = shared_resources.vocab
        self.emb_matrix = self.vocab.emb.lookup

    # We will now define the input and output TensorPorts of our model.

    @property
    def output_ports(self):
        return [embedded_question,           # Question embeddings
                question_length,             # Lengths of the questions
                embedded_support,            # Support embeddings
                support_length,              # Lengths of the supports
                XQAPorts.token_char_offsets  # Character offsets of tokens in support, used for in ouput module
               ]

    @property
    def training_ports(self):
        return [answer_span]                 # Answer span, one for each question

    # Now, we implement our preprocessing. This involves tokenization,
    # mapping to token IDs, mapping to to token embeddings,
    # and computing the answer spans.

    def _get_emb(self, idx):
        """Maps a token ID to it's respective embedding vector"""
        if idx < self.emb_matrix.shape[0]:
            return self.vocab.emb.lookup[idx]
        else:
            # <OOV>
            return np.zeros([self.vocab.emb_length])

    def preprocess(self, questions, answers=None, is_eval=False):
        """Maps a list of instances to a list of annotations.

        Since in our case, all instances can be preprocessed independently, we'll
        delegate the preprocessing to a `_preprocess_instance()` method.
        """

        if answers is None:
            answers = [None] * len(questions)

        return [self._preprocess_instance(q, a)
                for q, a in zip(questions, answers)]

    def _preprocess_instance(self, question, answers=None):
        """Maps an instance to an annotation.

        An annotation contains the embeddings and length of question and support,
        token offsets, and optionally answer spans.
        """

        has_answers = answers is not None

        # `prepare_data()` handles most of the computation in our case, but
        # you could implement your own preprocessing here.
        q_tokenized, q_ids, q_length, s_tokenized, s_ids, s_length, \
        word_in_question, token_offsets, answer_spans = \
            prepare_data(question, answers, self.vocab,
                         with_answers=has_answers,
                         max_support_length=100)

        # For both question and support, we'll fill an embedding tensor
        emb_support = np.zeros([s_length, self.emb_matrix.shape[1]])
        emb_question = np.zeros([q_length, self.emb_matrix.shape[1]])
        for k in range(len(s_ids)):
            emb_support[k] = self._get_emb(s_ids[k])
        for k in range(len(q_ids)):
            emb_question[k] = self._get_emb(q_ids[k])

        # Now, we build the annotation for the question instance. We'll use a
        # dict that maps from `TensorPort` to numpy array, but this could be
        # any data type, like a custom class, or a named tuple.

        annotation = {
            question_length: q_length,
            embedded_question: emb_question,
            support_length: s_length,
            embedded_support: emb_support,
            XQAPorts.token_char_offsets: token_offsets
        }

        if has_answers:
            # For the purpose of this tutorial, we'll only use the first answer, such
            # that we will have exactly as many answers as questions.
            annotation[answer_span] = list(answer_spans[0])

        return numpify(annotation, keys=[support_length, question_length,
                                         XQAPorts.token_char_offsets, answer_span])

    def create_batch(self, annotations, is_eval, with_answers):
        """Now, we need to implement the mapping of a list of annotations to a feed dict.
        
        Because our annotations already are dicts mapping TensorPorts to numpy
        arrays, we only need to do padding here.
        """

        return {key: stack_and_pad([a[key] for a in annotations])
                for key in annotations[0].keys()}


"""

## Model Module.

The model module defines the TensorFlow computation graph.
It takes input module outputs as inputs and produces outputs such as the loss
and outputs required by hte output module.

"""



class MyModelModule(SimpleModelModule):

    # We'll define a constant here for the hidden size. You could also pass this
    # as part of the `shared_config.config` dict.
    HIDDEN_SIZE = 50

    @property
    def input_ports(self) -> Sequence[TensorPort]:
        return [embedded_question, question_length,
                embedded_support, support_length]

    @property
    def output_ports(self) -> Sequence[TensorPort]:
        return [XQAPorts.start_scores, XQAPorts.end_scores,
                XQAPorts.span_prediction]

    @property
    def training_input_ports(self) -> Sequence[TensorPort]:
        return [XQAPorts.start_scores, XQAPorts.end_scores, answer_span]

    @property
    def training_output_ports(self) -> Sequence[TensorPort]:
        return [Ports.loss]

    def create_output(self, _, emb_question, question_length,
                      emb_support, support_length):
        """
        Implements the "core" model: The TensorFlow subgraph which computes the
        answer span from the embedded question and support.
        Args:
            emb_question: [Q, L_q, N]
            question_length: [Q]
            emb_support: [Q, L_s, N]
            support_length: [Q]

        Returns:
            start_scores [B, L_s, N], end_scores [B, L_s, N], span_prediction [B, 2]
        """

        with tf.variable_scope("fast_qa", initializer=tf.contrib.layers.xavier_initializer()):

            # set shapes for inputs
            emb_question.set_shape([None, None, self.HIDDEN_SIZE])
            emb_support.set_shape([None, None, self.HIDDEN_SIZE])

            # encode question and support
            rnn = tf.contrib.rnn.LSTMBlockFusedCell
            encoded_question = birnn_with_projection(self.HIDDEN_SIZE, rnn, emb_question, question_length,
                                                     projection_scope="question_proj")

            encoded_support = birnn_with_projection(self.HIDDEN_SIZE, rnn, emb_support, support_length,
                                                    share_rnn=True, projection_scope="support_proj")

            start_scores, end_scores, predicted_start_pointer, predicted_end_pointer = \
                self._output_layer(encoded_question, question_length,
                                  encoded_support, support_length)

            span = tf.concat([tf.expand_dims(predicted_start_pointer, 1), tf.expand_dims(predicted_end_pointer, 1)], 1)

            return start_scores, end_scores, span

    def _output_layer(self, encoded_question, question_length, encoded_support,
                     support_length):
        """Output layer of our network:
        - The question is summarized using an attention mechanism (`question_state`).
        - The start scores are predicted via a two-layer NN with the element-wise
          multiplication of the candidate start state and question state as input.
        - Similarly, we'll compute the end scores.
        - Predicted start and end pointers will be determined via greedy search.
        """

        batch_size = tf.shape(question_length)[0]

        # Computing single time attention over question
        attention_scores = tf.contrib.layers.fully_connected(encoded_question, 1,
                                                             scope="question_attention")
        q_mask = tfutil.mask_for_lengths(question_length, batch_size)
        attention_scores = attention_scores + tf.expand_dims(q_mask, 2)
        question_attention_weights = tf.nn.softmax(attention_scores, 1, name="question_attention_weights")
        question_state = tf.reduce_sum(question_attention_weights * encoded_question, [1])

        # Prediction
        # start
        start_input = tf.expand_dims(question_state, 1) * encoded_support

        q_start_inter = tf.contrib.layers.fully_connected(start_input, self.HIDDEN_SIZE,
                                                          activation_fn=tf.nn.relu,
                                                          scope="q_start_inter")

        start_scores = tf.contrib.layers.fully_connected(q_start_inter, 1,
                                                         scope="start_scores")
        start_scores = tf.squeeze(start_scores, [2])

        support_mask = tfutil.mask_for_lengths(support_length, batch_size)
        start_scores = start_scores + support_mask

        predicted_start_pointer = tf.arg_max(start_scores, 1)

        # end
        start_pointer_indices = tf.stack([tf.cast(tf.range(batch_size), tf.int64),
                                          predicted_start_pointer],
                                         axis=1)
        u_s = tf.gather_nd(encoded_support, start_pointer_indices)

        end_input = tf.concat([tf.expand_dims(u_s, 1) * encoded_support, start_input], 2)

        q_end_state = tf.contrib.layers.fully_connected(end_input, self.HIDDEN_SIZE,
                                                        activation_fn=tf.nn.relu,
                                                        scope="q_end")

        end_scores = tf.contrib.layers.fully_connected(tf.nn.relu(q_end_state), 1,
                                                       scope="end_scores")
        end_scores = tf.squeeze(end_scores, [2])
        end_scores = end_scores + support_mask
        predicted_end_pointer = tf.arg_max(end_scores, 1)

        return start_scores, end_scores, predicted_start_pointer, predicted_end_pointer

    def create_training_output(self, _, start_scores, end_scores, answer_span) \
            -> Sequence[TensorPort]:
        """Compute loss from start & end scores and the gold-standard `answer_span`."""

        start, end = [tf.squeeze(t, 1) for t in tf.split(answer_span, 2, 1)]

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=start_scores,
                                                              labels=start) + \
               tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_scores, labels=end)
        return [tf.reduce_mean(loss)]


"""

## Output Module

The output module converts our model predictions to `Answer` instances.
Since our model is a standard extractive QA model and since we used the standard
`TensorPort`s, we can reuse the existing `XQAOutputModule` rather than implementing
our own.

"""


"""

## Training

As a toy example, we'll use da dataset of just one question:

"""

data_set = [
    (QASetting(
        question="Which is it?",
        support=["While b seems plausible, answer a is correct."],
        id="1"),
     [Answer("a", (0, 6, 6))])
]

"""

The `build_vocab()` function builds a random embedding matrix. Normally,
we could load pre-trained embeddings here, such as GloVe.

"""

def build_vocab(questions):
    """Build a vocabulary of random vectors."""

    vocab = dict()
    for question in questions:
        for t in tokenize(question.question):
            if t not in vocab:
                vocab[t] = len(vocab)
    vocab = Vocabulary(vocab)
    embeddings = Embeddings(vocab, np.random.random([len(vocab),
                                                     MyModelModule.HIDDEN_SIZE]))

    vocab = Vocab(emb=embeddings, init_from_embeddings=True)
    return vocab

questions = [q for q, _ in data_set]
shared_resources = SharedResources(build_vocab(questions))

"""

Now, we'll instantiate all modules with the `shared_resources` as parameter.
The `JTReader` needs the three modules and is ready to train!

"""

input_module = MyInputModule(shared_resources)
model_module = MyModelModule(shared_resources)
output_module = XQAOutputModule(shared_resources)
reader = JTReader(shared_resources, input_module, model_module, output_module)

reader.train(tf.train.AdamOptimizer(), data_set, max_epochs=1)

answers = reader(questions)
