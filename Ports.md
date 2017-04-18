# Ports

## Introduction
The functional interfaces between modules in JTR are represented and implemented by so called TensorPorts. Much like,
TensorFlow placeholders, they are fully defined by a data-type and a corresponding shape, i.e., a known dimensionality 
of dimensions with potentially unknown size. They also offer the functionality for creating TF placeholders, which 
can also be created with default values in case a value is not always fed by the input module (e.g., a dropout rate 
which is 0.0 by default and is only fed during training). In addition, they should provide a human readable documentation. 
TensorPorts are responsible for defining outputs and expected inputs of JTR modules and can thus be used to validate 
their compatibility when combined to form a JTReader.

## Pre-defined Ports

JTR offers a bunch of pre-defined ports in [core.py](jtr/jack/core.py) (see below for a code snippet) that cover specific QA application scenarios. 
They include ports for multiple-choice- (mcqa), extractive- (xqa) and generative QA (genqa) settings. 

Depending on the 
actual task and setting different ways of treating the input can be imagined. In case of mcqa  with a fixed number C of 
integer candidates, for instance, we would feed candidates for a batch (size B) of instances with a `[B, C]` tensor.
Pre-defined ports for such cases can be found in the **Ports** class of core.py. However, this approach would not work 
anymore in case there is a varying number of candidates for each QA instance in the batch. Modeling such a setting is
possible by simply feeding a *flat* list of candidates in form of a tensor `[SUM(C_i)]` to the model while additionally 
providing a candidate-to-instance mapping *m* of shape `[SUM(C_i)]` that maps each of the candidates to the question 
(index) they belongs to. Although using such **FlatPorts** (see core.py) is the most general way of treating input it 
entails a bit of overhead which the use might want to avoid. Types of Ports can of course be mixed, for instance, a 
QA setting might have a variable number of candidates while always having a fixed number of supporting texts. 

Another categorization that is present in current implementation is the the division of potrs into `Ports.Input`,
`Ports.Prediction` and `Ports.Targets`. These group ports by their application. Input ports define typical inputs 
to a ModelModule within a QA setting while prediction ports should be used to define the output of a ModelModule. 
The target ports are used as typically used as `training_input_ports` of ModelModule or `training_output_ports` of
the InputModule. These should only be provided during training and are not necessary during evaluation or application.
Typically `training_output_ports` consists merely of `Ports.loss`, but they are not restricted to it.


## Conventions

From a user perspective it is advised to reuse as many pre-existing ports as possible. This allows for maximum
re-usability of individual modules (and Hooks). However, users are not bound to any port and can always define their 
own which, in some cases, will even be necessary.


## Existing Implementations

```python
class Ports:
    """
    This class groups input ports. Different modules can refer to these ports
    to define their input or output, respectively.
    """

    loss = TensorPort(tf.float32, [None],
                      "Represents loss on each instance in the batch",
                      "[batch_size]")

    class Input:
        question = TensorPort(tf.int32, [None, None], "question",
                              "Represents questions using symbol vectors",
                              "[batch_size, max_num_question_tokens]")

        single_support = TensorPort(tf.int32, [None, None], "single_support",
                                    "Represents instances with a single support document. ",
                                    "[batch_size, max_num_tokens]")

        multiple_support = TensorPort(tf.int32, [None, None, None], "multiple_support",
                                      "Represents instances with multiple support documents",
                                      "[batch_size, max_num_support, max_num_tokens]")

        atomic_candidates = TensorPort(tf.int32, [None, None], "candidates",
                                       "Represents candidate choices using single symbols",
                                       "[batch_size, num_candidates]")

        sample_id = TensorPort(tf.int32, [None], "sample_id",
                               "Maps this sample to the index in the input text data",
                               "[batch_size]")

        candidates1d = TensorPort(tf.int32, [None], "candidates_idx",
                                  "Represents candidate choices using single symbols",
                                  "[batch_size]")

        keep_prob = TensorPortWithDefault(1.0, tf.float32, [], "keep_prob",
                                          "scalar representing keep probability when using dropout",
                                          "[]")

        is_eval = TensorPortWithDefault(True, tf.bool, [], "is_eval",
                                        "boolean that determines whether input is eval or training.",
                                        "[]")

        support_length = TensorPort(tf.int32, [None], "support_length_flat",
                                    "Represents length of support in batch",
                                    "[S]")

        question_length = TensorPort(tf.int32, [None], "question_length_flat",
                                     "Represents length of questions in batch",
                                     "[Q]")

    class Prediction:
        candidate_scores = TensorPort(tf.float32, [None, None], "candidate_scores",
                                      "Represents output scores for each candidate",
                                      "[batch_size, num_candidates]")

        candidate_index = TensorPort(tf.int32, [None], "candidate_idx",
                                     "Represents answer as a single index",
                                     "[batch_size]")

        candidate_idx = TensorPort(tf.float32, [None], "candidate_predictions_flat",
                                   "Represents groundtruth candidate labels, usually 1 or 0",
                                   "[C]")

    class Targets:
        candidate_labels = TensorPort(tf.float32, [None, None], "candidate_targets",
                                      "Represents target (0/1) values for each candidate",
                                      "[batch_size, num_candidates]")
        target_index = TensorPort(tf.int32, [None], "target_index",
                                  "Represents symbol id of target candidate",
                                  "[batch_size]")

        candidate_idx = TensorPort(tf.int32, [None], "candidate_targets",
                                   "Represents groundtruth candidate labels, usually 1 or 0",
                                   "[C]")


class FlatPorts:
    """
     Number of questions in batch is Q, number of supports is S, number of answers is A, number of candidates is C.
    Typical input ports such as support, candidates, answers are defined together with individual mapping ports. This
    allows for more flexibility when numbers can vary between questions. Naming convention is to use suffix "_flat".
    """

    class Input:
        support_to_question = TensorPort(tf.int32, [None], "support2question",
                                         "Represents mapping to question idx per support",
                                         "[S]")
        candidate_to_question = TensorPort(tf.int32, [None], "candidate2question",
                                           "Represents mapping to question idx per candidate",
                                           "[C]")
        answer2question = TensorPort(tf.int32, [None], "answer2question",
                                     "Represents mapping to question idx per answer",
                                     "[A]")

        support = TensorPort(tf.int32, [None, None], "support_flat",
                             "Represents instances with a single support document. "
                             "[S, max_num_tokens]")

        atomic_candidates = TensorPort(tf.int32, [None], "candidates_flat",
                                       "Represents candidate choices using single symbols",
                                       "[C]")

        seq_candidates = TensorPort(tf.int32, [None, None], "seq_candidates_flat",
                                    "Represents candidate choices using single symbols",
                                    "[C, max_num_tokens]")

        support_length = TensorPort(tf.int32, [None], "support_length_flat",
                                    "Represents length of support in batch",
                                    "[S]")

        question_length = TensorPort(tf.int32, [None], "question_length_flat",
                                     "Represents length of questions in batch",
                                     "[Q]")

    class Prediction:
        candidate_scores = TensorPort(tf.float32, [None], "candidate_scores_flat",
                                      "Represents output scores for each candidate",
                                      "[C]")

        candidate_idx = TensorPort(tf.float32, [None], "candidate_predictions_flat",
                                   "Represents groundtruth candidate labels, usually 1 or 0",
                                   "[C]")

        # extractive QA
        start_scores = TensorPort(tf.float32, [None, None], "start_scores_flat",
                                  "Represents start scores for each support sequence",
                                  "[S, max_num_tokens]")

        end_scores = TensorPort(tf.float32, [None, None], "end_scores_flat",
                                "Represents end scores for each support sequence",
                                "[S, max_num_tokens]")

        answer_span = TensorPort(tf.int32, [None, 2], "answer_span_prediction_flat",
                                 "Represents answer as a (start, end) span", "[A, 2]")

        # generative QA
        generative_symbol_scores = TensorPort(tf.int32, [None, None, None], "symbol_scores",
                                              "Represents symbol scores for each possible "
                                              "sequential answer given during training",
                                              "[A, max_num_tokens, vocab_len]")

        generative_symbols = TensorPort(tf.int32, [None, None], "symbol_prediction",
                                        "Represents symbol sequence for each possible "
                                        "answer target_indexpredicted by the model",
                                        "[A, max_num_tokens]")

    class Target:
        candidate_idx = TensorPort(tf.int32, [None], "candidate_targets_flat",
                                   "Represents groundtruth candidate labels, usually 1 or 0",
                                   "[C]")

        answer_span = TensorPort(tf.int32, [None, 2], "answer_span_target_flat",
                                 "Represents answer as a (start, end) span", "[A, 2]")

        seq_answer = TensorPort(tf.int32, [None, None], "answer_seq_target_flat",
                                "Represents answer as a sequence of symbols",
                                "[A, max_num_tokens]")

        generative_symbols = TensorPort(tf.int32, [None, None], "symbol_targets",
                                        "Represents symbol scores for each possible "
                                        "sequential answer given during training",
                                        "[A, max_num_tokens]")

    class Misc:
        # MISC intermediate ports that might come in handy
        # -embeddings
        embedded_seq_candidates = TensorPort(tf.float32, [None, None, None], "embedded_seq_candidates_flat",
                                             "Represents the embedded sequential candidates",
                                             "[C, max_num_tokens, N]")

        embedded_candidates = TensorPort(tf.float32, [None, None], "embedded_candidates_flat",
                                         "Represents the embedded candidates",
                                         "[C, N]")

        embedded_support = TensorPort(tf.float32, [None, None, None], "embedded_support_flat",
                                      "Represents the embedded support",
                                      "[S, max_num_tokens, N]")

        embedded_question = TensorPort(tf.float32, [None, None, None], "embedded_question_flat",
                                       "Represents the embedded question",
                                       "[Q, max_num_question_tokens, N]")
```
