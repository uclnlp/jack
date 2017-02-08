import tensorflow as tf

from typing import List, Sequence

from jtr.jack import QuestionWithDefaults, AnswerWithDefault, SharedVocabAndConfig, JTReader, SimpleModelModule, \
    TensorPort, Ports
from jtr.jack.tasks.mcqa.simple_mcqa import SimpleMCInputModule, SimpleMCOutputModule


class SimpleKBPPorts:
    question_embedding = TensorPort(tf.float32, [None, None],
                                    "question_embedding",
                                    "embedding for a batch of questions",
                                    "[num_questions, emb_dim]")


class SimpleKBPModelModule(SimpleModelModule):
    @property
    def output_ports(self) -> List[TensorPort]:
        return [Ports.Prediction.candidate_scores, SimpleKBPPorts.question_embedding]

    @property
    def training_output_ports(self) -> List[TensorPort]:
        return [Ports.loss]

    @property
    def training_input_ports(self) -> List[TensorPort]:
        return [Ports.Input.atomic_candidates, Ports.Targets.candidate_labels, SimpleKBPPorts.question_embedding]

    @property
    def input_ports(self) -> List[TensorPort]:
        return [Ports.Input.question]

    def create_training_output(self,
                               shared_resources: SharedVocabAndConfig,
                               atomic_candidates: tf.Tensor,
                               candidate_labels: tf.Tensor,
                               question_embedding: tf.Tensor) -> Sequence[tf.Tensor]:
        loss = tf.constant(0.0)
        return loss,

    def create_output(self,
                      shared_resources: SharedVocabAndConfig,
                      question: tf.Tensor) -> Sequence[tf.Tensor]:
        emb_dim = shared_resources.config["emb_dim"]
        with tf.variable_scope("simplce_kbp"):
            # varscope.reuse_variables()
            embeddings = tf.get_variable(
                "embeddings", [len(self.shared_resources.vocab), emb_dim],
                trainable=True, dtype="float32")

            # embedded_supports = tf.reduce_sum(tf.gather(embeddings, multiple_support), (1, 2))  # [batch_size, emb_dim]
            # embedded_question = tf.reduce_sum(tf.gather(embeddings, question), (1,))  # [batch_size, emb_dim]
            # embedded_supports_and_question = embedded_supports + embedded_question
            # embedded_candidates = tf.gather(embeddings, atomic_candidates)  # [batch_size, num_candidates, emb_dim]
            #
            # scores = tf.batch_matmul(embedded_candidates,
            #                          tf.expand_dims(embedded_supports_and_question, -1))
            #
            # squeezed = tf.squeeze(scores, 2)
            return embeddings,


if __name__ == '__main__':
    data_set = [
        (QuestionWithDefaults("which is it?", ["a is true", "b isn't"], atomic_candidates=["a", "b", "c"]),
         AnswerWithDefault("a", score=1.0))
    ]
    questions = [q for q, _ in data_set]

    resources = SharedVocabAndConfig(Vocab(), {"emb_dim": 100})
    example_reader = JTReader(resources,
                              SimpleMCInputModule(resources),
                              SimpleMCModelModule(resources),
                              SimpleMCOutputModule())

    # example_reader.setup_from_data(data_set)

    # todo: chose optimizer based on config
    example_reader.train(tf.train.AdamOptimizer(), data_set, max_epochs=10)

    answers = example_reader(questions)

    print(answers)
