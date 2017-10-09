import tensorflow as tf

from jack.core import TensorPort


class KBPPorts:
    triple_logits = TensorPort(tf.float32, [None, None], "triple_logits",
                               "Represents output scores for each candidate", "[batch_size]")
