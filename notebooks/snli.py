# First change dir to JTR parent
import os
import sys
import logging
from jtr.jack.tasks.mcqa.simple_mcqa import MisclassificationOutputModule
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(sys.argv[0]))
os.chdir('..')

import jtr.jack.readers as readers
print("Existing models:\n%s" % ", ".join(readers.readers.keys()))

from jtr.preprocess.vocab import Vocab
# Create example reader with a basic config
embedding_dim = 128
config = {"batch_size": 128, "repr_dim": 128, "repr_dim_input": embedding_dim,
        'dropout' : 0.0}
reader = readers.readers["snli_reader"](Vocab(), config)

# Loaded some test data to work on
# This loads train, dev, and test data of sizes (2k, 1k, 1k)
from jtr.jack.core import TestDatasets
train, dev, test = TestDatasets.generate_SNLI()

# We creates hooks which keep track of the loss
# We also create 'the standard hook' for our model
from jtr.jack.train.hooks import LossHook, ClassificationEvalHook
hooks = [LossHook(reader, iter_interval=25)]
         #readers.eval_hooks['snli_reader'](reader, dev, iter_interval=25)]


# Here we initialize our optimizer
# we choose Adam with standard momentum values and learning rate 0.001
import tensorflow as tf
learning_rate = 0.001
optim = tf.train.AdamOptimizer(learning_rate)

# Lets train the reader on the CPU for 2 epochs
reader.train(optim, train,
             hooks=hooks, max_epochs=30,
             device='/cpu:0', dev_set = dev)

#hooks[0].plot()
#hooks[1].plot(ylim=[0.0, 1.0])

reader.output_module = MisclassificationOutputModule(interval=[0.0, 0.20], limit=10)
reader.process_outputs(test)



