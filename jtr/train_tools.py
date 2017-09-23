import os
import shutil

import tensorflow as tf

from jtr import readers
from jtr.util.hooks import LossHook, ExamplesPerSecHook
from jtr.util.util import Duration
import logging
import sys

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def train_reader(reader, train_data, dev_data, test_data, batch_size, clip_value, dataset_name, debug,
                 epochs, l2, learning_rate, learning_rate_decay, log_interval, model, model_dir, tensorboard_folder,
                 use_streaming, validation_interval, write_metrics_to):
    # build JTReader
    checkpoint = Duration()
    checkpoint()
    learning_rate = tf.get_variable("learning_rate", initializer=learning_rate, dtype=tf.float32,
                                    trainable=False)
    lr_decay_op = learning_rate.assign(learning_rate_decay * learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    if tensorboard_folder is not None:
        if os.path.exists(tensorboard_folder):
            shutil.rmtree(tensorboard_folder)
        sw = tf.summary.FileWriter(tensorboard_folder)
    else:
        sw = None

    # Hooks
    iter_interval = 1 if debug else log_interval
    hooks = [LossHook(reader, iter_interval, summary_writer=sw),
             ExamplesPerSecHook(reader, batch_size, iter_interval, sw)]
    preferred_metric, best_metric = readers.eval_hooks[model].preferred_metric_and_best_score()

    def side_effect(metrics, prev_metric):
        """Returns: a state (in this case a metric) that is used as input for the next call"""
        m = metrics[preferred_metric]
        if prev_metric is not None and m < prev_metric:
            reader.session.run(lr_decay_op)
            logger.info("Decayed learning rate to: %.5f" % reader.session.run(learning_rate))
        elif m > best_metric[0] and model_dir is not None:
            best_metric[0] = m
            if prev_metric is None:  # store whole model only at beginning of training
                reader.store(model_dir)
            else:
                reader.model_module.store(reader.session, os.path.join(model_dir, "model_module"))
            logger.info("Saving model to: %s" % model_dir)
        return m

    # this is the standard hook for the model
    hooks.append(readers.eval_hooks[model](
        reader, dev_data, summary_writer=sw, side_effect=side_effect,
        iter_interval=validation_interval,
        epoch_interval=(1 if validation_interval is None else None),
        write_metrics_to=write_metrics_to,
        dataset_name=dataset_name,
        dataset_identifier=('dev' if use_streaming else None)))
    # Train
    reader.train(optimizer, training_set=train_data,
                 max_epochs=epochs, hooks=hooks,
                 l2=l2, clip=clip_value, clip_op=tf.clip_by_value, dataset_name=dataset_name)
    # Test final model
    if test_data is not None and model_dir is not None:
        test_eval_hook = readers.eval_hooks[model](
            reader, test_data, summary_writer=sw, epoch_interval=1, write_metrics_to=write_metrics_to,
            dataset_name=dataset_name,
            dataset_identifier=('test' if use_streaming else None))

        reader.load(model_dir)
        test_eval_hook.at_test_time(1)
