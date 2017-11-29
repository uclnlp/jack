# -*- coding: utf-8 -*-

import logging
import math
import os
import random
import shutil

import tensorflow as tf

from jack import readers
from jack.util.hooks import LossHook, ExamplesPerSecHook, ETAHook

logger = logging.getLogger(__name__)


def train(reader, train_data, test_data, dev_data, configuration: dict, debug=False):
    seed = configuration.get('seed')

    # make everything deterministic
    random.seed(seed)
    tf.set_random_seed(seed)

    clip_value = configuration.get('clip_value')
    batch_size = configuration.get('batch_size')
    epochs = configuration.get('epochs')
    l2 = configuration.get('l2')
    optimizer = configuration.get('optimizer')
    learning_rate = configuration.get('learning_rate')
    learning_rate_decay = configuration.get('learning_rate_decay')
    log_interval = configuration.get('log_interval')
    validation_interval = configuration.get('validation_interval')
    tensorboard_folder = configuration.get('tensorboard_folder')
    model = configuration.get('model')
    model_dir = configuration.get('model_dir')
    write_metrics_to = configuration.get('write_metrics_to')

    if clip_value != 0.0:
        clip_value = - abs(clip_value), abs(clip_value)

    learning_rate = tf.get_variable("learning_rate", initializer=learning_rate, dtype=tf.float32, trainable=False)
    lr_decay_op = learning_rate.assign(learning_rate_decay * learning_rate)

    name_to_optimizer = {
        'gd': tf.train.GradientDescentOptimizer,
        'adam': tf.train.AdamOptimizer,
        'adagrad': tf.train.AdagradOptimizer,
        'adadelta': tf.train.AdadeltaOptimizer,
        'rmsprop': tf.train.RMSPropOptimizer
    }

    if optimizer not in name_to_optimizer:
        raise ValueError('Unknown optimizer: {}'.format(optimizer))

    tf_optimizer_class = name_to_optimizer[optimizer]
    tf_optimizer = tf_optimizer_class(learning_rate=learning_rate)

    sw = None
    if tensorboard_folder is not None:
        if os.path.exists(tensorboard_folder):
            shutil.rmtree(tensorboard_folder)
        sw = tf.summary.FileWriter(tensorboard_folder)

    # Hooks
    iter_interval = 1 if debug else log_interval
    hooks = [LossHook(reader, iter_interval, summary_writer=sw),
             ETAHook(reader, iter_interval, int(math.ceil(len(train_data) / batch_size)), epochs),
             ExamplesPerSecHook(reader, batch_size, iter_interval, sw)]

    preferred_metric, best_metric = readers.eval_hooks[model].preferred_metric_and_initial_score()

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
                reader.model_module.store(os.path.join(model_dir, "model_module"))
            logger.info("Saving model to: %s" % model_dir)
        return m

    # this is the standard hook for the model
    hooks.append(readers.eval_hooks[model](
        reader, dev_data, batch_size, summary_writer=sw, side_effect=side_effect,
        iter_interval=validation_interval,
        epoch_interval=(1 if validation_interval is None else None),
        write_metrics_to=write_metrics_to))

    # Train
    reader.train(tf_optimizer, train_data, batch_size, max_epochs=epochs, hooks=hooks,
                 l2=l2, clip=clip_value, clip_op=tf.clip_by_value)

    # Test final model
    if test_data is not None and model_dir is not None:
        test_eval_hook = readers.eval_hooks[model](
            reader, test_data, batch_size, summary_writer=sw, epoch_interval=1, write_metrics_to=write_metrics_to)

        reader.load(model_dir)
        test_eval_hook.at_test_time(1)
