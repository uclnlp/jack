import tensorflow as tf


def train(loss, optim, batches, placeholders=None, predict=None, max_epochs=10,
          hooks=[], pre_run=None, post_run=None, sess=None, l2=0.0, clip=None,
          clip_op=tf.clip_by_value, check_numerics=False):
    """Trains a model which can be decorated with various options.

       Args:
         loss (function): The TensorFlow function for the loss.
         optim (function): Optimizer for the loss function such as Adam.
         batches (list of feed dicts or list of numpy arrays): Input data.
         placeholders (TensorFlow placeholder=None): Needed if batches is a
                      list of arrays.
         predict (function): Function that predicts values via the model.
         max_epochs (int): How often to iterate over the entire data.
         hooks (list of TraceHook interfaces): Hooks are executed at the end of
                an iteration or at the end of an epoch.
         pre_run (function): A function that is execute before each iteration
                 on a batch.
         post_run (function): A function that is execute after each iteration
                  on a batch.
         sess (TensorFlow session): The TensorFlow session object.
         l2 (float): The L2 penalty for the parameters (0.0 == turned off).
         clip (float,float): Tuple for the lower and upper cut-off value for the
              gradient.
         clip_op (TensorFlow clip function): Either clip_by_value, or
                 clip_by_norm. Applies the respective TensowFlow function.

        Returns: None

    """
    if l2 != 0.0:
        loss += \
            tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * l2

    if clip is not None:
        gradients = optim.compute_gradients(loss)
        if clip_op == tf.clip_by_value:
            capped_gradients = [(tf.clip_by_value(grad, clip[0], clip[1]), var)
                                for grad, var in gradients]
        elif clip_op == tf.clip_by_norm:
            capped_gradients = [(tf.clip_by_norm(grad, clip), var)
                                for grad, var in gradients]
        min_op = optim.apply_gradients(capped_gradients)
    else:
        min_op = optim.minimize(loss)

    # Do not take up all the GPU memory, all the time.
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    if sess is None:
        sess = tf.Session(config=sess_config)

    tf.global_variables_initializer().run(session=sess)

    nodes = [min_op, loss]
    if check_numerics:
        nodes.append(tf.add_check_numerics_ops())

    for i in range(1, max_epochs + 1):
        for j, batch in enumerate(batches):
            if placeholders is not None:
                feed_dict = dict(zip(placeholders, batch))
            else:
                feed_dict = batch

            if pre_run is not None:
                pre_run(sess, i, feed_dict, loss, predict)

            result = sess.run(nodes, feed_dict=feed_dict)
            current_loss = result[1]

            if post_run is not None:
                post_run(sess, i, feed_dict, loss, predict)

            for hook in hooks:
                hook.at_iteration_end(sess, i, predict, current_loss, feed_dict)

        # calling post-epoch hooks
        for hook in hooks:
            hook.at_epoch_end(sess, i, predict, 0)
