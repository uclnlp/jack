import tensorflow as tf


def train(loss, optim, batches, placeholders=None, predict=None, max_epochs=10,
          hooks=[], pre_run=None, post_run=None, sess=None, l2=0.0, clip=None,
          clip_op=tf.clip_by_value):

    if l2 != 0.0:
        loss = loss + tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * l2

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

    if sess is None:
        sess = tf.Session()

    tf.initialize_all_variables().run(session=sess)

    for i in range(1, max_epochs + 1):
        for j, batch in enumerate(batches):
            if placeholders is not None:
                feed_dict = dict(zip(placeholders, batch))
            else:
                feed_dict = batch

            if pre_run is not None:
                pre_run(sess, i, feed_dict, loss, predict)

            _, current_loss = sess.run([min_op, loss], feed_dict=feed_dict)

            if post_run is not None:
                post_run(sess, i, feed_dict, loss, predict)

            for hook in hooks:
                hook.at_iteration_end(sess, i, predict, current_loss)
                #hook(sess, i, predict, current_loss)

        # calling post-epoch hooks
        for hook in hooks:
            hook.at_epoch_end(sess, i, predict, 0)
            #hook(sess, i, predict, 0, post_epoch=True)