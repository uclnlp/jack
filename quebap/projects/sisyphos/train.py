import tensorflow as tf


#train(loss, optim, train_feed_dicts, max_epochs=1000, hooks=hooks)


def train(loss, optim, batches, placeholders=None, predict=None, max_epochs=10,
          hooks=[], sess=None):
    min_op = optim.minimize(loss)
    if sess is None:
        sess = tf.Session()

    with sess:
        tf.initialize_all_variables().run()

        for i in range(max_epochs):
            for batch in batches:
                if placeholders is None: #then batches should be feed_dict
                    feed_dict = batch
                else:
                    if isinstance(placeholders, dict) and isinstance(batch, dict):
                        feed_dict = {placeholders[k]: batch[k] for k in placeholders}
                    else:
                        feed_dict = dict(zip(placeholders, batch))
                _, current_loss = sess.run([min_op, loss], feed_dict=feed_dict)

                for hook in hooks:
                    hook(sess, i+1, predict, current_loss)

            # calling post-epoch hooks
            for hook in hooks:
                hook(sess, i+1, predict, 0)
