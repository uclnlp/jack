import tensorflow as tf
import numpy as np
import time
from time import gmtime, strftime, localtime
from sklearn.metrics import classification_report


# todo: hooks should also have prefixes so that one can use the same hook with different parameters
class Hook(object):
    def __init__(self):
        raise NotImplementedError

    def __call__(self, sess, epoch, model, loss):
        raise NotImplementedError


class TraceHook(object):
    def __init__(self, summary_writer=None):
        self.summary_writer = summary_writer

    def __tag__(self):
        raise NotImplementedError

    def __call__(self, sess, epoch, model, loss):
        raise NotImplementedError

    def update_summary(self, sess, current_step, title, value):
        if self.summary_writer is not None:
            cur_summary = tf.scalar_summary(title, value)
            # if you are using some summaries, merge them
            merged_summary_op = tf.merge_summary([cur_summary])
            summary_str = sess.run(merged_summary_op)
            self.summary_writer.add_summary(summary_str, current_step)


class LossHook(TraceHook):
    def __init__(self, iter_interval, batch_size, summary_writer=None):
        super(LossHook, self).__init__(summary_writer)
        self.iter_interval = iter_interval
        self.acc_loss = 0
        self.batch_size = batch_size
        self.iter = 0

    def __tag__(self):
        return "Loss"

    def __call__(self, sess, epoch, model, loss):
        self.iter += 1
        self.acc_loss += loss / self.batch_size
        if not self.iter == 0 and self.iter % self.iter_interval == 0:
            loss = self.acc_loss / self.iter_interval
            print("Epoch " + str(epoch) +
                  "\tIter " + str(self.iter) +
                  "\tLoss " + str(loss))
            self.update_summary(sess, self.iter, self.__tag__(), loss)
            self.acc_loss = 0


class TensorHook(TraceHook):
    def __init__(self, iter_interval, tensorlist, feed_dicts=None,
                 summary_writer=None, modes=['mean_abs']):
        """
        Evaluate the tf.Tensor objects in `tensorlist` during training (every `iter_interval` iterations),
        and calculate statistics on them (in `modes`):  'mean_abs', 'std', 'min', and/or 'max'.
        Additionally, the `print` mode prints the entire tensor to stdout.

        If feed_dicts is a generator or iterator over feed_dicts (e.g. to iterate over the entire dev-set),
        each tensor in `tensorlist` is evaluated and concatenated for each feed_dict,
        before calculating the scores for the different `modes`.
        If it's a single feed_dict or `None`, only one evaluation is done.
        """

        super(TensorHook, self).__init__(summary_writer)
        self.iter_interval = iter_interval
        self.tensorlist = tensorlist
        self.feed_dicts = {} if feed_dicts is None else feed_dicts
        self.modes = modes
        self.iter = 0

    def __tag__(self):
        return "Tensor"

    def __call__(self, sess, epoch, model, loss):
        self.iter += 1
        if not self.iter == 0 and self.iter % self.iter_interval == 0:

            for tensor in self.tensorlist:
                tag = tensor.name
                if isinstance(self.feed_dicts, dict):
                    t = sess.run(tensor, feed_dict=self.feed_dicts)
                else:
                    for i, feed_dict in enumerate(self.feed_dicts):
                        t_i = sess.run(tensor, feed_dict=feed_dict)
                        if not hasattr(t_i,'__len__'):
                            t_i = [t_i]
                        t = t_i if i == 0 else np.concatenate([t, t_i], axis=0)
                if 'mean_abs' in self.modes:
                    value_mean = float(np.mean(t))
                    self.update_summary(sess, self.iter, tag+'_mean_abs', value_mean)
                if 'std' in self.modes:
                    value_std = float(np.std(t))
                    self.update_summary(sess, self.iter, tag+'_std', value_std)
                if 'min' in self.modes:
                    value_min = float(np.min(t))
                    self.update_summary(sess, self.iter, tag+'_min', value_min)
                if 'max' in self.modes:
                    value_max = float(np.max(t))
                    self.update_summary(sess, self.iter, tag + '_max', value_max)
                if 'print' in self.modes: #for debug purposes
                    print('\n%s\n%s\n'%(tag, str(t)))



class SpeedHook(TraceHook):
    def __init__(self, iter_interval, batch_size, summary_writer=None):
        super(SpeedHook, self).__init__(summary_writer)
        self.iter_interval = iter_interval
        self.batch_size = batch_size
        self.t0 = time.time()
        self.num_examples = iter_interval * batch_size
        self.iter = 0

    def __tag__(self):
        return "Speed"

    def __call__(self, sess, epoch, model, loss):
        self.iter += 1
        if not self.iter == 0 and self.iter % self.iter_interval == 0:
            diff = time.time() - self.t0
            speed = "%.2f" % (self.num_examples / diff)
            print("Epoch " + str(epoch) +
                  "\tIter " + str(self.iter) +
                  "\tExamples/s " + str(speed))
            self.update_summary(sess, self.iter, self.__tag__(), float(speed))
            self.t0 = time.time()


class ETAHook(TraceHook):
    def __init__(self, iter_interval, max_epochs, iter_per_epoch,
                 summary_writer=None):
        super(ETAHook, self).__init__(summary_writer)
        self.iter_interval = iter_interval
        self.max_iters = max_epochs * iter_per_epoch
        self.iter = 0
        self.epoch = 1
        self.max_epochs = max_epochs
        self.start = time.time()
        self.reestimate = True

    def __tag__(self):
        return "ETA"

    def __call__(self, sess, epoch, model, loss):
        self.iter += 1

        if self.reestimate and self.iter >= self.max_iters / self.max_epochs:
            self.max_iters *= self.max_epochs

        if self.reestimate and self.epoch != epoch:
            self.max_iters = self.iter * self.max_epochs
            self.reestimate = False

        if not self.iter == 0 and self.iter % self.iter_interval == 0:
            progress = float(self.iter) / self.max_iters
            current_time = time.time()
            elapsed = current_time - self.start
            eta = (1-progress) * elapsed
            eta_date = strftime("%y-%m-%d %H:%M:%S", localtime(current_time + eta))

            def format_eta(seconds):
                if seconds == float("inf"):
                    return "never"
                else:
                    seconds, _ = divmod(seconds, 1)
                    minutes, seconds = divmod(seconds, 60)
                    hours, minutes = divmod(minutes, 60)
                    seconds = str(int(seconds))
                    minutes = str(int(minutes))
                    hours = str(int(hours))

                    if len(hours) < 2:
                        hours = "0"+hours
                    if len(minutes) < 2:
                        minutes = "0"+minutes
                    if len(seconds) < 2:
                        seconds = "0"+seconds

                    return "%s:%s:%s" % (hours, minutes, seconds)

            print("Epoch %d\tIter %d\tETA in %s [%2.2f"
                  % (epoch, self.iter, format_eta(eta), progress * 100) + "%] " + eta_date)

            self.update_summary(sess, self.iter, self.__tag__(), float(eta))


class AccuracyHook(TraceHook):
    def __init__(self, batches, predict, target, at_every_epoch=1,
                 placeholders=None, summary_writer=None):
        super(AccuracyHook, self).__init__(summary_writer)
        self.batches = batches
        self.predict = predict
        self.target = target
        self.at_every_epoch = at_every_epoch
        self.placeholders = placeholders
        self.done_for_epoch = False
        self.iter = 0

    def __tag__(self):
        return "Acc"

    def __call__(self, sess, epoch, model, loss):
        self.iter += 1
        if epoch % self.at_every_epoch == 0:
            if not self.done_for_epoch:
                total = 0
                correct = 0

                for i, batch in enumerate(self.batches):
                    if self.placeholders is not None:
                        feed_dict = dict(zip(self.placeholders, batch))
                    else:
                        feed_dict = batch

                    predicted = sess.run(self.predict, feed_dict=feed_dict)

                    target = feed_dict[self.target]
                    gold = target if np.shape(target) == np.shape(predicted) else np.argmax(target)
                    overlap = gold == predicted
                    # todo: extend further, because does not cover all likely cases yet
                    #overlap = np.argmax(feed_dict[self.target]) == predicted
                    correct += np.sum(overlap)
                    total += predicted.size

                acc = float(correct) / total * 100

                self.update_summary(sess, self.iter, self.__tag__(), acc)
                print("Epoch " + str(epoch) +
                      "\tAcc %4.2f" % acc +
                      "%\tCorrect " + str(correct) + "\tTotal " + str(total))
                self.done_for_epoch = True
        else:
            self.done_for_epoch = False


class TestAllHook(TraceHook):
    """
    Hook which tests all kinds of metrics, intended for running on test data
    """
    def __init__(self, batches, logits, predict, target, at_epoch=1, placeholders=None,
                 metrics=['Acc'], summary_writer=None, print_details=False, print_to=""):
        super(TestAllHook, self).__init__(summary_writer)
        self.batches = batches
        self.logits = logits
        self.predict = predict
        self.target = target
        self.at_epoch = at_epoch
        self.placeholders = placeholders
        self.done_for_epoch = False
        self.iter = 0
        self.print_details = print_details
        self.print_to = print_to

        supported_metrics = ['Acc', 'MRR']
        metrics = [metrics] if isinstance(metrics, str) else metrics
        self.metrics = [m for m in metrics if m in supported_metrics]

    def _calc_mrr(self, scores, gold):
        mrrs = []
        for inst in scores:
            order = inst.argsort()
            ranks = order.argsort()
            rank = 0
            for i in range(len(inst)):
                if np.where(ranks == i)[0] == gold:
                    break
                rank += 1
            mrr = float(rank) / float(len(inst) - 1)
            # if self.print_details == True:
            #    print(gold, scores, mrr)
            mrrs.append(mrr)
        return mrrs


    def __call__(self, sess, epoch, model, loss):
        self.iter += 1
        if epoch != self.at_epoch:
            return

        if self.done_for_epoch == True:
            return

        print("\nApplying model to test data (metrics: %s):"%', '.join(self.metrics))
        predictions, corrects = [], []

        #initiate metrics
        rrs = []
        total = 0
        correct = 0

        for i, batch in enumerate(self.batches):
            if self.placeholders is not None:
                feed_dict = dict(zip(self.placeholders, batch))
            else:
                feed_dict = batch

            predicted = sess.run(self.predict, feed_dict=feed_dict)
            target = feed_dict[self.target]
            gold = target if np.shape(target) == np.shape(predicted) else np.argmax(target)
            #todo: extend further, because does not cover all likely cases yet
            corrects.append(gold)
            predictions.append(predictions)

            overlap = gold == predicted
            correct += np.sum(overlap)
            total += predicted.size

            if 'MRR' in self.metrics:
                rrs.extend(self._calc_mrr(sess.run(self.logits, feed_dict=feed_dict), gold))


            if self.print_details == True:
                #print(feed_dict[self.target], predicted)
                # alternative:
                # logits = sess.run(self.logits, feed_dict=feed_dict)
                # print(np.argmax(feed_dict[self.target]), np.argmax(logits))
                # alternative:
                probs = sess.run(tf.nn.softmax(self.logits), feed_dict=feed_dict)
                print('target/predicted: %s/%s, probs: %s'%(str(target),str(predicted),str(probs)))
                if self.print_to != "":
                    with open(self.print_to, "a") as myfile:
                        myfile.write('target/predicted: %s/%s, probs: %s\n'%(str(target),str(predicted),str(probs)))

        report = ""
        for metric in self.metrics:
            if metric == 'Acc':
                acc = float(correct) / total * 100
                self.update_summary(sess, self.iter, "TestAcc", acc)
                report += 'Acc %4.2f\t'%acc
            elif metric == 'MRR':
                mrr = sum(rrs) / float(total)
                self.update_summary(sess, self.iter, "TestMRR", mrr)
                report += 'MRR %4.2f\t' % mrr

        print("Epoch %d (correct %s/%s)\t%s"%(epoch, correct, total, report))
        #print("Epoch " + str(epoch) +
        #      "\tTestAcc %4.2f" % acc + "%" +
        #      "\tTestMRR %4.2f" % mrr +
        #      "\tCorrect " + str(correct) + "\tTotal " + str(total))

        self.done_for_epoch = True
        return corrects, predictions  # return those so more they can be printed to file, etc




class PRF1Hook(Hook):
    """
    Evaluate per-class and average precision, recall, F1
    """
    def __init__(self, batcher, placeholders, at_every_epoch):
        self.batcher = batcher
        self.placeholders = placeholders
        self.at_every_epoch = at_every_epoch

    def __call__(self, sess, epoch, iter, model, loss):
        if iter == 0 and epoch % self.at_every_epoch == 0:
            total = 0
            correct = 0
            truth_all = []
            pred_all = []
            for values in self.batcher:
                total += len(values[-1])
                feed_dict = {}
                for i in range(0, len(self.placeholders)):
                    feed_dict[self.placeholders[i]] = values[i]
                truth = np.argmax(values[-1], 1)
                predicted = sess.run(tf.arg_max(tf.nn.softmax(model), 1), feed_dict=feed_dict)
                correct += sum(truth == predicted)
                truth_all.extend(truth)
                pred_all.extend(predicted)
            print(classification_report(truth_all, pred_all, digits=4))


class SaveModelHook(Hook):
    def __init__(self, path, at_epoch, at_every_epoch=0):
        self.path = path + "/"
        self.at_epoch = at_epoch
        self.at_every_epoch = at_every_epoch

        # fixme: don't save optimizer parameters
        # self.saver = tf.train.Saver(tf.all_variables())
        self.saver = tf.train.Saver(tf.trainable_variables())

    def __call__(self, sess, epoch, iter, model, loss):
        if epoch == self.at_epoch:
            print("Saving model...")
            # todo
            pass
            #save_model(self.saver, sess, self.path, model, None)


class LoadModelHook(Hook):
    def __init__(self, path, at_epoch, at_every_epoch=0):
        self.path = path + "/"
        self.at_epoch = at_epoch
        self.at_every_epoch = at_every_epoch
        self.saver = tf.train.Saver(tf.all_variables())

    def __call__(self, sess, epoch, iter, model, loss):
        if epoch == self.at_epoch:
            print("Loading model...")
            # todo
            pass
            #model = load_model(sess, self.path + "latest/")