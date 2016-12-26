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

    def at_epoch_end(self, *args, **kwargs):
        self.__call__(*args, **kwargs)

    def at_iteration_end(self, *args, **kwargs):
        self.__call__(*args, **kwargs)

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
        self.reset = True

    def __tag__(self):
        return "Speed"

    def __call__(self, sess, epoch, model, loss):
        self.iter += 1
        if self.reset:
            self.t0 = time.time()
            self.reset = False
        elif self.iter % self.iter_interval == 0:
            diff = time.time() - self.t0
            speed = "%.2f" % (self.num_examples / diff)
            print("Epoch " + str(epoch) +
                  "\tIter " + str(self.iter) +
                  "\tExamples/s " + str(speed))
            self.update_summary(sess, self.iter, self.__tag__(), float(speed))
            self.t0 = time.time()

    def at_epoch_end(self, *args, **kwargs):
        #to eliminate drop in measured speed due to post-epoch hooks:
        #do not execute; reset for use during epochs only
        self.reset = True
        return

    def at_iteration_end(self, sess, epoch, model, loss):
        return self.__call__(sess, epoch, model, loss)



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
    #todo: will be deprecated; less general (e.g. for binary vectors in multi-label problems etc).
    #todo: accuracy already covered by EvalHook
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
        if epoch % self.at_every_epoch == 0 and loss==0:  #hacky: force to be post-epoch
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
                    correct += np.sum(overlap, axis=0)
                    total += predicted.size

                acc = float(correct) / total * 100

                self.update_summary(sess, self.iter, self.__tag__(), acc)
                print("Epoch " + str(epoch) +
                      "\tAcc %4.2f" % acc +
                      "%\tCorrect " + str(correct) + "\tTotal " + str(total))
                self.done_for_epoch = True
        else:
            self.done_for_epoch = False


class EvalHook(TraceHook):
    """
    Hook which applies various metrics.
    To be used during training on dev-data, and after training on test-data.
    """
    def __init__(self, batches, logits, predict, target, at_every_epoch=1, placeholders=None,
                 metrics=[], summary_writer=None, print_details=False, print_to="", info=""):
        """
        Initialize EvalHook object.
        Calling the hook prints calculated metrics to stdout, and returns targets, predictions, and a metrics dict.
        Meant as post-epoch hook; hence the argument `post_epoch=True` required when calling the hook.

        Args:
            batches:
                iterator / generator of batches; assumed each batch is a proper feed_dict in case placeholders=None
                otherwise paired with the placeholders to form feed_dicts.
            logits:
                tf op with logits
            predict:
                tf op that returns binary predictions, either for each instance as the index of the predicted answer (if unique)
                otherwise as a tensor of shape (batch_size, num_candidates), in which each prediction is a
                num_candidates-long binary vector (with 0 or more 1's and the rest 0).
                (num_candidates may vary over batches; in that case the metrics macro- microP/R/F1 are not available,
                but Acc and MRR are)
            target:
                binary targets; either for each instance as index of correct answer (if unique),
                or as binary vectors (with 0 or more 1's for correct answers), similar to predict.
            at_every_epoch:
                evaluated each at_every_epoch epochs. Note that calling EvalHook requires loss-argument to be 0,
                as automatically done post-epoch in quebap.sisyphos.train, hence always called after finishing the epoch.
            placeholders:
                placeholders, in case batches does not generate feed_dicts.
            metrics: list with metrics;
                default: [], which will return all available metrics (dep. on type of problem)
                Options are currently:
                'Acc', 'MRR', 'microP', 'microR', 'microF1', 'macroP', 'macroR', 'macroF1', 'precision', 'recall', 'F1'.
                Note: mostly not all the metrics are available, but deciding on sensible metrics is left to the user;
                    E.g.: if the number of candidates remains constant, micro- and macro-metrics will be returned,
                    even if the 'labels' for the actual candidates vary over instances.
            summary_writer:
                optional summary_writer; if set, each (epoch,metric) will be written by the summary_writer, with
                label `msg+"_"+metric` (e.g. "development_microF1")
            print_details:
                if True, prints for each instance:
                target, prediction, logits, and whether the prediction is entirely correct
            print_to:
                if a filename is specified, appends the details per instance (as for print_details) to that file.
            msg:
                print this message before the evaluation metrics; useful to distinguish between calls of en EvalHook on train/dev/test data.
        """
        super(EvalHook, self).__init__(summary_writer)
        self.batches = batches
        self.logits = logits
        self.predict = predict
        self.target = target
        self.at_every_epoch = at_every_epoch
        self.placeholders = placeholders
        #self.done_for_epoch = False
        self.iter = 0
        self.print_details = print_details
        self.print_to = print_to
        self.info = info
        self.metrics = metrics

    # def _calc_mrr(self, scores, target_index):
    #     if target_index:
    #
    #     mrrs = []
    #     for inst in scores:
    #         order = inst.argsort() #ranked list of answer ID's (increasing)
    #         ranks = order.argsort() #for each answer ID, rank in sorted scores
    #         rank = 0
    #         for i in range(len(inst)):
    #             if np.where(ranks == i)[0] == target_index:
    #                 break
    #             rank += 1
    #         mrr = float(rank) / float(len(inst) - 1)
    #         # if self.print_details == True:
    #         #    print(gold, scores, mrr)
    #         mrrs.append(mrr)
    #     return mrrs


    def _calc_mrr(self, scores, targets):
        #todo: verify!
        assert scores.shape == targets.shape #targets must be array with binary vector per instance (row)
        rrs = []
        for score, target in zip(scores, targets):
            order = score.argsort()[::-1] #sorted id's for highest to lost score
            #refIDs = np.where(target==1)
            refIDs = [i for i,t in enumerate(target) if t == 1]
            for i in range(len(order)):
                rank = i+1
                if order[i] in refIDs: #rank of first relevant answer
                    break
            rrs.append(1./rank)
        return rrs



    def _calc_micro_PRF(self, binary_predictions, binary_targets):
        #todo: verify!
        predict_complement = np.add(1.0, -binary_predictions)
        target_complement = np.add(1.0, -binary_targets)
        tp = np.sum(np.multiply(binary_predictions, binary_targets))  # p = 1, t = 1
        fp = np.sum(np.multiply(binary_predictions, target_complement))  # p = 1, t = 0
        fn = np.sum(np.multiply(predict_complement, binary_targets))  # p = 0, t = 1
        microp = tp / (tp + fp) if tp > 0 else 0.0
        micror = tp / (tp + fn) if tp > 0 else 0.0
        microf1 = 2. * microp * micror / (microp + micror) if (microp * micror > 0) else 0.0
        return microp, micror, microf1


    def _calc_macro_PRF(self, binary_predictions, binary_targets):
        """
        first dimension of input arguments: instances; 2nd dimension: different labels
        (returns None when only 1 label dimension is given)
        """
        #todo: verify!
        predict_complement = np.add(1.0, -binary_predictions)
        target_complement = np.add(1.0, -binary_targets)
        tp = np.sum(np.multiply(binary_predictions, binary_targets), axis=0)  # p = 1, t = 1
        fp = np.sum(np.multiply(binary_predictions, target_complement), axis=0)  # p = 1, t = 0
        fn = np.sum(np.multiply(predict_complement, binary_targets), axis=0)  # p = 0, t = 1
        inds_tp0 = np.where(tp!=0)
        r = np.zeros(np.shape(tp))
        r[inds_tp0] = np.divide(tp[inds_tp0], tp[inds_tp0] + fn[inds_tp0])
        p = np.zeros(np.shape(tp))
        p[inds_tp0] = np.divide(tp[inds_tp0], tp[inds_tp0] + fp[inds_tp0])
        inds_pr0 = np.where(np.multiply(p,r)!=0)
        f1 = np.zeros(np.shape(tp))
        f1[inds_pr0] = 2.*np.divide(np.multiply(p[inds_pr0], r[inds_pr0]), np.add(p[inds_pr0], r[inds_pr0]))
        macrop = np.mean(p)
        macror = np.mean(r)
        macrof1 = np.mean(f1)
        return macrop, macror, macrof1




    def __call__(self, sess, epoch, model, loss):
        """
        Call the EvalHook.

        Args:
            `sess`
            `epoch`
            `model`
            `loss`
            Additional inputs in **kwargs:
            `post_epoch` boolean;
                the hook is only executed when `post_epoch=True`
                (and provided epoch % self.at_every_epoch == 0).
        """
        self.iter += 1
        #post_epoch = False if not 'post_epoch' in kwargs else kwargs['post_epoch']
        #if epoch % self.at_every_epoch != 0 or not post_epoch:
        #    return

        #if self.done_for_epoch == True:
        #    return

        #print("Evaluation: ", self.info)
        predictions, targets = None, None
        predictions_bin, targets_bin = None, None

        convert2binary = lambda indices, n: np.asarray([[1 if j == i else 0 for j in range(n)] for i in indices])

        #initiate metrics
        rrs = []
        total = 0
        correct = 0
        n = 0

        consistent = True

        for i, batch in enumerate(self.batches):
            if self.placeholders is not None:
                feed_dict = dict(zip(self.placeholders, batch))
            else:
                feed_dict = batch

            prediction = sess.run(self.predict, feed_dict=feed_dict)
            target = feed_dict[self.target]
            logits = sess.run(self.logits, feed_dict=feed_dict)
            n = logits.shape[-1]  #may be variable per batch; may be 1 for binary problem

            #store evaluated forms (whether indices or binary vectors)
            predictions = prediction if predictions is None else np.concatenate([predictions, prediction], axis=0)
            targets = target if targets is None else np.concatenate([targets, target], axis=0)

            #create binary vectors
            target_bin = target if target.shape == logits.shape else convert2binary(target, n)
            prediction_bin = prediction if prediction.shape == logits.shape else convert2binary(prediction, n)

            overlap = [np.asarray([p==t]).all() for p,t in zip(prediction_bin, target_bin)]
            correct += np.sum(overlap)
            total += prediction_bin.shape[0]

            if consistent:
                try:
                    predictions_bin = prediction_bin if predictions_bin is None else np.concatenate([predictions_bin, prediction_bin], axis=0)
                    targets_bin = target_bin if targets_bin is None else np.concatenate([targets_bin, target_bin], axis=0)
                #in case not compatible over different batches:
                except:
                    consistent = False

            if n > 1:
                rrs.extend(self._calc_mrr(logits, target_bin))
            #else: makes no sense to calculate MRR


            report = ""
            for i in range(prediction_bin.shape[0]):
                is_correct = all(target_bin[i] == prediction_bin[i])
                report += 'target:%s\tprediction:%s\tlogits:%s\tcorrect:%s\n'\
                              %(str(target[i]), str(prediction[i]), str(logits[i]), is_correct)
                # alternative:
                # probs = sess.run(tf.nn.softmax(self.logits), feed_dict=feed_dict)
            if self.print_details:
                print(report)
            if self.print_to != "":
                with open(self.print_to, "a") as myfile:
                    myfile.write(report)

        #calculate metrics:
        metrics = {}
        metrics['Acc'] = correct/float(total) if correct > 0 else 0.0
        if len(rrs) > 0:
            metrics['MRR'] = np.mean(rrs)
        if consistent:
            mp, mr, mf = self._calc_micro_PRF(predictions_bin, targets_bin)
            Mp, Mr, Mf = self._calc_macro_PRF(predictions_bin, targets_bin)
            if n > 1:
                metrics['microP'] = mp
                metrics['microR'] = mr
                metrics['microF1'] = mf
                metrics['macroP'] = Mp
                metrics['macroR'] = Mr
                metrics['macroF1'] = Mf
            elif n == 1:
                metrics['Prec'] = mp
                metrics['Rec'] = mr
                metrics['F1'] = mf

        #output
        if len(self.metrics) == 0:
            printmetrics = sorted(metrics.keys())
        else:
            printmetrics = [m for m in self.metrics if m in metrics.keys()]
        res = "Epoch %d\tcorrect: %d/%d"%(epoch, correct, total)
        for m in printmetrics:
            #if len(printmetrics) > 2:
            #    res += '\n'
            res += '\t%s: %.3f'%(m, metrics[m])
            self.update_summary(sess, epoch, self.info+'_'+m, metrics[m])
        res += '\t(%s)'%self.info
        print(res)



        #self.done_for_epoch = True
        return targets, predictions, metrics  # return those so more they can be printed to file, etc


    def at_epoch_end(self, sess, epoch, model, loss):
        if epoch % self.at_every_epoch == 0:
            self.__call__(sess, epoch, model, loss)
        else:
            return

    def at_iteration_end(self, *args, **kwargs):
        return




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