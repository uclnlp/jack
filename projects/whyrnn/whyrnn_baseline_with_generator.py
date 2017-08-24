# general
import argparse
import sys
import os
import tensorflow as tf
import numpy as np
from time import time
import logging

# jack
import jtr.jack.readers as readers
from jtr.load.embeddings.embeddings import load_embeddings
from jtr.jack.data_structures import load_labelled_data
from jtr.preprocess.vocabulary import Vocab
from jtr.jack.tasks.mcqa.simple_mcqa import \
    SingleSupportFixedClassInputs, PairOfBiLSTMOverQuestionAndSupportModel
from jtr.jack.core import \
    SharedVocabAndConfig, JTReader, SharedResources, Sequence
from jtr.jack.tf_fun import rnn, simple


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(sys.argv[0]))

decoder_symbols = {
    'EOS': "<EOS>",
    'SOS': "<SOS>",
}


class PairOfBiLSTMOverSupportAndQuestionWithDecoderModel(
        PairOfBiLSTMOverQuestionAndSupportModel):

    def __init__(self, shared_resources, mcqa_input_module):
        super().__init__(shared_resources)
        # We store input module only to get to its .answer_vocab.id2sym.
        # We can only access that after input module has been set up from
        # data, so there is no way to directly get the id2sym table upon
        # creation of the current model.
        self.input_module = mcqa_input_module
        self.decoder_outputs_train = {}
        self.decoder_logits_train = {}
        self.decoder_outputs_infer = {}

    def forward_pass(self, shared_resources, embeddings,
                     Q, S, Q_lengths, S_lengths,
                     num_classes, keep_prob=1):
        # final states_fw_bw dimensions:
        # [[[batch, output dim], [batch, output_dim]]
        self.answer_id2sym = self.input_module.answer_vocab.id2sym

        Q_seq = tf.nn.embedding_lookup(embeddings, Q)
        S_seq = tf.nn.embedding_lookup(embeddings, S)

        # NOTE: this is *different* from the original
        # PairOfBiLSTMOverQuestionAndSupportModel, which had the
        # Question in the 1st LSTM, and the Support in the 2nd and
        # thus had Support conditional on Question, and used the
        # Support's LSTM final output as  state for predition
        all_states_fw_bw, final_states_fw_bw = rnn.pair_of_bidirectional_LSTMs(
            # OLD: # Q_seq, Q_lengths, S_seq, S_lengths,
            S_seq, S_lengths, Q_seq, Q_lengths,
            shared_resources.config['repr_dim'], drop_keep_prob=keep_prob,
            conditional_encoding=True)

        # ->  [batch, 2*output_dim]
        final_states = tf.concat([final_states_fw_bw[0][1],
                                 final_states_fw_bw[1][1]], axis=1)

        # [batch, 2*output_dim] -> [batch, num_classes]
        outputs = simple.fully_connected_projection(
            final_states, num_classes, name='output_projection')

        # Add decoder for each target class
        # 1. Get parameters
        self.decoder_outputs_train = {}
        self.decoder_logits_train = {}
        self.decoder_outputs_infer = {}
        self.decoder_targets = Q  # vocab indices, no embeddings (no Q_seq)!
        self.decoder_target_lengths = Q_lengths
        sos_id = shared_resources.vocab.sym2id[decoder_symbols['SOS']]
        eos_id = shared_resources.vocab.sym2id[decoder_symbols['EOS']]
        num_decoder_symbols = len(shared_resources.vocab)
        decoder_embeddings = embeddings  # DONE... decoder_vocab.get_embedding_matrix()
        max_inference_seq_len = tf.reduce_max(Q_lengths)
        # make input state as LSTMTuple
        state_fw, state_bw = final_states_fw_bw
        state_h = tf.concat(axis=1, values=[state_fw.h, state_bw.h])
        state_c = tf.concat(axis=1, values=[state_fw.c, state_bw.c])
        decoder_input_state = tf.contrib.rnn.LSTMStateTuple(
            state_c, state_h)

        # 2. Construct the decoders
        logger.debug("Constructing decoders")
        for i in range(0, num_classes):
            decoder_name = "decoder_class_{:d}_{}".format(
                i, self.answer_id2sym[i])
            # In decoder, the 'targets', which will serve as inputs
            # are the embeddings of the input sequence tokens
            outputs_train, logits_train, outputs_infer = \
                rnn.dynamic_lstm_decoder(
                    targets=Q_seq, target_lengths=Q_lengths,
                    output_size=shared_resources.config['repr_dim'],
                    input_state=decoder_input_state,
                    num_decoder_symbols=num_decoder_symbols,
                    decoder_embedding_matrix=decoder_embeddings,
                    max_decoder_inference_seq_len=max_inference_seq_len,
                    start_of_sequence_id=sos_id,
                    end_of_sequence_id=eos_id,
                    scope=decoder_name,
                    drop_keep_prob=keep_prob
                )
            self.decoder_outputs_train[i] = outputs_train
            self.decoder_logits_train[i] = logits_train
            self.decoder_outputs_infer[i] = outputs_infer

        return outputs

    def create_training_output(self, shared_resources: SharedResources,
                               logits: tf.Tensor,
                               labels: tf.Tensor) -> Sequence[tf.Tensor]:

        # PART 1. Label prediction loss
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels
            ), name='predictor_loss')

        # PART 2. Interpretation loss from decoder
        print("### Whoa! What's all this about? (Need to add loss...)")
        num_classes = len(self.decoder_outputs_train)
        # (a) convert labels to one hot vector
        labels_hot = tf.one_hot(
            indices=labels,
            depth=num_classes,
            on_value=1.0,
            off_value=0.0,
            axis=-1)
        # (c) get the different label decoder logits as single tensor,
        # taking the correct decoder output depending on the labels
        decoder_logits_stacked = tf.stack(
            [self.decoder_logits_train[i] for i in range(num_classes)],
            axis=-1)
        print("labels_hot.get_shape() = {}".format(labels_hot.get_shape()))
        print("decoder_logits_stacked.get_shape() = {}".format(decoder_logits_stacked.get_shape()))
        # b = batch, t = time (symbol), l = logit, j = the label index
        self.decoder_logits_merged = tf.einsum(
            'bj,btlj->btl', labels_hot, decoder_logits_stacked)
        print("decoder_logits_merged.get_shape() = {}".format(self.decoder_logits_merged.get_shape()))
        # (d) now calculate the interpretation loss
        num_decoder_symbols = len(shared_resources.vocab)
        interpretation_loss, _ = rnn.dynamic_lstm_decoder_loss(
            self.decoder_logits_merged,
            self.decoder_targets,
            self.decoder_target_lengths,
            num_decoder_symbols,
            scope='interpretation_loss'
        )
        return [loss + 0.1 * interpretation_loss]


# not anymore: @__mcqa_reader
def snli_reader_with_generator(vocab, config):
    """ Creates a SNLI reader instance (multiple choice qa model) with
    generator that aims to generate the entailed/contradictory sentence. """
    from jtr.jack.tasks.mcqa.simple_mcqa import \
        SingleSupportFixedClassInputs, EmptyOutputModule
    shared_resources = SharedVocabAndConfig(vocab, config)
    input_module = SingleSupportFixedClassInputs(shared_resources)
    return JTReader(
        shared_resources,
        input_module,
        PairOfBiLSTMOverSupportAndQuestionWithDecoderModel(shared_resources, input_module),
        EmptyOutputModule()
    )


def main():
    # input
    train_file = "data/SNLI/snli_1.0/snli_1.0_train_jtr_v1.json"
    dev_file = "data/SNLI/snli_1.0/snli_1.0_dev_jtr_v1.json"
    test_file = "data/SNLI/snli_1.0/snli_1.0_test_jtr_v1.json"

    parser = argparse.ArgumentParser(
        description='Baseline SNLI model experiments')

    # data files
    parser.add_argument('--jtr_path', default='.', help='path to jtr base')

    # debug mode
    parser.add_argument('--debug', action='store_true',
                        help="Run in debug mode")
    parser.add_argument('--debug_examples', default=2000, type=int,
                        help="If in debug mode, "
                        "how many examples should be used (default 2000)")

    # tensorboard path
    parser.add_argument('--tensorboard_path', default='./.tb/',
                        help='Folder for tensorboard logs')

    # config for preprocessing
    parser.add_argument('--lowercase', action='store_true',
                        help="Lowercase data")

    # config of Vocab
    parser.add_argument('--vocab_max_size', default=sys.maxsize, type=int)
    parser.add_argument('--vocab_min_freq', default=1, type=int)

    # config of embeddings
    parser.add_argument('--pretrain', action='store_true',
                        help="Use pretrained embeddings, "
                        "by default the initialisation is random")
    parser.add_argument('--normalize_embeddings', action='store_true',
                        help="Normalize (initial) embeddings")
    parser.add_argument('--init_embeddings', default='uniform',
                        choices=['uniform', 'normal'])

    # config of model architecture
    parser.add_argument('--hidden_dim', default=100, type=int,
                        help="Size of the hidden representations, default 100")

    # training
    parser.add_argument('--batch_size', default=256,
                        type=int, help="Batch size for training data, "
                        "default 256")
    parser.add_argument('--eval_batch_size', default=256,
                        type=int, help="Batch size when eval=True, "
                        "default 256")
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help="Learning rate, default 0.001")
    parser.add_argument('--l2', default=0.0, type=float,
                        help="L2 regularization weight, default 0.0")
    parser.add_argument('--clip_value', default=0, type=float,
                        help="Gradients clipped between "
                        "[-clip_value, clip_value] (default = 0, no clipping)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout probability "
                        "(default 0.0 for no dropout)")
    parser.add_argument('--epochs', default=30, type=int,
                        help="Number of train epochs, default 30")

    # misc
    parser.add_argument('--seed', default=1337, type=int, help='random seed')
    parser.add_argument('--write_metrics_to', default=None, type=str,
                        help='Filename to log the metrics of the EvalHooks')

    args = parser.parse_args()

    # read out explicitly
    jtr_path = args.jtr_path
    debug, debug_examples = args.debug, args.debug_examples
    tensorboard_path = args.tensorboard_path
    lowercase = args.lowercase
    vocab_max_size, vocab_min_freq = args.vocab_max_size, args.vocab_min_freq
    pretrain = args.pretrain
    init_embeddings = args.init_embeddings
    normalize_embeddings = args.normalize_embeddings
    repr_dim_input = 50 if debug else 300
    hidden_dim = args.hidden_dim
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    learning_rate = args.learning_rate
    dropout, l2, clip_value = args.dropout, args.l2, args.clip_value
    epochs = args.epochs
    write_metrics_to = args.write_metrics_to

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)

    # config params needed for JTReader
    config = {
        'batch_size': batch_size,
        'eval_batch_size': eval_batch_size,
        'learning_rate': learning_rate,
        'vocab_min_freq': vocab_min_freq,
        'vocab_max_size': vocab_max_size,
        'lowercase': lowercase,
        'repr_dim_input': repr_dim_input,
        'repr_dim': hidden_dim,
        'dropout': dropout,
        'init_embeddings': init_embeddings,
        'normalize_embeddings': normalize_embeddings,
    }

    # logging
    sw = tf.summary.FileWriter(tensorboard_path)

    # load SNLI data
    splits = [train_file, dev_file, test_file]
    max_count = debug_examples if debug else None
    train_set, dev_set, test_set = [
        load_labelled_data(os.path.join(args.jtr_path, f), max_count)
        for f in splits
    ]
    for s, l in zip([train_set, dev_set, test_set], ['train', 'dev', 'test']):
        logger.info('loaded {:d} {:s} instances'.format(len(s), l))

    # Note: questions and support already have start/end of sequence symbols
    # see simple_mcqa.py:preprocess(...)

    # load pre-trained embeddings
    embeddings = None
    if pretrain:
        if debug:
            emb_file = 'glove.6B.50d.txt'
            embeddings = load_embeddings(
                os.path.join(jtr_path, 'data', 'GloVe', emb_file),
                'glove')
        else:
            emb_file = 'GoogleNews-vectors-negative300.bin.gz'
            embeddings = load_embeddings(
                os.path.join(jtr_path, 'data', 'SG_GoogleNews', emb_file),
                'word2vec')
        logger.info('loaded pre-trained embeddings ({})'.format(emb_file))

    # create Vocab object
    vocab = Vocab(emb=embeddings)

    # filter dev and test tokens which have pre-trained embeddings
    # (to avoid having to load them all)
    if pretrain:
        dev_tmp = SingleSupportFixedClassInputs.preprocess(
            dev_set, lowercase=config['lowercase'],
            test_time=False, add_lengths=False)
        test_tmp = SingleSupportFixedClassInputs.preprocess(
            test_set, lowercase=config['lowercase'],
            test_time=False, add_lengths=False)
        vocab.add_pretrained_for_testing(
            dev_tmp['question'], dev_tmp['support'])
        vocab.add_pretrained_for_testing(
            test_tmp['question'], test_tmp['support'])
        logger.debug(
            'loaded {:d} filtered pretrained symbols into '
            'vocab for dev and test data'.format(len(vocab.symset_pt)))

    # create reader
    # reader = readers.readers['snli_reader_with_generator'](vocab, config)
    reader = snli_reader_with_generator(vocab, config)

    # add hooks
    from jtr.jack.train.hooks import LossHook
    hooks = [
        LossHook(reader, iter_interval=50, summary_writer=sw),
        readers.eval_hooks['snli_reader'](
            reader, dev_set, iter_interval=100, info='dev',
            summary_writer=sw, write_metrics_to=write_metrics_to),
        readers.eval_hooks['snli_reader'](
            reader, test_set, epoch_interval=args.epochs,
            info='test', write_metrics_to=write_metrics_to)
    ]
    if args.debug:
        hooks.append(readers.eval_hooks['snli_reader'](
            reader, train_set, iter_interval=100, info='train',
            summary_writer=sw, write_metrics_to=write_metrics_to))

    # Here we initialize our optimizer
    # we choose Adam with standard momentum values
    optim = tf.train.AdamOptimizer(config['learning_rate'])

    # TODO loss!!!!
    t0 = time()
    reader.train(
        optim, train_set,
        hooks=hooks,
        max_epochs=epochs,
        l2=l2,
        clip=None if abs(clip_value) < 1.e-12 else [-clip_value, clip_value]
    )
    # TODO: check device setup in JTReader.train
    print('training took {:.3f} hours'.format((time() - t0) / 3600.))


if __name__ == "__main__":
    main()
