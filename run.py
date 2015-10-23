import logging
import os
import sys
import numpy
import time
import theano
from theano.tensor.type import TensorType
from blocks.algorithms import GradientDescent, Adam
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from fuel.schemes import ShuffledScheme, SequentialScheme
from utils import prepare_dir, load_df, DummyLoop
from utils import SaveExpParams, SaveLog, SaveParams, AttributeDict
from nn import ApproxTestMonitoring, FinalTestMonitoring, TestMonitoring
from nn import LRDecay
from ladder import LadderAE
from utils import make_datastream, setup_data
from random import randint
logger = logging.getLogger('main')


def setup_model(p):
    ladder = LadderAE(p)
    # Setup inputs
    input_type = TensorType(
        'float32', [False] * (len(p.encoder_layers[0]) + 1))
    x_only = input_type('features_unlabeled')
    x = input_type('features_labeled')
    y = theano.tensor.lvector('targets_labeled')
    ladder.apply(x, y, x_only)

    # Load parameters if requested
    if p.get('load_from'):
        with open(p.load_from + '/trained_params.npz') as f:
            loaded = numpy.load(f)
            model = Model(ladder.costs.total)
            params_dicts = model.params
            params_names = params_dicts.keys()
            for param_name in params_names:
                param = params_dicts[param_name]
                # '/f_6_.W' --> 'f_6_.W'
                slash_index = param_name.find('/')
                param_name = param_name[slash_index + 1:]
                assert param.get_value().shape == loaded[param_name].shape
                param.set_value(loaded[param_name])

    return ladder


def load_and_log_params(cli_params):
    cli_params = AttributeDict(cli_params)
    if cli_params.get('load_from'):
        p = load_df(cli_params.load_from, 'params').to_dict()[0]
        p = AttributeDict(p)
        for key in cli_params.iterkeys():
            if key not in p:
                p[key] = None
        new_params = cli_params
        loaded = True
    else:
        p = cli_params
        new_params = {}
        loaded = False

        # Make dseed seed unless specified explicitly
        if p.get('dseed') is None and p.get('seed') is not None:
            p['dseed'] = p['seed']

    logger.info('== COMMAND LINE ==')
    logger.info(' '.join(sys.argv))

    logger.info('== PARAMETERS ==')
    for k, v in p.iteritems():
        if new_params.get(k) is not None:
            p[k] = new_params[k]
            replace_str = "<- " + str(new_params.get(k))
        else:
            replace_str = ""
        logger.info(" {:20}: {:<20} {}".format(k, v, replace_str))
    return p, loaded


def get_error(args):
    """ Calculate the classification error """
    args['data_type'] = args.get('data_type', 'test')
    args['no_load'] = 'g_'

    targets, acts = analyze(args)
    guess = numpy.argmax(acts, axis=1)
    correct = numpy.sum(numpy.equal(guess, targets.flatten()))

    return (1. - correct / float(len(guess))) * 100.


def analyze(cli_params):
    p, _ = load_and_log_params(cli_params)
    _, data, whiten, cnorm = setup_data(p, test_set=True)
    ladder = setup_model(p)

    # Analyze activations
    dset, indices, calc_batchnorm = {
        'train': (data.train, data.train_ind, False),
        'valid': (data.valid, data.valid_ind, True),
        'test': (data.test, data.test_ind, True),
    }[p.data_type]

    if calc_batchnorm:
        monitored_variables = [
            ladder.costs.CE_clean,
            ladder.costs.CE_corr,
            ladder.error.clean,
            ladder.costs.total] + ladder.costs.denois.values()
        logger.info('Calculating batch normalization for clean.labeled path')
        main_loop = DummyLoop(
            extensions=[
                FinalTestMonitoring(
                    monitored_variables,
                    make_datastream(data.train, data.train_ind,
                                    # These need to match with the training
                                    p.batch_size,
                                    n_labeled=p.labeled_samples,
                                    n_unlabeled=len(data.train_ind),
                                    cnorm=cnorm,
                                    whiten=whiten, scheme=ShuffledScheme),
                    make_datastream(data.valid, data.valid_ind,
                                    p.valid_batch_size,
                                    n_labeled=len(data.valid_ind),
                                    n_unlabeled=len(data.valid_ind),
                                    cnorm=cnorm,
                                    whiten=whiten, scheme=ShuffledScheme),
                    prefix="valid_final", before_training=True),
                Printing(),
            ])
        main_loop.run()

    # Make a datastream that has all the indices in the labeled pathway
    ds = make_datastream(dset, indices,
                         batch_size=p.get('batch_size'),
                         n_labeled=len(indices),
                         n_unlabeled=len(indices),
                         balanced_classes=False,
                         whiten=whiten,
                         cnorm=cnorm,
                         scheme=SequentialScheme)

    # We want out the values after softmax
    outputs = ladder.act.clean.labeled.h[len(ladder.layers) - 1]

    # Replace the batch normalization paramameters with the shared variables
    if calc_batchnorm:
        outputreplacer = TestMonitoring()
        _, _, outputs = outputreplacer._get_bn_params(outputs)

    cg = ComputationGraph(outputs)
    f = cg.get_theano_function()

    it = ds.get_epoch_iterator(as_dict=True)
    res = []
    inputs = {'features_labeled': [],
              'targets_labeled': [],
              'features_unlabeled': []}
    # Loop over one epoch
    for d in it:
        # Store all inputs
        for k, v in d.iteritems():
            inputs[k] += [v]
        # Store outputs
        res += [f(*[d[str(inp)] for inp in cg.inputs])]

    # Concatenate all minibatches
    res = [numpy.vstack(minibatches) for minibatches in zip(*res)]
    inputs = {k: numpy.vstack(v) for k, v in inputs.iteritems()}

    return inputs['targets_labeled'], res[0]


def train(cli_params):
    cli_params['save_dir'] = prepare_dir(cli_params['save_to'])
    logfile = os.path.join(cli_params['save_dir'], 'log.txt')
    fh = logging.FileHandler(filename=logfile)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    p, loaded = load_and_log_params(cli_params)
    in_dim, data, whiten, cnorm = setup_data(p, test_set=True)
    if not loaded:
        p.encoder_layers = (in_dim,) + p.encoder_layers

    ladder = setup_model(p)

    # Training
    all_params = ComputationGraph([ladder.costs.total]).parameters
    logger.info('Number of found parameters: %s' % str(len(all_params)))
    logger.info('Found parameters:\n %s' % str(all_params))

    # Fetch all batch normalization updates. They are in the clean path.
    bn_updates = ComputationGraph([ladder.costs.CE_clean]).updates
    assert 'counter' in [u.name for u in bn_updates.keys()], \
        'No batch norm params in graph - the graph has been cut?'

    training_algorithm = GradientDescent(
        cost=ladder.costs.total, params=all_params,
        step_rule=Adam(learning_rate=ladder.lr))
    # In addition to actual training, also do BN variable approximations
    training_algorithm.add_updates(bn_updates)

    model = Model(ladder.costs.total)

    monitored_variables = [
        ladder.costs.CE_clean,
        ladder.costs.CE_corr,
        ladder.error.clean,
        training_algorithm.total_gradient_norm,
        ladder.costs.total] + ladder.costs.denois.values()

    main_loop = MainLoop(
        training_algorithm,
        make_datastream(data.train, data.train_ind,
                        p.batch_size,
                        n_labeled=p.labeled_samples,
                        n_unlabeled=p.unlabeled_samples,
                        whiten=whiten,
                        cnorm=cnorm),
        model=model,
        extensions=[
            FinishAfter(after_n_epochs=p.num_epochs),
            ApproxTestMonitoring(
                monitored_variables,
                make_datastream(data.valid, data.valid_ind,
                                p.valid_batch_size, whiten=whiten, cnorm=cnorm,
                                scheme=ShuffledScheme),
                prefix="valid_approx"),
            FinalTestMonitoring(
                monitored_variables,
                make_datastream(data.train, data.train_ind,
                                p.batch_size,
                                n_labeled=p.labeled_samples,
                                whiten=whiten, cnorm=cnorm,
                                scheme=ShuffledScheme),
                make_datastream(data.valid, data.valid_ind,
                                p.valid_batch_size,
                                n_labeled=len(data.valid_ind),
                                whiten=whiten, cnorm=cnorm,
                                scheme=ShuffledScheme),
                prefix="valid_final",
                after_n_epochs=p.num_epochs),
            FinalTestMonitoring(
                monitored_variables,
                make_datastream(data.train, data.train_ind,
                                p.batch_size,
                                n_labeled=p.labeled_samples,
                                whiten=whiten, cnorm=cnorm,
                                scheme=ShuffledScheme),
                make_datastream(data.test, data.test_ind,
                                p.valid_batch_size,
                                n_labeled=len(data.test_ind),
                                whiten=whiten, cnorm=cnorm,
                                scheme=ShuffledScheme),
                prefix="test_final",
                after_n_epochs=p.num_epochs),

            TrainingDataMonitoring(
                monitored_variables,
                prefix="train", after_epoch=True),

            SaveParams('valid_approx_CE_clean', model, p.save_dir,
                       after_epoch=False),
            SaveExpParams(p, p.save_dir, before_training=True),
            SaveLog(p.save_dir, after_epoch=True),
            LRDecay(ladder.lr, p.num_epochs * p.lrate_decay, p.num_epochs,
                    after_epoch=True),
            Printing()])
    main_loop.run()


if __name__ == "__main__":
    index = int(sys.argv[1])

    dseeds = [1, 777, 405, 186, 620, 209, 172, 734, 154, 996]
    lrs = [0.001, 0.002, 0.0001, 0.0002]
    labeled_sampless = [100, 60000]
    denoising_cost_xs = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                         (100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                         (500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                         (1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                         (4000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                         (1000, 10, 0.1, 0.1, 0.1, 0.1, 0.1),
                         (6000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                         (500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                         (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                         (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0)]
    f_local_noise_std = [(0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3),
                         (0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                         (0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                         (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                         (0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                         (0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                         (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                         (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3),
                         (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
                         (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7)]
    decoder_specs = [('sig', 'sig', 'sig', 'sig', 'sig', 'sig', 'sig'),
                     ('sig', 'vert', 'vert', 'vert', 'vert', 'vert', 'vert')]
    seed = int(time.time())

    def pick_random(options):
        random_index = randint(0, len(options) - 1)
        return options[random_index]

    logging.basicConfig(level=logging.INFO)
    evaluate = False
    t_start = time.time()
    if evaluate:
        d = {'load_from': 'results/mnist_all_bottom51',
             'cmd': 'evaluate',
             'data_type': 'test'}
        err = get_error(d)
        logger.info('Test error: %f' % err)
    else:
        d = {'dseed': pick_random(dseeds), 'seed': 1, 'num_epochs': 150,
             'lr': pick_random(lrs), 'lrate_decay': 0.67,
             'unlabeled_samples': 60000, 'labeled_samples': pick_random(labeled_sampless),
             'denoising_cost_x': pick_random(denoising_cost_xs),
             'f_local_noise_std': pick_random(f_local_noise_std),
             'decoder_spec': pick_random(decoder_specs),
             'save_to': 'mnist_all_bottom',
             'cmd': 'train',
             'top_c': True, 'batch_size': 100, 'dataset': 'mnist',
             'valid_set_size': 10000, 'whiten_zca': 0,
             'act': 'relu', 'valid_batch_size': 100, 'contrast_norm': 0,
             'encoder_layers': ('1000', '500', '250', '250', '250', '10'),
             }
        train(d)
    logger.info('Took %.1f minutes' % ((time.time() - t_start) / 60.))
