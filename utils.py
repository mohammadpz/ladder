import os
import logging
import numpy as np
import theano
from pandas import DataFrame, read_hdf
from blocks.extensions import Printing, SimpleExtension
from blocks.main_loop import MainLoop
from blocks.roles import add_role, AuxiliaryRole
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from fuel.datasets import MNIST, CIFAR10
from fuel.transformers import Transformer
from picklable_itertools import cycle, imap
from nn import ZCA, ContrastNorm

logger = logging.getLogger('main.utils')


class BnParamRole(AuxiliaryRole):
    pass
BNPARAM = BnParamRole()


def shared_param(init, name, cast_float32, role, **kwargs):
    v = np.float32(init)
    p = theano.shared(v, name=name, **kwargs)
    add_role(p, role)
    return p


class AttributeDict(dict):
    __getattr__ = dict.__getitem__

    def __setattr__(self, a, b):
        self.__setitem__(a, b)


class DummyLoop(MainLoop):
    def __init__(self, extensions):
        return super(DummyLoop, self).__init__(algorithm=None,
                                               data_stream=None,
                                               extensions=extensions)

    def run(self):
        for extension in self.extensions:
            extension.main_loop = self
        self._run_extensions('before_training')
        self._run_extensions('after_training')


class SaveParams(SimpleExtension):
    """Finishes the training process when triggered."""
    def __init__(self, early_stop_var, model, save_path, **kwargs):
        super(SaveParams, self).__init__(**kwargs)
        self.early_stop_var = early_stop_var
        self.save_path = save_path
        params_dicts = model.params
        self.params_names = params_dicts.keys()
        self.params_values = params_dicts.values()
        self.to_save = {}
        self.best_value = None
        self.add_condition('after_training', self.save)
        self.add_condition('on_interrupt', self.save)
        # self.add_condition('after_epoch', self.do)

    def save(self, which_callback, *args):
        to_save = {}
        for p_name, p_value in zip(self.params_names, self.params_values):
            to_save[p_name] = p_value.get_value()
        path = self.save_path + '/trained_params'
        np.savez_compressed(path, **to_save)

    def do(self, which_callback, *args):
        val = self.main_loop.log.current_row[self.early_stop_var]
        if self.best_value is None or val < self.best_value:
            self.best_value = val
            to_save = {}
            for p_name, p_value in zip(self.params_names, self.params_values):
                to_save[p_name] = p_value.get_value()
            path = self.save_path + '/trained_params_best'
            np.savez_compressed(path, **to_save)


class SaveExpParams(SimpleExtension):
    def __init__(self, experiment_params, dir, **kwargs):
        super(SaveExpParams, self).__init__(**kwargs)
        self.dir = dir
        self.experiment_params = experiment_params

    def do(self, which_callback, *args):
        df = DataFrame.from_dict(self.experiment_params, orient='index')
        df.to_hdf(os.path.join(self.dir, 'params'), 'params', mode='w',
                  complevel=5, complib='blosc')


class SaveLog(SimpleExtension):
    def __init__(self, dir, show=None, **kwargs):
        super(SaveLog, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        epoch = self.main_loop.status['epochs_done']
        current_row = self.main_loop.log.current_row
        logger.info("\nIter:%d" % epoch)
        for element in current_row:
            logger.info(str(element) + ":%f" % current_row[element])


def prepare_dir(save_to, results_dir='results'):
    base = os.path.join(results_dir, save_to)
    i = 0

    while True:
        name = base + str(i)
        try:
            os.makedirs(name)
            break
        except:
            i += 1

    return name


def load_df(dirpath, filename, varname=None):
    varname = filename if varname is None else varname
    fn = os.path.join(dirpath, filename)
    return read_hdf(fn, varname)


def filter_funcs_prefix(d, pfx):
    pfx = 'cmd_'
    fp = lambda x: x.find(pfx)
    return {n[fp(n) + len(pfx):]: v for n, v in d.iteritems() if fp(n) >= 0}


class Whitening(Transformer):
    """ Makes a copy of the examples in the underlying dataset and whitens it
        if necessary.
    """
    def __init__(self, data_stream, iteration_scheme, whiten, cnorm=None,
                 **kwargs):
        super(Whitening, self).__init__(data_stream,
                                        iteration_scheme=iteration_scheme,
                                        **kwargs)
        data = data_stream.get_data(slice(data_stream.dataset.num_examples))
        self.data = []
        for s, d in zip(self.sources, data):
            if 'features' == s:
                # Fuel provides Cifar in uint8, convert to float32
                d = np.require(d, dtype=np.float32)
                if cnorm is not None:
                    d = cnorm.apply(d)
                if whiten is not None:
                    d = whiten.apply(d)
                self.data += [d]
            elif 'targets' == s:
                d = unify_labels(d)
                self.data += [d]
            else:
                raise Exception("Unsupported Fuel target: %s" % s)

    def get_data(self, request=None):
        return (s[request] for s in self.data)


class SemiDataStream(Transformer):
    """ Combines two datastreams into one such that 'target' source (labels)
        is used only from the first one. The second one is renamed
        to avoid collision. Upon iteration, the first one is repeated until
        the second one depletes.
        """
    def __init__(self, data_stream_labeled, data_stream_unlabeled, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.ds_labeled = data_stream_labeled
        self.ds_unlabeled = data_stream_unlabeled
        # Rename the sources for clarity
        self.ds_labeled.sources = ('features_labeled', 'targets_labeled')
        # Rename the source for input pixels and hide its labels!
        self.ds_unlabeled.sources = ('features_unlabeled',)

    @property
    def sources(self):
        if hasattr(self, '_sources'):
            return self._sources
        return self.ds_labeled.sources + self.ds_unlabeled.sources

    @sources.setter
    def sources(self, value):
        self._sources = value

    def close(self):
        self.ds_labeled.close()
        self.ds_unlabeled.close()

    def reset(self):
        self.ds_labeled.reset()
        self.ds_unlabeled.reset()

    def next_epoch(self):
        self.ds_labeled.next_epoch()
        self.ds_unlabeled.next_epoch()

    def get_epoch_iterator(self, **kwargs):
        unlabeled = self.ds_unlabeled.get_epoch_iterator(**kwargs)
        labeled = self.ds_labeled.get_epoch_iterator(**kwargs)
        assert type(labeled) == type(unlabeled)

        return imap(self.mergedicts, cycle(labeled), unlabeled)

    def mergedicts(self, x, y):
        return dict(list(x.items()) + list(y.items()))


def unify_labels(y):
    """ Work-around for Fuel bug where MNIST and Cifar-10
    datasets have different dimensionalities for the targets:
    e.g. (50000, 1) vs (60000,) """
    yshape = y.shape
    y = y.flatten()
    assert y.shape[0] == yshape[0]
    return y


def make_datastream(dataset, indices, batch_size,
                    n_labeled=None, n_unlabeled=None,
                    balanced_classes=True, whiten=None, cnorm=None,
                    scheme=ShuffledScheme):
    if n_labeled is None or n_labeled == 0:
        n_labeled = len(indices)
    if batch_size is None:
        batch_size = len(indices)
    if n_unlabeled is None:
        n_unlabeled = len(indices)
    assert n_labeled <= n_unlabeled, 'need less labeled than unlabeled'

    if balanced_classes and n_labeled < n_unlabeled:
        # Ensure each label is equally represented
        logger.info('Balancing %d labels...' % n_labeled)
        all_data = dataset.data_sources[dataset.sources.index('targets')]
        y = unify_labels(all_data)[indices]
        n_classes = y.max() + 1
        assert n_labeled % n_classes == 0
        n_from_each_class = n_labeled / n_classes

        i_labeled = []
        for c in range(n_classes):
            i = (indices[y == c])[:n_from_each_class]
            i_labeled += list(i)
    else:
        i_labeled = indices[:n_labeled]

    # Get unlabeled indices
    i_unlabeled = indices[:n_unlabeled]

    ds = SemiDataStream(
        data_stream_labeled=Whitening(
            DataStream(dataset),
            iteration_scheme=scheme(i_labeled, batch_size),
            whiten=whiten, cnorm=cnorm),
        data_stream_unlabeled=Whitening(
            DataStream(dataset),
            iteration_scheme=scheme(i_unlabeled, batch_size),
            whiten=whiten, cnorm=cnorm)
    )
    return ds


def setup_data(p, test_set=False):
    dataset_class, training_set_size = {
        'cifar10': (CIFAR10, 40000),
        'mnist': (MNIST, 50000),
    }[p.dataset]

    # Allow overriding the default from command line
    if p.get('unlabeled_samples') is not None:
        training_set_size = p.unlabeled_samples

    train_set = dataset_class(("train",))

    # Make sure the MNIST data is in right format
    if p.dataset == 'mnist':
        train_set.data_sources = (
            (train_set.data_sources[0] / 255.).astype(np.float32),
            train_set.data_sources[1])
        d = train_set.data_sources[train_set.sources.index('features')]
        assert np.all(d <= 1.0) and np.all(d >= 0.0), \
            'Make sure data is in float format and in range 0 to 1'

    # Take all indices and permutate them
    all_ind = np.arange(train_set.num_examples)
    if p.get('dseed'):
        rng = np.random.RandomState(seed=p.dseed)
        rng.shuffle(all_ind)

    d = AttributeDict()

    # Choose the training set
    d.train = train_set
    d.train_ind = all_ind[:training_set_size]

    # Then choose validation set from the remaining indices
    d.valid = train_set
    d.valid_ind = np.setdiff1d(all_ind, d.train_ind)[:p.valid_set_size]
    # logger.info('Using %d examples for validation' % len(d.valid_ind))

    # Only touch test data if requested
    if test_set:
        d.test = dataset_class(("test",))
        d.test.data_sources = (
            (d.test.data_sources[0] / 255.).astype(np.float32),
            d.test.data_sources[1])
        d.test_ind = np.arange(d.test.num_examples)

    # Setup optional whitening, only used for Cifar-10
    in_dim = train_set.data_sources[
        train_set.sources.index('features')].shape[1:]
    if len(in_dim) > 1 and p.whiten_zca > 0:
        assert np.product(in_dim) == p.whiten_zca, \
            'Need %d whitening dimensions, not %d' % (np.product(in_dim),
                                                      p.whiten_zca)
    cnorm = ContrastNorm(p.contrast_norm) if p.contrast_norm != 0 else None

    def get_data(d, i):
        data = d.get_data(request=i)[d.sources.index('features')]
        # Fuel provides Cifar in uint8, convert to float32
        data = np.require(data, dtype=np.float32)
        return data if cnorm is None else cnorm.apply(data)

    if p.whiten_zca > 0:
        # logger.info('Whitening using %d ZCA components' % p.whiten_zca)
        whiten = ZCA()
        whiten.fit(p.whiten_zca, get_data(d.train, d.train_ind))
    else:
        whiten = None

    return in_dim, d, whiten, cnorm
