import os
import argparse
import warnings

from cuda import set_cuda


def parse_command():
    parser = argparse.ArgumentParser()

    _add_dataset_parser(parser)
    _add_model_parser(parser)
    _add_optimization_parser(parser)
    _add_loss_parser(parser)
    _add_misc_parser(parser)

    args = parser.parse_args()
    filter_args(args)

    return args


def _add_dataset_parser(parser):
    d_parser = parser.add_argument_group(title='Dataset parameters')
    d_parser.add_argument('--dataset', type=str, default=None,
                          help='dataset')
    d_parser.add_argument('--train_size', type=int, default=None,
                          help="training data size")
    d_parser.add_argument('--val_size', type=int, default=None,
                          help="val data size")
    d_parser.add_argument('--test_size', type=int, default=None,
                          help="test data size")
    d_parser.add_argument('--no_data_augmentation', dest='augment',
                          action='store_false', help='no data augmentation')
    d_parser.set_defaults(augment=True)


def _add_model_parser(parser):
    m_parser = parser.add_argument_group(title='Model parameters')
    m_parser.add_argument('--model', type=str,
                          help="model name")
    m_parser.add_argument('--depth', type=int, default=None,
                          help="depth of network on densenet / wide resnet")
    m_parser.add_argument('--width', type=int, default=None,
                          help="width of network on wide resnet")
    m_parser.add_argument('--growth', type=int, default=None,
                          help="growth rate of densenet")
    m_parser.add_argument('--no_bottleneck', dest="bottleneck", action="store_false",
                          help="bottleneck on densenet")
    m_parser.add_argument('--dropout', type=float, default=0,
                          help="dropout rate")
    m_parser.add_argument('--load_model', default=None,
                          help='data file with model')
    m_parser.set_defaults(pretrained=False, wrn=False, densenet=False, bottleneck=True)


def _add_optimization_parser(parser):
    o_parser = parser.add_argument_group(title='Training parameters')
    o_parser.add_argument('--epochs', type=int, default=None,
                          help="number of epochs")
    o_parser.add_argument('--batch_size', type=int, default=None,
                          help="batch size")
    o_parser.add_argument('--eta', type=float, default=0.1,
                          help="initial / maximal learning rate")
    o_parser.add_argument('--momentum', type=float, default=0.9,
                          help="momentum value for SGD")
    o_parser.add_argument('--opt', type=str, required=True,
                          help="optimizer to use")
    o_parser.add_argument('--T', type=int, default=[-1], nargs='+',
                          help="number of epochs between proximal updates / lr decay")
    o_parser.add_argument('--decay_factor', type=float, default=0.1,
                          help="decay factor for the learning rate / proximal term")
    o_parser.add_argument('--load_opt', default=None,
                          help='data file with opt')


def _add_loss_parser(parser):
    l_parser = parser.add_argument_group(title='Loss parameters')
    l_parser.add_argument('--weight_decay', type=float, default=0,
                          help="l2-regularization")
    l_parser.add_argument('--max_norm', type=float, default=None,
                          help="maximal l2-norm for constrained optimizers")
    l_parser.add_argument('--loss', type=str, default='ce', choices=("svm", "ce"),
                          help="loss function to use ('svm' or 'ce')")
    l_parser.add_argument('--smooth_svm', dest="smooth_svm", action="store_true",
                          help="smooth SVM")
    l_parser.set_defaults(smooth_svm=False)


def _add_misc_parser(parser):
    m_parser = parser.add_argument_group(title='Misc parameters')
    m_parser.add_argument('--seed', type=int, default=None,
                          help="seed for pseudo-randomness")
    m_parser.add_argument('--cuda', type=int, default=1,
                          help="use cuda")
    m_parser.add_argument('--no_visdom', dest='visdom', action='store_false',
                          help='do not use visdom')
    m_parser.add_argument('--server', type=str, default=None,
                          help="server for visdom")
    m_parser.add_argument('--port', type=int, default=None,
                          help="port for visdom")
    m_parser.add_argument('--xp_name', type=str, default=None,
                          help="name of experiment")
    m_parser.add_argument('--no_log', dest='log', action='store_false',
                          help='do not log results')
    m_parser.add_argument('--debug', dest='debug', action='store_true',
                          help='debug mode')
    m_parser.add_argument('--parallel_gpu', dest='parallel_gpu', action='store_true',
                          help="parallel gpu computation")
    m_parser.add_argument('--no_tqdm', dest='tqdm', action='store_false',
                          help="use of tqdm progress bars")
    m_parser.set_defaults(visdom=True, log=True, debug=False, parallel_gpu=False, tqdm=True)


def set_xp_name(args):
    if args.debug:
        args.log = args.visdom = False
        args.xp_name = '../debug'
        if not os.path.exists(args.xp_name):
            os.makedirs(args.xp_name)
    elif args.xp_name is None:
        xp_name = '../results/{data}/'.format(data=args.dataset)
        xp_name += "{model}{data}-{opt}--eta-{eta}--l2-{l2}--b-{b}"
        l2 = args.max_norm if args.opt == 'alig' else args.weight_decay
        data = args.dataset.replace("cifar", "")
        args.xp_name = xp_name.format(model=args.model, data=data, opt=args.opt, eta=args.eta, l2=l2, b=args.batch_size)

    if args.log:
        # generate automatic experiment name if not provided
        if os.path.exists(args.xp_name):
            warnings.warn('An experiment already exists at {}'
                          .format(os.path.abspath(args.xp_name)))
        else:
            os.makedirs(args.xp_name)


def set_num_classes(args):
    if args.dataset == 'cifar10':
        args.n_classes = 10
    elif args.dataset == 'cifar100':
        args.n_classes = 100
    elif args.dataset == 'snli':
        args.n_classes = 3
    elif 'svhn' in args.dataset:
        args.n_classes = 10
    else:
        raise ValueError


def set_visdom(args):
    if not args.visdom:
        return
    if args.server is None:
        if 'VISDOM_SERVER' in os.environ:
            args.server = os.environ['VISDOM_SERVER']
        else:
            args.visdom = False
            print("Could not find a valid visdom server, de-activating visdom...")


def filter_args(args):
    args.T = list(args.T)
    set_cuda(args)

    set_xp_name(args)
    set_visdom(args)
    set_num_classes(args)
