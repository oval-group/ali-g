import torch.nn as nn
from dfw.losses import MultiClassHingeLoss, set_smoothing_enabled


def get_loss(args):
    if args.opt == 'dfw':
        loss_fn = MultiClassHingeLoss()
        if 'cifar' in args.dataset:
            args.smooth_svm = True
    else:
        loss_fn = nn.CrossEntropyLoss()

    print('L2 regularization: \t {}'.format(args.weight_decay))
    print('\nLoss function:')
    print(loss_fn)

    if args.cuda:
        loss_fn = loss_fn.cuda()

    return loss_fn
