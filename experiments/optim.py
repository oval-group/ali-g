import torch.optim

from dfw import DFW
from dfw.baselines import BPGrad
from l4pytorch import L4Mom, L4Adam
from alig.th import AliG
from alig.th.projection import l2_projection


def get_optimizer(args, parameters):
    parameters = list(parameters)
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=args.eta, weight_decay=args.weight_decay,
                                    momentum=args.momentum, nesterov=bool(args.momentum))
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.eta, weight_decay=args.weight_decay)
    elif args.opt == "adagrad":
        optimizer = torch.optim.Adagrad(parameters, lr=args.eta, weight_decay=args.weight_decay)
    elif args.opt == "amsgrad":
        optimizer = torch.optim.Adam(parameters, lr=args.eta, weight_decay=args.weight_decay, amsgrad=True)
    elif args.opt == 'dfw':
        optimizer = DFW(parameters, eta=args.eta, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'bpgrad':
        optimizer = BPGrad(parameters, eta=args.eta, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'alig':
        optimizer = AliG(parameters, max_lr=args.eta, momentum=args.momentum,
                         projection_fn=lambda: l2_projection(parameters, args.max_norm))
    elif args.opt == 'bpgrad':
        optimizer = BPGrad(parameters, eta=args.eta, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'l4adam':
        optimizer = L4Adam(parameters, weight_decay=args.weight_decay)
    elif args.opt == 'l4mom':
        optimizer = L4Mom(parameters, weight_decay=args.weight_decay)
    else:
        raise ValueError(args.opt)

    print("Optimizer: \t {}".format(args.opt.upper()))

    optimizer.step_size = args.eta
    optimizer.step_size_unclipped = args.eta
    optimizer.momentum = args.momentum

    if args.load_opt:
        state = torch.load(args.load_opt)['optimizer']
        optimizer.load_state_dict(state)
        print('Loaded optimizer from {}'.format(args.load_opt))

    return optimizer


def decay_optimizer(optimizer, decay_factor=0.1):
    if isinstance(optimizer, torch.optim.SGD):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_factor

        optimizer.step_size = optimizer.param_groups[0]['lr']
        optimizer.step_size_unclipped = optimizer.param_groups[0]['lr']
    else:
        raise ValueError
