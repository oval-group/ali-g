import torch
import torch.nn as nn
from dfw.losses import MultiClassHingeLoss, set_smoothing_enabled


def get_loss(args):
    if args.opt == 'dfw':
        loss_fn = MultiClassHingeLoss()
        if 'cifar' in args.dataset:
            args.smooth_svm = True
    elif args.dataset == 'imagenet':
        return EntrLoss(n_classes=args.n_classes)
    else:
        loss_fn = nn.CrossEntropyLoss()

    print('L2 regularization: \t {}'.format(args.weight_decay))
    print('\nLoss function:')
    print(loss_fn)

    if args.cuda:
        loss_fn = loss_fn.cuda()

    return loss_fn

class EntrLoss(nn.Module):
    """Implementation from https://github.com/locuslab/lml/blob/master/smooth-topk/src/losses/entr.py.

    The MIT License

    Copyright 2019 Intel AI, CMU, Bosch AI

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
    """
    def __init__(self, n_classes, k=5, tau=1.0):
        super(EntrLoss, self).__init__()
        self.n_classes = n_classes
        self.k = k
        self.tau = tau

    def forward(self, x, y):
        n_batch = x.shape[0]

        x = x/self.tau
        x_sorted, I = x.sort(dim=1, descending=True)
        x_sorted_last = x_sorted[:,self.k:]
        I_last = I[:,self.k:]

        fy = x.gather(1, y.unsqueeze(1))
        J = (I_last != y.unsqueeze(1)).type_as(x)

        # Could potentially be improved numerically by using
        # \log\sum\exp{x_} = c + \log\sum\exp{x_-c}
        safe_z = torch.clamp(x_sorted_last-fy, max=80)
        losses = torch.log(1.+torch.sum(safe_z.exp()*J, dim=1))

        return losses.mean()
