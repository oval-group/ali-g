# Adaptive Learning Rates for Interpolation with Gradients

This repository contains the implementation of the paper [Training Neural Networks for and by Interpolation](http://arxiv.org/abs/1906.05661) in PyTorch and Tensorflow. If you use this work for your research, please cite the paper:

```
@Article{berrada2019training,
  author       = {Berrada, Leonard and Zisserman, Andrew and Kumar, M Pawan},
  title        = {Training Neural Networks for and by Interpolation},
  journal      = {arxiv},
  year         = {2019},
}
```

The implementation of the optimization algorithm is self-contained as a python package.

The code to reproduce the experiments is provided in [experiments](experiments).

## Code Requirements

This code should work for PyTorch >= 1.0 in python3. A Tensorflow implementation is also provided for reference.

The package only requires PyTorch / Tensorflow to use the PyTorch / Tensorflow implementation. The test requires both PyTorch and Tensorflow, as well as numpy. The experiments requires additional packages, which are listed in the [requirements file](requirements.txt).

## Installation

* Clone this repository: `git clone --recursive https://github.com/oval-group/ali-g` (the `recursive` option is only needed to reproduce experiments using sub-modules).
* Go to directory and install the requirements: `cd ali-g && pip install -r requirements.txt`.
* Install the AliG package `python setup.py install`.

## Example

### Import:
```python
# tensorflow import
from alig.tf import AliG as AliG_tf

# pytorch import
from alig.th import AliG as AliG_th
```

### Usage without any regularization:
```python
import torch
from alig.th import AliG


# boilerplate code:
# `model` is a nn.Module
# `x` is an input sample, `y` is a label

# create AliG optimizer with maximal learning-rate of 0.1
optimizer = AliG(model.parameters(), max_lr=0.1)

# AliG can be used with standard pytorch syntax
optimizer.zero_grad()
loss = torch.nn.functional.cross_entropy(model(x), y)
loss.backward()
# NB: AliG needs to have access to the current loss value,
# (this syntax is compatible with standard pytorch optimizers too)
optimizer.step(lambda: float(loss))
```

### Usage with the regularization as a constraint:

In order to enforce constraints on the parameter space, you can provide a projection function to the optimizer (only implemented in PyTorch at the moment).
The implementation assumes that (i) the projection function requires no input argument, and (ii) it performs the projection in-place.

For example:
```python
import torch
from alig.th import AliG


# Implementation of a projection on a Euclidean ball
def l2_projection(parameters, max_norm):
    total_norm = torch.sqrt(sum(p.norm() ** 2 for p in parameters))
    if total_norm > max_norm:
        ratio = max_norm / total_norm
        for p in parameters:
            p *= ratio

# create an optimizer that will apply the projection at each update (every time `optimizer.step` is called)
params = list(model.parameters())
optimizer = AliG(params, max_lr=0.1,
                 projection_fn=lambda: l2_projection(parameters=params, max_norm=100))

# the rest of the code is identical
```

## Technical Requirements for Applicability

AliG exploits the interpolation property to compute a step-size. This induces the following two requirements for its applicability:
* The model should be able to achieve a loss of zero on all training samples (typically in the order of 0.01 or less).
* If there is regularization, it should be expressed as a constraint (see example above).


## Reproducing the Results

The following command lines assume that the current working directory is `experiments`.

* To reproduce the DNC experiments: run `python reproduce/dnc.py`
* To reproduce the SVHN experiments: run `python reproduce/svhn.py`
* To reproduce the SNLI experiments: follow the [preparation instructions](https://github.com/lberrada/InferSent/tree/ali-g#download-datasets) and run  `python reproduce/snli.py`
* To reproduce the CIFAR experiments: run `python reproduce/cifar.py`
* To reproduce the CIFAR experiments on training performance: run `python reproduce/cifar_train.py`
* To reproduce the ImageNet experiment: run `python reproduce/imagenet.py`


## Acknowledgments

We use the following third-party implementations:
* [DNC](https://github.com/deepmind/dnc).
* [InferSent](https://github.com/facebookresearch/InferSent).
* [DenseNets](https://github.com/andreasveit/densenet-pytorch).
* [Wide ResNets](https://github.com/xternalz/WideResNet-pytorch).
* [Top-k Truncated Cross-Entropy](https://github.com/locuslab/lml).