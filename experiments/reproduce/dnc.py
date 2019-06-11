import os
import numpy as np
import yaml

from scheduling import launch


def add_gradient_jobs(jobs):
    optimizers = ('adam', 'adagrad', 'nag', 'sgd', 'alig', 'rmsprop')
    lr_list = (1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)

    for optimizer in optimizers:
        for lr in lr_list:
            jobs.append('python train.py --optimizer {optimizer} --learning_rate {lr}'
                        .format(optimizer=optimizer, lr=lr))
    jobs.append("python train.py --optimizer alig")


def add_l4_jobs(jobs):
    optimizers = ('l4adam', 'l4mom')
    fraction_list = list(np.round(np.arange(0.05, 1, 0.05), 2))
    for optimizer in optimizers:
        for fraction in fraction_list:
            jobs.append('python train.py --optimizer {optimizer} --fraction {fraction}'
                        .format(optimizer=optimizer, fraction=fraction))


def create_jobs():
    jobs = []
    add_gradient_jobs(jobs)
    add_l4_jobs(jobs)
    return jobs


if __name__ == "__main__":
    jobs = create_jobs()

    # change current directory to InferSent
    os.chdir('./dnc/')
    launch(jobs, on_gpu=False)
    # change current directory back to original
    os.chdir('..')
