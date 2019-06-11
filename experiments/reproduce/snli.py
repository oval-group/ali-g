import os
import yaml

from scheduling import launch


def create_jobs():
    template = "python train_nli.py --no_tqdm --no_visdom "
    with open("reproduce/hparams/snli.yaml", "r") as f:
        hparams = yaml.safe_load(f)

    jobs = []
    for hparam in hparams:
        command = template + " ".join("--{} {}".format(key, value) for key, value in hparam.items())
        jobs.append(command)

    return jobs


if __name__ == "__main__":
    jobs = create_jobs()

    # change current directory to InferSent
    os.chdir('./InferSent/')
    launch(jobs)
    # change current directory back to original
    os.chdir('..')
