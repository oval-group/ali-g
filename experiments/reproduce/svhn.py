import yaml

from scheduling import launch


def create_jobs():
    template = """python main.py --dataset svhn-extra --model wrn --depth 16
        --width 4 --batch_size 128 --momentum 0 --epochs 160
        --dropout 0.4 --no_data_augmentation --no_visdom --no_tqdm """

    with open("reproduce/hparams/svhn.yaml", "r") as f:
        hparams = yaml.safe_load(f)

    jobs = []
    for hparam in hparams:
        jobs.append(template + " ".join("--{} {}".format(key, value) for key, value in hparam.items()))
    return jobs


if __name__ == "__main__":
    jobs = create_jobs()
    launch(jobs)
