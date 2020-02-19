import yaml
from scheduling import launch


def create_jobs():
    jobs = ["""python main.py --dataset imagenet --model resnet18 --opt alig
            --eta 10.0 --momentum 0.0 --batch_size 1024 --epochs 90
            --max_norm 400 --no_data_augmentation"""]
    return jobs


if __name__ == "__main__":
    jobs = create_jobs()
    launch(jobs)
