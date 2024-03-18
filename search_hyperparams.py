# This file contain hyper-parameter searching
import argparse
import os
from subprocess import check_call
import sys

from utils.misc import Params

PYTHON = sys.executable


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent_dir",
        default="experiments/some_dir",
        help="Directory containing params.json",
    )
    parser.add_argument(
        "--data_dir", default="data/", help="Directory containing the dataset"
    )

    return parser


def launch_training_job(parent_dir, data_dir, job_name, params):
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, "params.json")
    params.save(json_path)

    # launching train with this config
    cmd = "{python} train.py --model_dir={model_dir} --data_dir {data_dir}".format(
        python=PYTHON, model_dir=model_dir, data_dir=data_dir
    )
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    args = get_parser().parse_args()
    json_path = os.path.join(args.parent_dir, "params.json")
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(
        json_path
    )
    params = Params(json_path)

    # Learning rate search
    learning_rates = [1e-4, 1e-3, 1e-2]
    for learning_rate in learning_rates:
        params.learning_rate = learning_rate
        job_name = "learning_rate_{}".format(learning_rate)
        launch_training_job(args.parent_dir, args.data_dir, job_name, params)
