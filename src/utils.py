import yaml
import numpy as np
import random
from datetime import datetime

import torch as th

def set_seed(seed):
    """
    For seed to some modules.
    :param seed: int. The seed.
    :return:
    """
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True

def load_config(config_file=None):
    assert config_file is not None, "Error: config file not found."
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def parse_args(args_file="./data/configs/default.json"):
    config = load_config(config_file=args_file)
    return config


def now():
    return datetime.now()