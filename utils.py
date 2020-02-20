import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.optim as optim
import os
import logging
import yaml
import shutil


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def parse_args_and_config(args, config_dir="../configs", logdir="../logs", resume=False):
    """
    :return args, config: namespace objects that stores information in args and config files.
    """
    args.log = os.path.join(logdir, args.doc)
    # parse config file
    if not resume:
        with open(os.path.join(config_dir, args.config), 'r') as f:
            config = yaml.load(f)
        new_config = dict2namespace({**config, **vars(args)})
    else:
        with open(os.path.join(args.log, 'config.yml'), 'r') as f:
            config = yaml.load(f)
        new_config = dict2namespace({**vars(config), **vars(args)})

    # add device information to config
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    new_config.device = device

    if not resume:
        if os.path.exists(args.log):
            shutil.rmtree(args.log)
        os.makedirs(args.log)
        with open(os.path.join(args.log, 'config.yml'), 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)

    # setup logger
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)
    logging.info("Using device: {}".format(device))

    # set random seed
    torch.manual_seed(new_config.seed)
    torch.cuda.manual_seed_all(new_config.seed)
    np.random.seed(new_config.seed)
    logging.info("Run name: {}".format(args.doc))

    return args, new_config

def get_optimizer(parameters, config):
    if config.optim.weight_decay > 0:
        logging.info("Using weight decay" + "!" * 80)

    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(0.9, 0.999))
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))