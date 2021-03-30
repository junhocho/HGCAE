from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
# import torch
from config import parser
from models.base_models import LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics, assign_gpus

from solver import Solver



def train(args):
    np.random.seed(args.seed)

    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            if args.node_cluster == 1:
                task = 'nc'
            else:
                task = 'lp'
            models_dir = os.path.join(os.environ['LOG_DIR'], task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])
        logging.info(f"Logging model in {save_dir}")
        args.save_dir = save_dir

    if args.node_cluster == 1:
        ### NOTE : node clustering use full edge
        args.val_prop = 0.0
        args.test_prop = 0.0

    import pprint
    args_info_pprint = pprint.pformat(vars(args))

    logging.info(args_info_pprint)

    # Load data
    logging.info("Loading Data : {}".format(args.dataset))
    t_load = time.time()
    data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))
    st0 = np.random.get_state()
    args.np_seed = st0

    t_load = time.time() - t_load
    logging.info(data['info'])
    logging.info('Loading data took time: {:.4f}s'.format(t_load))

    sol = Solver(args, data)
    sol.fit()
    sol.eval()

if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
