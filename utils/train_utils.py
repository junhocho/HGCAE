import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.modules.loss


def format_metrics(metrics, split):
    """Format metric in metric dict for logging."""
    return " ".join(
            ["{}_{}: {:.8f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])

def get_dir_name(models_dir):
    """Gets a directory to save the model.

    If the directory already exists, then append a new integer to the end of
    it. This method is useful so that we don't overwrite existing models
    when launching new jobs.

    Args:
        models_dir: The directory where all the models are.

    Returns:
        The name of a new directory to save the training logs and model weights.
    """
    if not os.path.exists(models_dir):
        save_dir = os.path.join(models_dir, '0')
        os.makedirs(save_dir)
    else:
        existing_dirs = np.array(
                [
                    d
                    for d in os.listdir(models_dir)
                    if os.path.isdir(os.path.join(models_dir, d))
                    ]
        ).astype(np.int)
        if len(existing_dirs) > 0:
            dir_id = str(existing_dirs.max() + 1)
        else:
            dir_id = "1"
        save_dir = os.path.join(models_dir, dir_id)
        os.makedirs(save_dir)
    return save_dir


def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                            f"--{param}",
                            action="append",
                            type=type(default[0]),
                            default=default,
                            help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser



import subprocess
def check_gpustats(columns=None):
    query = r'nvidia-smi --query-gpu=%s --format=csv,noheader' % ','.join(columns)
    smi_output = subprocess.check_output(query, shell=True).decode().strip()

    gpustats = []
    for line in smi_output.split('\n'):
        if not line:
            continue
        gpustat = line.split(',')
        gpustats.append({k: v.strip() for k, v in zip(columns, gpustat)})

    return gpustats


def assign_gpus(num_gpu, memory_threshold=1000):    # (MiB)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    columns = ['index', 'memory.used']
    gpustats = {i['index']: i['memory.used'] for i in check_gpustats(columns)}



    available_gpus = []
    for gpu in sorted(gpustats.keys()):
        if int(gpustats.get(gpu).split(' ')[0]) < memory_threshold:
            available_gpus.append(gpu)

    if len(available_gpus) < num_gpu:
        raise MemoryError('{} GPUs requested, but only {} available'.format(num_gpu, len(available_gpus)))

    gpus_to_assign = available_gpus[:num_gpu]
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpus_to_assign)
    return gpus_to_assign


