import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (0.01, 'learning rate'),
        'dropout': (0.0, 'dropout probability'),
        'cuda': (-1, 'which cuda device to use (-1 for cpu training, 99 for auto gpu assign)'),
        'epochs': (5000, 'maximum number of epochs to train for'),
        'weight-decay': (0., 'l2 regularization strength'),
        'optimizer': ('Adam', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'momentum': (0.999, 'momentum in optimizer'),
        'patience': (100, 'patience for early stopping'),
        'seed': (1234, 'seed for training'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'save': (0, '1 to save model and logs and 0 otherwise'),
        'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'sweep-c': (0, ''),
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        'print-epoch': (True, ''),
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min-epochs': (100, 'do not early stop before min-epochs'),
        'lambda-rec': (1.0, 'loss weight for reconstruction task.'),
        'lambda-lp': (1.0, 'lp loss weight. Used with lambda_lp=0 for HNN + rec decoder without lp loss')
    },
    'model_config': {
        'model': ('HGCAE', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT]'),
        'hidden-dim': ('16', 'hidden layer feature dimension. , comma seprated number'),
        'dim': (16, 'embedding dimension'),
        'manifold': ('Euclidean', 'which manifold to use, can be any of [Euclidean, PoincareBall]'),
        'c': (1.0, 'init hyperbolic radius'),
        'c-trainable': (1, '1 for trainable curvature'),
        'r': (2., 'fermi-dirac decoder parameter for lp'),
        't': (1., 'fermi-dirac decoder parameter for lp'),
        'pretrained-embeddings': (None, 'path to pretrained embeddings (.npy file) for Shallow node classification'),
        'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
        'num-layers': (2, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'use-att': (0, 'whether to use hyperbolic attention in model'),
        'double-precision': ('0', 'whether to use double precision'),
        'att-type': ('mlp', 'Specify Attention type, can bye any of [mlp, dist] for GAT.\
            Also [dense_mlp, dense_adjmask_mlp, dense_adjmask_dist, sparse_adjmask_dist] for HGCAE'), 
        'att-logit': (None, 'Specify logit for attention, can be any of [exp, sigmoid, tanh, ... from torch.<loigt>]') ,
        'beta': (0., 'coefficient of feature-distance in when --att-type dist') ,
        'non-param-dec': ('fermidirac', 'Non-param decoder for link prediction. [fermidirac, innerproduct]') ,
        'num-dec-layers': (2, 'number of hidden layers in encoder'),
        'node-cluster': (0, 'Set test,val prop to 0 and adjust [log/eval]_freq and patience'),
        'visualize-dim2': (0, 'If set, fix dim==2 for visualization.')
    },
    'data_config': {
        'dataset': ('cora', 'which dataset to use'),
        'val-prop': (0.05, 'proportion of validation edges for link prediction'),
        'test-prop': (0.1, 'proportion of test edges for link prediction'),
        'use-feats': (1, 'whether to use node features or not'),
        'normalize-feats': (1, 'whether to normalize input node features'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
        'split-seed': (1234, 'seed for data splits (train/test/val)'),
    },
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
