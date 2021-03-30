from models.base_models import LPModel
import logging
import optimizers
import torch
import numpy as np
import os
import time
from utils.train_utils import get_dir_name, format_metrics

import json
import pickle


from utils.data_utils import sparse_mx_to_torch_sparse_tensor 

class Solver(object):
    def __init__(self, args, data):
        np.random.set_state(args.np_seed)
        torch.manual_seed(args.seed)
        if int(args.double_precision):
            torch.set_default_dtype(torch.float64)
        if int(args.cuda) >= 0:
            torch.cuda.manual_seed(args.seed)
        args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
        args.patience = args.epochs if not args.patience else  int(args.patience)
        logging.info(f'Using: {args.device}')
        logging.info("Using seed {}.".format(args.seed))

        args.n_nodes, args.feat_dim = data['features'].shape
        logging.info(f'Num classes: {args.n_classes}')
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        Model = LPModel

        if not args.lr_reduce_freq:
            args.lr_reduce_freq = args.epochs

        # Model and optimizer
        model = Model(args)
        logging.info(str(model))
        optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                        weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(args.lr_reduce_freq),
            gamma=float(args.gamma)
        )
        tot_params = sum([np.prod(p.size()) for p in model.parameters()])
        logging.info(f"Total number of parameters: {tot_params}")
        for x, val in data.items():
            if 'adj' in x:
                data[x] = sparse_mx_to_torch_sparse_tensor(data[x])
        if args.cuda is not None and int(args.cuda) >= 0 :
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
            model = model.to(args.device)
            for x, val in data.items():
                if torch.is_tensor(data[x]):
                    data[x] = data[x].to(args.device)
        # Train model
        t_total = time.time()
        counter = 0
        best_val_metrics = model.init_metric_dict()
        best_test_metrics = {}
        best_emb = None

        self.adj_train_enc = data['adj_train_enc']

        self.t_total = t_total
        self.counter = counter
        self.best_val_metrics = best_val_metrics
        self.best_test_metrics = best_test_metrics
        self.best_emb = best_emb

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.data = data
        self.lr_scheduler = lr_scheduler


    def fit(self):
        args = self.args
        model = self.model
        optimizer = self.optimizer
        data = self.data
        lr_scheduler = self.lr_scheduler
        save_dir = self.args.save_dir

        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            embeddings = model.encode(data['features'], self.adj_train_enc)
            train_metrics = model.compute_metrics(embeddings, data, 'train', epoch)
            train_metrics['loss'].backward()
            if args.grad_clip is not None:
                max_norm = float(args.grad_clip)
                all_params = list(model.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            optimizer.step()
            lr_scheduler.step()
            with torch.no_grad():
                if (epoch + 1) % args.log_freq == 0:
                    logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                           'lr: {}'.format(lr_scheduler.get_lr()[0]),
                                           format_metrics(train_metrics, 'train'),
                                           'time: {:.4f}s'.format(time.time() - t)
                                           ]))
                if (epoch + 1) % args.eval_freq == 0:
                    model.eval()
                    embeddings = model.encode(data['features'], self.adj_train_enc)
                    if args.node_cluster != 1:
                        ## Link Prediction Task that use train/val/test
                        val_metrics = model.compute_metrics(embeddings, data, 'val')
                        if (epoch + 1) % args.log_freq == 0:
                            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
                        if model.has_improved(self.best_val_metrics, val_metrics):
                            self.best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                            self.best_emb = embeddings
                            if args.save:
                                np.save(os.path.join(save_dir, 'embeddings.npy'), self.best_emb.cpu().detach().numpy())
                            self.best_val_metrics = val_metrics
                            self.best_val_metrics['epoch'] = epoch + 1
                            self.counter = 0
                            # logging.info("improved")
                        else:
                            # logging.info("not improved :"+str(self.counter))
                            self.counter += 1
                            if self.counter >= args.patience and epoch > args.min_epochs:  # NOTE : fixed when improve only epoch0
                                logging.info("Early stopping")
                                break
                    else:
                        ## Node Clustering Task that use 100 % trainset.
                        if self.best_test_metrics.get('loss', 999) > train_metrics['loss']:
                            '''
                            # NOTE : when kmeans calculated, affect np.state and takes time, and not fair to monitor it.
                            # metrics_clustering = model.eval_cluster(embeddings, data, 'all')
                            # logging.info(" ".join(["Cluster results:", format_metrics(metrics_clustering, 'all')]))
                            '''
                            self.best_emb = embeddings
                            logging.info("Best loss found")
                            self.best_test_metrics['loss'] = train_metrics['loss']
                            self.best_test_metrics['epoch'] = epoch + 1
                            self.counter = 0
                        else:
                            self.counter += 1
                            if self.counter >= args.patience and epoch > args.min_epochs:  # NOTE : fixed when improve only epoch0
                                logging.info("Early stopping")
                                break


        logging.info("Optimization Finished!")
        logging.info("Total time elapsed: {:.4f}s".format(time.time() - self.t_total))

    def eval(self):
        model = self.model
        data = self.data
        args = self.args
        save_dir = self.args.save_dir
        args.np_seed = None


        if not self.best_test_metrics and args.test_prop > 0:
            model.eval()
            self.best_emb = model.encode(data['features'], self.adj_train_enc)
            self.best_test_metrics = model.compute_metrics(self.best_emb, data, 'test')

        ## CLUSTERING EVAL START
        if args.node_cluster == 1:
            metrics_clustering, pred_label = model.eval_cluster(self.best_emb, data, 'all')
            self.best_test_metrics.update(metrics_clustering)
            self.best_val_metrics = self.best_test_metrics
        else:
            metrics_clustering, pred_label = model.eval_cluster(self.best_emb, data, 'all')
            self.best_test_metrics.update(metrics_clustering)
        ## CLUSTERING EVAL END
        logging.info(" ".join(["Val set results:", format_metrics(self.best_val_metrics, 'val')]))
        logging.info(" ".join(["Test set results:", format_metrics(self.best_test_metrics, 'test')]))
        if args.save:
            np.save(os.path.join(save_dir, 'embeddings.npy'), self.best_emb.cpu().detach().numpy())
            if hasattr(model.encoder, 'att_adj'):
                filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
                pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
                print('Dumped attention adj: ' + filename)

            json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
            logging.info(f"Saved model in {save_dir}")

        return self.best_emb, pred_label
