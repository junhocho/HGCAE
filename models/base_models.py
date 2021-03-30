import manifolds
import models.encoders as encoders
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import layers.hyp_layers as hyp_layers
from models.decoders import model2decoder
from layers.layers import FermiDiracDecoder, InnerProductDecoder
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.eval_utils import acc_f1
from sklearn import cluster
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        if args.model not in ['HGCAE', 'HNN', 'Shallow'] and args.manifold != 'Euclidean':
            raise RuntimeError('Make sure the manifold of model:{} is Euclidean, not {}'.format(args.model, args.manifold))
        self.manifold = getattr(manifolds, self.manifold_name)()
        self.nnodes = args.n_nodes
        self.n_classes = args.n_classes
        self.encoder = getattr(encoders, args.model)(self.c, args)
        self.num_layers=args.num_layers

        # Embedding c
        self.hyperbolic_embedding = True if args.use_att and args.model == 'HGCAE' else False

        if args.non_param_dec == 'fermidirac':
            self.decoder_type = 'FermiDiracDecoder'
            self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        else:
            self.decoder_type = 'InnerProductDecoder'
            self.dc = InnerProductDecoder(dropout=0, act=torch.sigmoid)


    def encode(self, x, adj):
        h = self.encoder.encode(x, adj)
        return h

    def pred_link_score(self, h, idx):  # for LP,REC 
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        if self.decoder_type == 'FermiDiracDecoder':
            '''
            NOTE : previously, output embedding c was fixed, not learnable always.
            print(self.c)
            sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
            '''
            # NOTE : use last embedding curvature
            if self.manifold_name != 'Euclidean':
                emb_c = self.encoder.layers[-1].hyp_act.c_out
            else:
                emb_c = None
            sqdist = self.manifold.sqdist(emb_in, emb_out, emb_c)
            probs = self.dc.forward(sqdist)
        else:
            # NOTE : not implemented for hyperbolics
            probs = self.dc.forward(emb_in, emb_out)

        return probs

    def decode(self, h, adj, idx): # REC
        output = self.decoder.decode(h, adj)
        return output


    def eval_cluster(self, embeddings, data, split):
        if self.hyperbolic_embedding:
            emb_c = self.encoder.layers[-1].hyp_act.c_out
            embeddings = self.manifold.logmap0(embeddings.to(emb_c.device), c=emb_c).cpu()

        idx = data[f'idx_{split}']
        n_classes = self.n_classes

        embeddings_to_cluster = embeddings[idx].detach().cpu().numpy()
        gt_label = data['labels'][idx].cpu().numpy()

        kmeans = cluster.KMeans(n_clusters=n_classes, algorithm='auto')
        kmeans.fit(embeddings_to_cluster)
        pred_label = kmeans.fit_predict(embeddings_to_cluster)

        from munkres import Munkres
        def best_map(L1,L2):
            #L1 should be the groundtruth labels and L2 should be the clustering labels we got
            Label1 = np.unique(L1)
            nClass1 = len(Label1)
            Label2 = np.unique(L2)
            nClass2 = len(Label2)
            nClass = np.maximum(nClass1,nClass2)
            G = np.zeros((nClass,nClass))
            for i in range(nClass1):
                ind_cla1 = L1 == Label1[i]
                ind_cla1 = ind_cla1.astype(float)
                for j in range(nClass2):
                    ind_cla2 = L2 == Label2[j]
                    ind_cla2 = ind_cla2.astype(float)
                    G[i,j] = np.sum(ind_cla2 * ind_cla1)
            m = Munkres()
            index = m.compute(-G.T)
            index = np.array(index)
            c = index[:,1]
            newL2 = np.zeros(L2.shape)
            for i in range(nClass2):
                newL2[L2 == Label2[i]] = Label1[c[i]]
            return newL2


        def err_rate(gt_s, s):
            c_x = best_map(gt_s, s)
            err_x = np.sum(gt_s[:] !=c_x[:])
            missrate = err_x.astype(float) / (gt_s.shape[0])
            return missrate


        acc = 1-err_rate(gt_label, pred_label)
        # acc = accuracy_score(gt_label, pred_label)
        nmi = normalized_mutual_info_score(gt_label, pred_label, average_method='arithmetic')
        ari = adjusted_rand_score(gt_label, pred_label)
    
        metrics = { 'cluster_acc': acc, 'nmi': nmi, 'ari': ari}
        return metrics, pred_label


    def compute_metrics(self, embeddings, data, split, epoch=None):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError

class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.nb_false_edges = args.nb_false_edges
        self.positive_edge_samplig = True
        if self.positive_edge_samplig:
            self.nb_edges = min(args.nb_edges, 5000) # NOTE : be-aware too dense edges
        else:
            self.nb_edges = args.nb_edges

        if args.lambda_rec > 0:
            self.num_dec_layers = args.num_dec_layers
            self.lambda_rec = args.lambda_rec
            c = self.encoder.curvatures if hasattr(self.encoder, 'curvatures') else args.c ### handle HNN
            self.decoder = model2decoder[args.model](c, args, 'rec')
        else:
            self.lambda_rec = 0
            
        if args.lambda_lp > 0:
            self.lambda_lp = args.lambda_lp
        else:
            self.lambda_lp = 0

    def compute_metrics(self, embeddings, data, split, epoch=None):
        if split == 'train':
            num_true_edges = data[f'{split}_edges'].shape[0]
            if self.positive_edge_samplig and num_true_edges > self.nb_edges:
                edges_true = data[f'{split}_edges'][np.random.randint(0, num_true_edges, self.nb_edges)]
            else:
                edges_true = data[f'{split}_edges']
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_true = data[f'{split}_edges']
            edges_false = data[f'{split}_edges_false']

        pos_scores = self.pred_link_score(embeddings, edges_true)
        neg_scores = self.pred_link_score(embeddings, edges_false)
        assert not torch.isnan(pos_scores).any()
        assert not torch.isnan(neg_scores).any()
        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}

        assert not torch.isnan(loss).any()
        if self.lambda_rec:
            idx = data['idx_all']
            recon = self.decode(embeddings, data['adj_train_dec'], idx) ## NOTE : adj
            assert not torch.isnan(recon).any()
            if self.num_dec_layers == self.num_layers:
                target = data['features'][idx]
            elif self.num_dec_layers == self.num_layers - 1: 
                target = self.encoder.features[0].detach()[idx]
            else:
                raise RuntimeError('num_dec_layers only support 1,2')
            loss_rec = self.lambda_rec * torch.nn.functional.mse_loss(recon[idx], target , reduction='mean')
            assert not torch.isnan(loss_rec).any()
            loss_lp = loss * self.lambda_lp
            metrics.update({'loss': loss_lp + loss_rec, 'loss_rec': loss_rec, 'loss_lp': loss_lp})

        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])
