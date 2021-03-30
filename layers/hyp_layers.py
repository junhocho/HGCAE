"""
Hyperbolic layers.
Major codes of hyperbolic layers are from HGCN
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from layers.att_layers import HypAggAtt, SpecialSpmm


def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)

    dims = [args.feat_dim]
    # Check layer_num and hdden_dim match
    if args.num_layers > 1:
        hidden_dim = [int(h) for h in args.hidden_dim.split(',')]
        if args.num_layers != len(hidden_dim) + 1:
            raise RuntimeError('Check dimension hidden:{}, num_layers:{}'.format(args.hidden_dim, args.num_layers) )
        dims = dims + hidden_dim

    dims += [args.dim]
    acts += [act]
    n_curvatures = args.num_layers
    if args.c_trainable == 1: # NOTE : changed from # if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([args.c]).to(args.device)) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures



class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att,
            att_type='sparse_adjmask_dist', att_logit=torch.exp, beta=0., decode=False):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, use_att, out_features, dropout, att_type=att_type, att_logit=att_logit, beta=beta, decode=decode)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        self.decode = decode

    def forward(self, input):
        x, adj = input
        assert not torch.isnan(self.hyp_act.c_in).any()
        if self.hyp_act.c_out:
            assert not torch.isnan(self.hyp_act.c_out).any()
        assert not torch.isnan(x).any()
        h = self.linear.forward(x)
        assert not torch.isnan(h).any()
        h = self.agg.forward(h, adj, prev_x=x)
        assert not torch.isnan(h).any()
        h = self.hyp_act.forward(h)
        assert not torch.isnan(h).any()
        output = h, adj
        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        # self.bias = nn.Parameter(torch.Tensor(out_features))
        self.bias = nn.Parameter(torch.Tensor(1, out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias: 
            bias = self.bias
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
                self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, use_att, in_features, dropout, att_type='sparse_adjmask_dist', att_logit=None, beta=0, decode=False):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.use_att = use_att

        self.in_features = in_features
        self.dropout = dropout
        if use_att:
            self.att = HypAggAtt(in_features, manifold, dropout, act=None, att_type=att_type, att_logit=att_logit, beta=beta)
            self.att_type = att_type

            self.special_spmm = SpecialSpmm()
        self.decode = decode

    def forward(self, x, adj, prev_x=None):

        if self.use_att:
            dist = 'dist' in self.att_type
            if dist:
                if 'sparse' in self.att_type:
                    if self.decode:
                        # NOTE : AGG(prev_x)
                        edge_e, e_rowsum = self.att(prev_x, adj, self.c) # SparseAtt
                    else:
                        # NOTE : AGG(x)
                        edge_e, e_rowsum = self.att(x, adj, self.c) # SparseAtt
                    self.edge_e = edge_e
                    self.e_rowsum = e_rowsum
                    ## SparseAtt
                    x_tangent = self.manifold.logmap0(x, c=self.c)
                    N = x.size()[0]
                    edge = adj._indices()
                    support_t = self.special_spmm(edge, edge_e, torch.Size([N, N]), x_tangent) 
                    assert not torch.isnan(support_t).any()
                    support_t = support_t.div(e_rowsum)
                    assert not torch.isnan(support_t).any()
                    output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
                else:
                    adj = self.att(x, adj, self.c) # DenseAtt
                    x_tangent = self.manifold.logmap0(x, c=self.c)
                    support_t = torch.spmm(adj, x_tangent)
                    output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
            else:
                ## MLP attention
                x_tangent = self.manifold.logmap0(x, c=self.c)
                adj = self.att(x_tangent, adj)
                support_t = torch.spmm(adj, x_tangent)
                output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        else:
            x_tangent = self.manifold.logmap0(x, c=self.c)
            support_t = torch.spmm(adj, x_tangent)
            output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)

        return output

    def extra_repr(self):
        return 'c={}, use_att={}, decode={}'.format(
                self.c, self.use_att, self.decode
        )


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        if self.manifold.name == 'PoincareBall':
            if self.c_out:
                xt = self.manifold.activation(x, self.act, self.c_in, self.c_out)
                return xt
            else:
                xt = self.manifold.logmap0(x, c=self.c_in)
                return xt
        else:
            NotImplementedError("not implemented")

    def extra_repr(self):
        return 'Manifold={},\n c_in={},\n act={},\n c_out={}'.format(
                self.manifold.name, self.c_in, self.act.__name__, self.c_out
        )
