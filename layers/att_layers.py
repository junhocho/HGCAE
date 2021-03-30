"""Attention layers (some modules are copied from https://github.com/Diego999/pyGAT.)"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import manifolds 

def HypAggAtt(in_features, manifold, dropout, act=None, att_type=None, att_logit=None, beta=0):
    att_logit = get_att_logit(att_logit, att_type)
    return GeometricAwareHypAggAtt(in_features, manifold, dropout, lambda x: x, att_logit=att_logit, beta=beta)

class GeometricAwareHypAggAtt(nn.Module):
    def __init__(self, in_features, manifold, dropout, act, att_logit=torch.tanh, beta=0.):
        super(GeometricAwareHypAggAtt, self).__init__()
        self.dropout = dropout
        self.att_logit=att_logit
        self.special_spmm = SpecialSpmm()


        self.m = manifold
        self.beta = nn.Parameter(torch.Tensor([1e-6]))
        self.con = nn.Parameter(torch.Tensor([1e-6]))
        self.act = act
        self.in_features = in_features

    def forward (self, x, adj, c=1):
        n = x.size(0)
        edge = adj._indices()

        assert not torch.isnan(self.beta).any()
        edge_h = self.beta * self.m.sqdist(x[edge[0, :], :], x[edge[1, :], :], c) + self.con

        self.edge_h = edge_h
        assert not torch.isnan(edge_h).any()
        edge_e = self.att_logit(edge_h)
        self.edge_e = edge_e
        ones = torch.ones(size=(n, 1))
        if x.is_cuda:
            ones = ones.to(x.device)
        e_rowsum = self.special_spmm(edge, abs(edge_e), torch.Size([n, n]), ones) + 1e-10

        return edge_e, e_rowsum

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    # generate sparse matrix from `indicex, values, shape` and matmul with b
    # Previously, `AXW` computing did not need bp to `A`.
    # To trian attention of `A`, now bp through sparse matrix needed.
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape, device=b.device) # make sparse matrix shaped of `NxN` 
        ctx.save_for_backward(a, b) # save sparse matrix for bp
        ctx.N = shape[0] # number of nodes
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        assert not torch.isnan(grad_output).any()

        # grad_output : Nxd  gradient
        # a : NxN adj(attention) matrix, b: Nxd node feature
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :] # flattening (x,y) --> nx + y
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

def get_att_logit(att_logit, att_type):
    if att_logit:
        att_logit = getattr(torch, att_logit)
    return att_logit
