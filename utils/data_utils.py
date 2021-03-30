"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from scipy import sparse
import logging

def load_data(args, datapath):
    ## Load data
    data = load_data_lp(args.dataset, args.use_feats, datapath)
    adj = data['adj_train']

    ## TAKES a lot of time

    if args.node_cluster == 1:
        task = 'nc'
    else:
        task = 'lp'
    cached_dir = os.path.join('/root/tmp', task, args.dataset,
            f"seed{args.split_seed}-val{args.val_prop}-test{args.test_prop}")

    if not os.path.isdir(cached_dir):
        logging.info(f"Caching at `{cached_dir}`randomly masked edges")
        os.makedirs(cached_dir, exist_ok=True)
        adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                adj, args.val_prop, args.test_prop, args.split_seed
        )
        if args.val_prop + args.test_prop > 0:
            torch.save(val_edges, os.path.join(cached_dir, 'val_edges.pth'))
            torch.save(val_edges_false, os.path.join(cached_dir, 'val_edges_false.pth'))
            torch.save(test_edges, os.path.join(cached_dir, 'test_edges.pth'))
            torch.save(test_edges_false, os.path.join(cached_dir, 'test_edges_false.pth'))

        torch.save(train_edges, os.path.join(cached_dir, 'train_edges.pth'))
        torch.save(train_edges_false, os.path.join(cached_dir, 'train_edges_false.pth'))
        sparse.save_npz(os.path.join(cached_dir, "adj_train.npz"), adj_train)

        st0 = np.random.get_state()
        np.save(os.path.join(cached_dir, 'np_state.npy'), st0)

    else:
        logging.info(f"Loading from `{cached_dir}` randomly masked edges")
        if args.val_prop + args.test_prop > 0:
            val_edges = torch.load(os.path.join(cached_dir, 'val_edges.pth'))
            val_edges_false = torch.load(os.path.join(cached_dir, 'val_edges_false.pth'))
            test_edges = torch.load(os.path.join(cached_dir, 'test_edges.pth'))
            test_edges_false = torch.load(os.path.join(cached_dir, 'test_edges_false.pth'))

        adj_train = sparse.load_npz(os.path.join(cached_dir, "adj_train.npz"))
        train_edges = torch.load(os.path.join(cached_dir, 'train_edges.pth'))
        train_edges_false = torch.load(os.path.join(cached_dir, 'train_edges_false.pth'))

        st0 = np.load(os.path.join(cached_dir, 'np_state.npy'))
        np.random.set_state(st0)

    ## TAKES a lot of time
    data['adj_train'] = adj_train
    data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
    if args.val_prop + args.test_prop > 0:
        data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
        data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    all_info=""

    ## Adj matrix
    adj = data['adj_train']
    data['adj_train_enc'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )

    if args.lambda_rec:
        data['adj_train_dec'] = rowwise_normalizing(data['adj_train'])

    adj_2hop = get_adj_2hop(adj)
    data['adj_train_enc_2hop'] = symmetric_laplacian_smoothing(adj_2hop)

    # NOTE : Re-adjust labels
    # Some data omit `0` class, thus n_classes are wrong with `max(labels)+1`
    label_set = set(list(data['labels'].numpy()))
    label_convert_table = {list(label_set)[i]:i for i in range(len(label_set))}
    for label_prev, label_now in label_convert_table.items():
        data['labels'][data['labels']==label_prev] = label_now
    args.n_classes = int(data['labels'].max() + 1)

    data['idx_all'] =  range(data['features'].shape[0])
    data_info = "Dataset {} Loaded : dimensions are adj:{}, edges:{}, features:{}, labels:{}\n".format(
            args.dataset, data['adj_train'].shape, data['adj_train'].sum(), data['features'].shape, data['labels'].shape)
    data['info'] = data_info


    return data

def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    return adj, features

def get_adj_2hop(adj):
    adj_self = adj + sp.eye(adj.shape[0])
    adj_2hop = adj_self.dot(adj_self)
    adj_2hop.data = np.clip(adj_2hop.data, 0, 1)
    adj_2hop = adj_2hop - sp.eye(adj.shape[0]) - adj
    return adj_2hop

def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def symmetric_laplacian_smoothing(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = adj + sp.eye(adj.shape[0])  # self-loop

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def rowwise_normalizing(adj):
    """Row-wise normalize adjacency matrix."""
    adj = adj + sp.eye(adj.shape[0])  # self-loop
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum, -1.0).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    return adj.dot(d_mat_inv).transpose().tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()   #  LONG
    neg_edges = np.array(list(zip(x, y)))   #  EVEN LONGER
    np.random.shuffle(neg_edges)  # ALSO LONG

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def load_data_lp(dataset, use_feats, data_path):
    if dataset in ['cora', 'pubmed', 'citeseer']:

        # adj, features, labels = load_citation_data(dataset, use_feats, data_path)[:3]
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed=None
        )
    elif dataset in ['cora_planetoid', 'pubmed_planetoid']:
        from torch_geometric.datasets import Planetoid
        import torch_geometric as tg
        import scipy.sparse as sp
        if dataset == 'cora':
            name = 'Cora'
        elif dataset == 'pubmed':
            name = 'Pubmed'
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        loaded_dataset = Planetoid(root='/root/tmp/'+name, name='Cora')
        adj = tg.utils.to_scipy_sparse_matrix(loaded_dataset.data.edge_index)
        adj = sp.coo_matrix.asformat(adj, format='csr')
        features = sp.lil_matrix(loaded_dataset.data.x.numpy())
        labels = loaded_dataset.data.y.numpy()
    elif 'amazon' in dataset:
        from torch_geometric.datasets import Amazon
        import torch_geometric as tg
        import scipy.sparse as sp
        if dataset == 'amazon-photo':
            name = 'Photo'
        elif dataset == 'amazon-computers':
            name = 'Computers'
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        loaded_dataset = Amazon(root='/root/tmp/' + name, name=name)
        adj = tg.utils.to_scipy_sparse_matrix(loaded_dataset.data.edge_index)
        adj = sp.coo_matrix.asformat(adj, format='csr')
        features = sp.lil_matrix(loaded_dataset.data.x.numpy())
        labels = loaded_dataset.data.y.numpy()
    elif dataset == 'BlogCatalog':
        import scipy.io as sio
        import scipy.sparse as sp
        data = sio.loadmat('./data/BlogCatalog/BlogCatalog.mat')
        features = sp.lil_matrix(data['Attributes'])
        labels = np.squeeze(data['Label'])
        adj = sp.csr_matrix(data['Network'])
    elif dataset == 'wiki':
        import scipy.sparse as sp
        features = np.loadtxt('./data/wiki/wiki_feature.txt')
        features = sp.coo_matrix((features[:, 2], (features[:, 0].astype(int), features[:, 1].astype(int))))
        features = sp.lil_matrix(features)
        adj = np.loadtxt('./data/wiki/wiki_graph.txt')
        adj = np.ndarray.tolist(adj)
        adj = nx.from_edgelist(adj)
        adj = nx.adjacency_matrix(adj)
        labels = np.loadtxt('./data/wiki/wiki_group.txt')
        labels = labels[:, 1]
        labels = labels.astype(np.int64)
        labels = np.squeeze(np.reshape(labels, (2405, 1)) - 1)
    elif 'PICA' in dataset:

        if 'ImageNet10' in dataset:
            dataset_lower = 'imagenet10'
            dataset_name = 'PICA-ImageNet10'
        elif 'ImageNetDog' in dataset:
            dataset_lower = 'imagenetdog'
            dataset_name = 'PICA-ImageNetDog'

        if 'feat10' in dataset:
            name = 'picafeat10_{}'.format(dataset_lower)
        elif 'feat70' in dataset:
            name = 'picafeat70_{}'.format(dataset_lower)
        elif 'feat512' in dataset:
            name = 'picafeat512_{}'.format(dataset_lower)

        orig_dataset = dataset
        suffix = dataset.split(dataset_name)[-1]
        dataset = dataset_name


        print('name : {},  suffix : {}'.format(name,suffix))

        y_true = np.load('./data/{}/label.npy'.format(dataset))
        y_true = y_true.astype('int64')
        labels = y_true

        features = np.load('./data/{}/{}.npy'.format(dataset,name))
        import scipy.sparse as sp
        features = sp.lil_matrix(features)

        A  = sp.load_npz('./data/{}/A{}.npz'.format(dataset,suffix))



        adj = A.astype('float64')

        labels = torch.LongTensor(labels)
        data = {'adj_train': adj, 'features': features, 'labels': labels}
        return data
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels}
    return data

# ############### DATASETS ####################################


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)
    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels
