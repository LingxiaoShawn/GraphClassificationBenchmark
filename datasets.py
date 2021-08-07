import os.path as osp
import numpy as np 

import torch
from torch_geometric.datasets import TUDataset

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import degree, from_scipy_sparse_matrix
import torch.nn.functional as F

import scipy.io as sio


class MatDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name 
        super(MatDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name)
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')
    @property
    def raw_file_names(self):
        names = ['all_graphs.mat', 'all_attributes.mat', 'all_labels.mat', 'graph_labels.txt']
        return [f'{self.name}_{name}.txt' for name in names] 
    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = mat_to_pyg(self.root, self.name)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'

def mat_to_pyg(root, dataset):
    G_path = osp.join(root, dataset, f'{dataset}_all_graphs.mat')
    attr_path = osp.join(root, dataset, f'{dataset}_all_attributes.mat')
    label_path = osp.join(root, dataset, f'{dataset}_all_labels.mat')
    Glabel_path = osp.join(root, dataset, f'{dataset}_graph_labels.txt')
    labeled = True if osp.exists(label_path) else False
    attributed = True if osp.exists(attr_path) else False
    Gs = sio.loadmat(G_path)['all_graphs'].flatten()
    Fs = sio.loadmat(attr_path)['all_attributes'].flatten() if attributed else sio.loadmat(label_path)['all_labels'].flatten()
    if labeled:
        Fs = [f.todense() for f in Fs]
    Fs = torch.from_numpy(np.stack([np.asarray(f) for f in Fs]).astype(np.float32))

    Ys = np.loadtxt(Glabel_path).astype(int)

    graphs = []
    for i, (G, F) in enumerate(zip(Gs, Fs)):
        edge_index, edge_attr = from_scipy_sparse_matrix(G.astype(np.float32))
        data = Data(edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([Ys[i]]), x=F)
        graphs.append(data)
    return graphs
        
class OneHotDegree(object):
    r"""Adds the node degree as one hot encodings to the node features.

    Args:
        max_degree (int): Maximum degree.
        in_degree (bool, optional): If set to :obj:`True`, will compute the
            in-degree of nodes instead of the out-degree.
            (default: :obj:`False`)
        cat (bool, optional): Concat node degrees to node features instead
            of replacing them. (default: :obj:`True`)
    """

    def __init__(self, max_degree, in_degree=False, cat=True):
        self.max_degree = max_degree
        self.in_degree = in_degree
        self.cat = cat

    def __call__(self, data):
        idx, x = data.edge_index[1 if self.in_degree else 0], data.x
        deg = truncate_degree(degree(idx, data.num_nodes, dtype=torch.long))
        deg = F.one_hot(deg, num_classes=self.max_degree + 1).to(torch.float)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg

        return data

def truncate_degree(degree):
    degree[ (100<=degree) & (degree <200) ] = 101
    degree[ (200<=degree) & (degree <500) ] = 102
    degree[ (500<=degree) & (degree <1000) ] = 103
    degree[ (1000<=degree) & (degree <2000) ] = 104
    degree[ degree >= 2000] = 105
    return degree

def get_dataset(name,  cleaned=False):
    root = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data')
    if name in ['BANDPASS', 'congress-LS', 'congress-sim3','mig-sim3']:
        dataset = MatDataset(root, name)
    else:
        dataset = TUDataset(root, name, cleaned=cleaned)

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset: # ATTENTION: use dataset_raw instead of downsampled version!
            degs += [truncate_degree(degree(data.edge_index[0], dtype=torch.long))]
            max_degree = max(max_degree, degs[-1].max().item())
        dataset.transform = OneHotDegree(max_degree)

    return dataset

if __name__ == "__main__":
    mat_to_pyg('data', 'congress-sim3')
