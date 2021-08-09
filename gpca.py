import torch
import torch.nn as nn
from torch_sparse import SparseTensor
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.inits import reset
import torch.nn.functional as F


def to_normalized_sparsetensor(edge_index, N, mode='DA'):
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5) 
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    if mode == 'DA':
        return deg_inv_sqrt.view(-1,1) * deg_inv_sqrt.view(-1,1) * adj
    if mode == 'DAD':
        return deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

def approximate_invphi_x(A, x, alpha=1, n_powers=10):
    invphi_x = x
    # power method
    for _ in range(n_powers):
        part1 = A.matmul(invphi_x)
        invphi_x = alpha/(1+alpha)*part1 + 1/(1+alpha)*x
    return invphi_x    

class GPCALayer(nn.Module):
    def __init__(self, nin, nout, alpha, center=True, n_powers=50, mode='DA'):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(nin, nout))
        self.bias = nn.Parameter(torch.FloatTensor(1, nout))
        self.nin = nin
        self.nhid = nout
        self.alpha = alpha
        self.center = center
        self.n_powers = n_powers
        self.mode = mode
        # init default parameters
        self.reset_parameters()
       
    
    def freeze(self, requires_grad=False):
        self.weight.requires_grad = requires_grad
        self.bias.requires_grad = requires_grad
                
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0) 

    def forward(self, data, return_invphi_x=False, minibatch=False, center=True):
        """
            Assume data.adj is SparseTensor and normalized
        """
        # inputs
        n = data.x.size(0)
        edge_index, x = data.edge_index, data.x
        A = to_normalized_sparsetensor(edge_index, n, self.mode)        
        if center:
            x = x - x.mean(dim=0)
        # calculate inverse of phi times x
        if return_invphi_x:
            if center:
                x = x - x.mean(dim=0) # center
            invphi_x = approximate_invphi_x(A, x, self.alpha, self.n_powers)
            return invphi_x, x
        else:     
            # AXW + bias
            invphi_x = approximate_invphi_x(A, x, self.alpha, self.n_powers)
            return invphi_x.mm(self.weight) + self.bias
    
    def init(self, full_data, center=True, posneg=False):
        """
        Init always use full batch, same as inference/test. 
        """
        self.eval() 
        with torch.no_grad():
            invphi_x, x = self.forward(full_data, return_invphi_x=True, center=center)
            eig_val, eig_vec = torch.symeig(x.t().mm(invphi_x), eigenvectors=True)
            if self.nhid <= (int(posneg)+1)*self.nin:
                weight = torch.cat([eig_vec[:,-self.nhid//2:], -eig_vec[:,-self.nhid//2:]], dim=-1) \
                      if posneg else eig_vec[:, -self.nhid:] #when 

            elif self.nhid <= 2*(int(posneg)+1)*self.nin:
                eig_val1, eig_vec1 = torch.symeig(x.t().mm(x), eigenvectors=True)
                m = self.nhid % ((int(posneg)+1)*self.nin)
                weight = torch.cat([eig_vec, -eig_vec, eig_vec1[:, -m//2:], -eig_vec1[:, -m//2:]], dim=-1) \
                      if posneg else torch.cat([eig_vec, eig_vec1[:, -m:]], dim=-1)
                                
            elif self.nhid <= 3*(int(posneg)+1)*self.nin:
                eig_val1, eig_vec1 = torch.symeig(x.t().mm(x), eigenvectors=True)
                eig_val2, eig_vec2 = torch.symeig(invphi_x.t().mm(invphi_x), eigenvectors=True)
                m = self.nhid % ((int(posneg)+1)*self.nin)
                weight = torch.cat([eig_vec, eig_vec1, eig_vec2[:, -m//2:]
                             -eig_vec, -eig_vec1, -eig_vec2[:, -m//2:]], dim=-1) \
                      if posneg else torch.cat([eig_vec, eig_vec1, eig_vec2[:, -m:]], dim=-1)
            else:
                raise ValueError('Larger hidden size is not supported yet.')

            # assign 
            self.weight.data = weight

class GPCANet(nn.Module):
    def __init__(self, dataset, num_layers, hidden, alpha=10,
                 dropout=0.5, n_powers=5, center=True, act='ReLU', 
                 mode='DA', out_nlayer=2, **kwargs):
        super().__init__()
        nclass = dataset.num_classes
        nfeat = dataset.num_features
        nlayer = num_layers
        nhid = hidden

        self.convs = torch.nn.ModuleList()
        for i in range(nlayer-1):
            self.convs.append(
                GPCALayer(nhid if i>0 else nfeat, nhid, alpha, center, n_powers, mode))
        # last layer
        self.convs.append(
            GPCALayer(nhid if nlayer>1 else nfeat, 
                      nclass if out_nlayer==0 else nhid, alpha, center, n_powers, mode))
        """
        out_nlayer = 0 should only be used for non frezzed setting
        """
         
        self.dropout = nn.Dropout(dropout)
        self.relu = getattr(nn,act)()
        
        # fc layers
        if out_nlayer == 0:
            self.out_mlp = nn.Identity()
        elif out_nlayer == 1:
            self.out_mlp = nn.Sequential(nn.Linear(nhid, nclass)) 
        else: 
            self.out_mlp = nn.Sequential(nn.Linear(nhid, nhid), self.relu, self.dropout, nn.Linear(nhid, nclass))    

        self.freeze_status = False # only support full batch    
        
    def freeze(self, requires_grad=False):
        self.freeze_status = not requires_grad
        for conv in self.convs:
            conv.freeze(requires_grad)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        reset(self.out_mlp)


    def forward(self, data):
        # inputs
#         A, x, y, train_mask = data.adj, data.x, data.y, data.train_mask
#         n, c = data.num_nodes, data.num_classes

        original_x = data.x
        for i, conv in enumerate(self.convs):
            x = conv(data)
            if not self.freeze_status:
                x = self.relu(x)
            data.x = x

        x = global_mean_pool(x, data.batch)    
        out = self.out_mlp(x)
        data.x = original_x # restore 
        return F.log_softmax(out, dim=-1)
        
    def init(self, full_data, center=True, posneg=False, **kwargs):
        """
        Init always use full batch, same as inference/test. 
        Btw, should we increase the scale of weight based on relu scale?
        Think about this later. 
        """
        self.eval()
        with torch.no_grad():
            original_x = full_data.x
            for i, conv in enumerate(self.convs):
                # init
                conv.init(full_data, center, posneg) # init using GPCA
                # next layer
                x = conv(full_data)
                #----- init without relu and dropout?
                full_data.x = self.relu(x) if posneg else x
#                 x = self.dropout(x)
            full_data.x = original_x # restore 
