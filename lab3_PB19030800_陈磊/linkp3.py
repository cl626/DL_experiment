import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim 
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import numpy as np


from torch_geometric.nn import GCNConv                      #图卷积层
from torch_geometric.data import Data                       #图结构加载
from torch_geometric.utils import train_test_split_edges    #训练集划分
from torch_geometric.utils import negative_sampling         #负采样
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import PairNorm

from typing_extensions import Literal, TypedDict
from torch_geometric.typing import torch_sparse
from torch_geometric.nn import PairNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj,OptTensor,SparseTensor
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops 
from torch_geometric.utils import scatter,spmm
from torch_geometric.utils.num_nodes import maybe_num_nodes

import pdb

device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

pos_to_index={}
index_to_pos={}
cls_to_idx={}

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, 
             add_self_loops=True, flow="source_to_target", dtype=None):

    fill_value = 1.

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:  #添加自环
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight

class myGCNConv2(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 add_self_loops: bool = True,bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()         #卷积层
        zeros(self.bias)                    #偏置层

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        edge_index, edge_weight = gcn_norm(  # yapf: disable
            edge_index, edge_weight, x.size(self.node_dim),
            self.add_self_loops, self.flow, x.dtype)

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

class CustomDataset(Dataset):
    def __init__(self, txt_file1, transform=None):
        feature = []
        label= []
        pos=0
        with open(txt_file1, 'r') as f:
            for line in f:
                line_tran=line.strip().split('\t')
                if(line_tran[-1] not in cls_to_idx):
                    ci_len=len(cls_to_idx)
                    cls_to_idx[line_tran[-1]]=ci_len
                index,feature_i,label_i = int(line_tran[0]),[int(i) for i in line_tran[1:-1]],cls_to_idx[line_tran[-1]] #strip去除两端的空格和换行符,split代表去除空格
                feature.append(feature_i)
                label.append(label_i)
                pos_to_index[pos]=index;    index_to_pos[index]=pos
                pos=pos+1
        self.feature=torch.tensor(feature).float()
        self.label=torch.tensor(label).long()

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return self.feature[idx],self.label[idx]

# 加载数据集
nodeset = CustomDataset('./cora/cora.content')
print(f'size of dataset={len(nodeset)}')
# nodeloader = DataLoader(nodeset, batch_size=32, shuffle=True)

edge_index = [[],[]]
with open('./cora/cora.cites', 'r') as f:
    for line in f:
        line1=[int(i) for i in line.strip().split('\t')]
        cited,cite=line1[0],line1[1]
        # print(cited);   print(cite)
        edge_index[0].append(index_to_pos[cited])
        edge_index[1].append(index_to_pos[cite])
edge_index=torch.tensor(edge_index)
print(f"shape of edge={edge_index.shape}")

# 建立data数据集
tg_data=Data(x=nodeset.feature,edge_index=edge_index,y=nodeset.label)
print(f'tg_data{tg_data}')
transform=RandomLinkSplit(num_val=0.2,num_test=0.1,is_undirected=False)
train_data,val_data,test_data=transform(tg_data)
print(f'train_data={train_data}')
print(f'val_data={val_data}')
print(f'test_data={test_data}')
# pdb.set_trace()

class Net(torch.nn.Module):
    def __init__(self,p):
        super(Net, self).__init__()
        self.dropout = torch.nn.Dropout(p)
        self.conv1 = myGCNConv2(tg_data.num_features, 128,add_self_loops=True)
        self.conv2 = myGCNConv2(128, 64,add_self_loops=True)
        self.conv3 = myGCNConv2(128, 64,add_self_loops=True)
        self.pn = PairNorm()


    def encode(self,x, net_edge_index):
        x = self.pn(x)
        x = self.dropout(x)
        x = self.conv1(x, net_edge_index)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, net_edge_index)
        # x = x.relu()
        # x = self.dropout(x)
        # x = self.conv3(x, net_edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)      #xi,xj为边的两个端点的向量表示,sum(xi·xj)得到边的值，
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()   
        b=prob_adj>0
        # print(prob_adj.max());  print(prob_adj.min())
        return (prob_adj > 0).nonzero(as_tuple=False).t()               #得到整个图上边的预测

# 无向边的drop_edge
def drop_edge(single_edge_index, dropout_rate):
    # 计算需要丢弃的边数
    num_edges = single_edge_index.shape[1]
    num_drop = int(num_edges * dropout_rate)

    # 随机选择要丢弃的边
    remain_indices = torch.randperm(num_edges)[num_drop:]
    remain_single_edges = single_edge_index[:, remain_indices]
    reverse_edges = torch.stack([remain_single_edges[1],remain_single_edges[0]],dim=0)
    remain_edges=torch.cat([remain_single_edges,reverse_edges],dim=1)

    return remain_edges

# 有向边转无向边，得到single_edge
def dire_to_undir(edge_lable_index):
    single_edge={}
    for i in range(len(edge_lable_index[0])):
        if(((edge_lable_index[0][i],edge_lable_index[1][i]) not in single_edge.items()) and 
            ((edge_lable_index[1][i],edge_lable_index[0][i]) not in single_edge.items())):
            single_edge[edge_lable_index[0][i]]=edge_lable_index[1][i]

    single_edge_index=[[],[]]

    for key,value in single_edge.items():
        single_edge_index[0].append(key)
        single_edge_index[1].append(value)        

    single_edge_index=torch.tensor(single_edge_index)

    return single_edge_index

train_undir_edge_index=dire_to_undir(train_data.edge_label_index)

# directed_edge's drop_edge
def drop_edge3(single_edge_index, dropout_rate):
    # 计算需要丢弃的边数
    num_edges = single_edge_index.shape[1]
    num_drop = int(num_edges * dropout_rate)

    # 随机选择要丢弃的边
    remain_indices = torch.randperm(num_edges)[num_drop:]
    remain_single_edges = single_edge_index[:, remain_indices]
    reverse_edges = torch.stack([remain_single_edges[1],remain_single_edges[0]],dim=0)
    remain_edges=torch.cat([remain_single_edges,reverse_edges],dim=1)

    return remain_edges
# val_undir_edge_index=dire_to_undir(val_data.edge_label_index)
# test_undir_edge_index=dire_to_undir(test_data.edge_label_index)

# certify dimension

p=0.5

model, tg_data = Net(p).to(device), tg_data.to(device)
train_data, val_data, test_data=train_data.to(device) , val_data.to(device), test_data.to(device)

criterion=nn.CrossEntropyLoss() #交叉熵损失函数
optimizer = optim.Adam(params=model.parameters(), lr=0.01)

def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

global_num_nodes=len(train_data.x)
dp=0.1

def train():
    model.train()
    train_pos_edge_index = drop_edge3(train_undir_edge_index,dp)
    neg_edge_index = negative_sampling(
        edge_index=train_pos_edge_index,
        num_nodes=tg_data.num_nodes,
        num_neg_samples=int(tg_data.num_nodes**2/len(edge_index))*len(train_pos_edge_index[0]),
        force_undirected=True,
    )
    optimizer.zero_grad()
    z = model.encode(train_data.x,train_pos_edge_index)          #encode node_feature&adjacent matrix for node vector expression
    link_logits = model.decode(z, train_pos_edge_index,neg_edge_index)     #decode，节点的vector expression for edge connection
    link_probs = link_logits.sigmoid()                      #sigm
    link_labels = get_link_labels(train_pos_edge_index,neg_edge_index)     
    loss = torch.nn.functional.binary_cross_entropy_with_logits(link_logits, link_labels)
    # print(f'in train type)lable={type(link_labels)},link)probs={type(link_probs)}')
    perfs=roc_auc_score(link_labels.detach().numpy(), link_probs.detach().numpy())
    loss.backward()
    optimizer.step()
    return perfs,loss.item()


@torch.no_grad()
def eval(edge_label_index):
    model.eval()
    perfs = []
    pos_edge_index = edge_label_index
    neg_edge_index = negative_sampling(
        edge_index= pos_edge_index,
        num_nodes=len(val_data.x),      #train_data.x=val_data.x=test_data.x,只是train/val/test_data.edge_label_index不一样
        num_neg_samples=int(tg_data.num_nodes**2/len(edge_index))*len(pos_edge_index[0]),
        force_undirected=True,
    )
    z = model.encode(val_data.x,pos_edge_index)
    link_logits = model.decode(z, pos_edge_index, neg_edge_index)
    link_probs = link_logits.sigmoid()                      #sigmoid 从edge_value得到 edge_connection?
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(link_logits, link_labels)
    # print(f'in eval type)lable={type(link_labels)},link)probs={type(link_probs)}')
    perfs=roc_auc_score(link_labels.cpu(), link_probs.cpu())
    return perfs,loss.item()

best_val_perf = test_perf = 0
epoch_set=[]
train_loss_set=[]
train_auc_set=[]
val_loss_set=[]
val_auc_set=[]

for epoch in range(1, 50):
    if epoch%5==0:
        optimizer.param_groups[0]['lr']*=0.5
    train_perf, train_loss= train()
    val_perf, val_loss= eval(val_data.edge_label_index)
    log = 'Epoch: {}, Loss: {:.4f}, train_auc :{:.4f}, Val loss:{:.4f}, Val_auc: {:.4f}' 
    print(log.format(epoch, train_loss, train_perf, val_loss, val_perf))
    epoch_set.append(epoch)
    train_loss_set.append(train_loss)
    train_auc_set.append(train_perf)
    val_loss_set.append(val_loss)
    val_auc_set.append(val_perf)

plt.suptitle("link prediction", fontsize=14)
ax1 = plt.subplot(121)      #1*2中第一个子图
ax1.set_title("Loss")
ax1.plot(epoch_set,train_loss_set, label="train")
ax1.plot(epoch_set,val_loss_set, label="val")
plt.xlabel("Epoch")
ax1.legend()

ax2 = plt.subplot(122)
ax2.set_title("AUC")
ax2.plot(epoch_set,train_auc_set, label="train")
ax2.plot(epoch_set,val_auc_set, label="val")
plt.xlabel("Epoch")
ax2.legend()

test_perf, test_loss=eval(test_data.edge_label_index)
print(f"group7, test_loss ={test_loss:.4f}, test_auc ={test_perf:.4f}")

z = model.encode(tg_data.x,edge_index)
final_edge_index = model.decode_all(z)
# print(len(final_edge_index[0]))

plt.savefig('linkp/group7')
plt.show()
