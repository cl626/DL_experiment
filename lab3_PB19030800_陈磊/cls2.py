import pdb
from typing import Any, Callable, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import Tensor
from torch.optim import Optimizer
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
# from torch_geometric.utils import accuracy
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

# download data and display relative information

dataset = Planetoid("./tmp/Cora", name="Cora", transform=T.NormalizeFeatures())
num_nodes = dataset.data.num_nodes
# For num. edges see:
# - https://github.com/pyg-team/pytorch_geometric/issues/343
# - https://github.com/pyg-team/pytorch_geometric/issues/852

num_edges = dataset.data.num_edges // 2
train_len = dataset[0].train_mask.sum()
val_len = dataset[0].val_mask.sum()
test_len = dataset[0].test_mask.sum()
other_len = num_nodes - train_len - val_len - test_len
print(f"Dataset: {dataset.name}")
print(f"Num. nodes: {num_nodes} (train={train_len}, val={val_len}, test={test_len}, other={other_len})")
print(f"Num. edges: {num_edges}")
print(f"Num. node features: {dataset.num_node_features}")
print(f"Num. classes: {dataset.num_classes}")
print(f"Dataset len.: {dataset.len()}")

from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
# customerized GCNConv
    
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

class myGCNConv1(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 add_self_loops: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.kernel = Linear(in_channels,out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.kernel.reset_parameters()         #卷积层

    def forward(self,x,edge_index):
        col=len(x)
        A=torch.zeros(col,col).float()
        D=torch.zeros(col,col).float()        
        for i in range(len(edge_index[0])):
            A[edge_index[0][i]][edge_index[1][i]]=1
        for i in range(col):
            D[i][i]=A[i].sum()
        if(self.add_self_loops):
            A=A+torch.eye(col,col)
            D=D+torch.eye(col,col)
        D_inv=D.pow(-0.5)
        D_inv.masked_fill(D_inv == float('inf'),0)
        out=torch.chain_matmul(D_inv,D,D_inv,self.kernel(x))
        
        return out

class myGCNConv2(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 add_self_loops: bool = True,bias: bool = True):
        super().__init__()

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


# define our graphic convolutional network
class GCN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        hidden_dim: int = 16,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.conv1 = myGCNConv2(num_node_features, hidden_dim, add_self_loops=True)
        self.relu = torch.nn.ReLU(inplace=True)
        self.lrelu = torch.nn.LeakyReLU()
        self.sigm = torch.nn.Sigmoid()
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        self.conv2 = myGCNConv2(hidden_dim, num_classes,add_self_loops=True)
        self.pn=PairNorm()

    def forward(self, x: Tensor, edge_index: Tensor) -> torch.Tensor:
        x = self.pn(x)
        x = self.dropout1(x)
        x = self.conv1(x, edge_index)
        x = self.lrelu(x)
        x = self.dropout2(x)
        x = self.conv2(x, edge_index)
        return x
    
print("Graph Convolutional Network (GCN):")
print(GCN(dataset.num_node_features, dataset.num_classes))

#single_edge and drop_edge
dp=0.1

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

#single_edge_index
single_edge={}
for i in range(len(dataset.data.edge_index[0])):
    if(((dataset.data.edge_index[0][i],dataset.data.edge_index[1][i]) not in single_edge.items()) and 
        ((dataset.data.edge_index[1][i],dataset.data.edge_index[0][i]) not in single_edge.items())):
        single_edge[dataset.data.edge_index[0][i]]=dataset.data.edge_index[1][i]

single_edge_index=[[],[]]

for key,value in single_edge.items():
    single_edge_index[0].append(key)
    single_edge_index[1].append(value)        

single_edge_index=torch.tensor(single_edge_index)

#other handling
LossFn = Callable[[Tensor, Tensor], Tensor]
Stage = ["train", "val", "test"]

def accuracy(pred: Tensor, target: Tensor) -> float:
    """
    计算分类任务的准确度。
    Args:
        pred (torch.Tensor): 模型的预测输出，形状为 [num_examples, num_classes]。
        target (torch.Tensor): 样本的真实标签，形状为 [num_examples]。
    Returns:
        float: 分类任务的准确度。
    """
    # 将预测输出转换为标签
    correct = float(pred.eq(target).sum().item())
    acc = correct / len(target)
    return acc


def train_step(
    model: torch.nn.Module, data: Data, optimizer: torch.optim.Optimizer, loss_fn: LossFn
) -> Tuple[float, float]:
    model.train()
    optimizer.zero_grad()
    mask = data.train_mask
    logits = model(data.x, drop_edge(single_edge_index,dp))[mask]
    preds = logits.argmax(dim=1)
    y = data.y[mask]
    loss = loss_fn(logits, y)
    # + L2 regularization to the first layer only
    # for name, params in model.state_dict().items():
    #     if name.startswith("conv1"):
    #         loss += 5e-4 * params.square().sum() / 2.0

    acc = accuracy(preds, y)

    loss.backward()
    optimizer.step()
    return loss.item(), acc


@torch.no_grad()
def eval_step(model: torch.nn.Module, data: Data, loss_fn: LossFn, stage: Stage) -> Tuple[float, float]:
    model.eval()
    mask = getattr(data, f"{stage}_mask")
    logits = model(data.x, drop_edge(single_edge_index,dp))[mask]
    preds = logits.argmax(dim=1)
    y = data.y[mask]
    loss = loss_fn(logits, y)
    # + L2 regularization to the first layer only
    # for name, params in model.state_dict().items():
    #     if name.startswith("conv1"):
    #         loss += 5e-4 * params.square().sum() / 2.0
    #print(preds)

    acc = accuracy(preds, y)
    return loss.item(), acc

# define a class to describe the loss and acc of training
class HistoryDict(TypedDict):
    loss: List[float]
    acc: List[float]
    val_loss: List[float]
    val_acc: List[float]


def train(
    model: torch.nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn = torch.nn.CrossEntropyLoss(),
    max_epochs: int = 200,
    # early_stopping: int = 10,
    # print_interval: int = 20,
    # verbose: bool = True,
) -> HistoryDict:
    history = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
    for epoch in range(max_epochs):
        if epoch%5==0 and epoch>=70:
            optimizer.param_groups[0]['lr']*=0.8
        loss, acc = train_step(model, data, optimizer, loss_fn)
        val_loss, val_acc = eval_step(model, data, loss_fn, "val")
        history["loss"].append(loss)
        history["acc"].append(acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        # The official implementation in TensorFlow is a little different from what is described in the paper...
        # if epoch > early_stopping and val_loss > np.mean(history["val_loss"][-(early_stopping + 1) : -1]):
        #     if verbose:
        #         print("\nEarly stopping...")

        #     break

        # if verbose and epoch % print_interval == 0:
        print(f"\nEpoch: {epoch} , Train loss: {loss:.4f} | Train acc: {acc:.4f} | Val loss: {val_loss:.4f} |   Val acc: {val_acc:.4f}")

    test_loss, test_acc = eval_step(model, data, loss_fn, "test")
    # if verbose:
    print(f"group7_lrelu Test loss: {test_loss:.4f} |  Test acc: {test_acc:.4f}")

    return history

def plot_history(history: HistoryDict, title: str, font_size: Optional[int] = 14) -> None:
    plt.suptitle(title, fontsize=font_size)
    ax1 = plt.subplot(121)
    ax1.set_title("Loss")
    ax1.plot(history["loss"], label="train")
    ax1.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    ax1.legend()

    ax2 = plt.subplot(122)
    ax2.set_title("Accuracy")
    ax2.plot(history["acc"], label="train")
    ax2.plot(history["val_acc"], label="val")
    plt.xlabel("Epoch")
    ax2.legend()
    plt.savefig('cls/group7')

SEED = 42
MAX_EPOCHS = 200
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EARLY_STOPPING = 10


torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GCN(dataset.num_node_features, dataset.num_classes).to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
history = train(model, data, optimizer, max_epochs=MAX_EPOCHS, )

plot_history(history,"node classification")
plt.show()