import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from functools import partial
from networks.value_win_network import ValueWinNetwork

from sevn import State


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, with_bias=True, activation=None, on_cuda=False):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels + 1  # add one for the identity relation
        self.with_bias = with_bias
        self.activation = activation

        # init weights
        self.weight = nn.Parameter(torch.empty(
            self.num_rels,
            self.in_feat,
            self.out_feat,
            device=torch.device('cuda' if on_cuda else 'cpu')
        ))

        nn.init.xavier_uniform_(
            self.weight, gain=nn.init.calculate_gain('relu'))

        # init bias
        if self.with_bias:
            self.bias = nn.Parameter(torch.empty(
                self.num_rels,
                out_feat,
                device=torch.device('cuda' if on_cuda else 'cpu')
            ))

            nn.init.xavier_uniform_(
                self.bias, gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        def message_func(edges):
            """Transforms data along edges"""

            w = self.weight[edges.data['rel_type']]

            msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze(1)

            if self.with_bias:
                msg += self.bias[edges.data['rel_type']]

            # msg = msg * edges.data['norm']
            return {'msg': msg}

        def apply_func(nodes):
            """Transforms nodes' own data and combines with aggregated data from neighbours"""

            num_nodes = len(nodes.data['h'])
            w = self.weight[-torch.ones(num_nodes, dtype=torch.int64)]

            id_msg = torch.bmm(nodes.data['h'].unsqueeze(1), w).squeeze(1)
            h = id_msg + nodes.data.get('agg', 0)

            if self.with_bias:
                h += self.bias[-torch.ones(num_nodes, dtype=torch.int64)]

            if self.activation:
                h = self.activation(h)
            # print("h:", h.requires_grad)

            return {'h': h}

        g.update_all(message_func, fn.max(msg='msg', out='agg'), apply_func)


class DGLValueWinNetwork(ValueWinNetwork, nn.Module):
    def __init__(self, dims=[3, 64, 64, 32, 32, 16, 8, 2], num_rels=5, on_cuda=False):
        super().__init__()
        self.dims = dims
        self.num_rels = num_rels
        self.on_cuda = on_cuda

        self.layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            activation_fn = torch.tanh if i+2 == len(dims) else F.relu
            self.layers.append(
                RGCNLayer(dims[i], dims[i+1], num_rels,
                          activation=activation_fn, on_cuda=on_cuda)
            )

    def forward(self, g):
        if self.on_cuda:
            g = g.to('cuda:0')

        g.ndata['h'] = g.ndata['features']
        g.ndata['h'].requires_grad = True

        if self.on_cuda:
            torch.device('cuda')
            with torch.cuda.device(0):

                for layer in self.layers:
                    layer(g)

                return torch.mean(g.ndata.pop('h'), 0).to(device=torch.device('cpu'))

        else:
            for layer in self.layers:
                layer(g)

            return torch.mean(g.ndata.pop('h'), 0)

    def evaluate(self, state):
        g = state.to_dgl_graph()
        return self.forward(g)
