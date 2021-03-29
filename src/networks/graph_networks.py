import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from functools import partial
from game.sevn import State
from networks.network import Network, PolicyValueNetwork, ValueWinNetwork


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, with_bias=True, activation=F.relu, on_cuda=False, first_act_softmax=False):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels + 1  # add one for the identity relation
        self.with_bias = with_bias
        self.activation = activation
        self.first_act_softmax = first_act_softmax

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

            return {'msg': msg}

        def apply_func(nodes):
            """Transforms nodes' own data and combines with aggregated data from neighbours"""

            num_nodes = len(nodes.data['h'])
            w = self.weight[-torch.ones(num_nodes, dtype=torch.int64)]

            id_msg = torch.bmm(nodes.data['h'].unsqueeze(1), w).squeeze(1)
            h = id_msg + nodes.data.get('agg', 0)

            if self.with_bias:
                h += self.bias[-torch.ones(num_nodes, dtype=torch.int64)]

            if self.first_act_softmax:
                col0 = nn.functional.softmax(h[:, 0], dim=0)
                rest = self.activation(h[:, 1:])
                h = torch.cat((col0.unsqueeze(1), rest), dim=1)
            else:
                h = self.activation(h)

            return {'h': h}

        g.update_all(message_func, fn.max(msg='msg', out='agg'), apply_func)


class RGCNet(Network):
    def __init__(self, dims=[3, 64, 64, 32, 32, 16, 8, 2], num_rels=5, on_cuda=False, first_act_softmax=False):
        super().__init__()
        self.dims = dims
        self.num_rels = num_rels
        self.on_cuda = on_cuda

        self.layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            activation_fn = torch.tanh if i+2 == len(dims) else F.relu
            self.layers.append(
                RGCNLayer(dims[i], dims[i+1], num_rels,
                          activation=activation_fn, on_cuda=on_cuda, first_act_softmax=first_act_softmax)
            )

    def forward(self, g):
        if self.on_cuda:
            g = g.to('cuda:0')

        g.ndata['h'] = g.ndata['features']
        g.ndata['h'].requires_grad = True

        def transform():
            for layer in self.layers:
                layer(g)

            return self.aggregate(g)

        if self.on_cuda:
            torch.device('cuda')
            with torch.cuda.device(0):

                return transform().to(device=torch.device('cpu'))

        return transform()

    def aggregate(self, g):
        raise NotImplemented

    def evaluate(self, state):
        g = state.to_dgl_graph()
        return self.forward(g).detach()


class DGLValueWinNetwork(RGCNet, ValueWinNetwork):

    def aggregate(self, g):
        return torch.mean(g.ndata.pop('h'), 0)


class PolicyValRGCN(RGCNet, PolicyValueNetwork):

    def __init__(self, dims=[3, 64, 64, 32, 32, 16, 8, 2], num_rels=7, on_cuda=False):
        super().__init__(dims, num_rels, on_cuda, first_act_softmax=True)

    def aggregate(self, g):
        h_data = g.ndata.pop('h')

        return torch.cat((h_data[:, 0], torch.mean(h_data[:, 1]).unsqueeze(0)))

    def evaluate(self, state):
        g = state.to_dgl_graph(with_move_nodes=True)
        result = self.forward(g).detach()
        # print(result.sort()[0].tolist())

        policy = result[:len(state.get_moves())]
        value = result[-1]
        # value assumes first value is the average value, as done in aggregate()

        return (
            policy/torch.sum(policy),
            value
        )
