import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from functools import partial

from sevn import State

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))

        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases (equation (3))
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                        self.in_feat, self.out_feat)
        else:
            weight = self.weight

        # if self.is_input_layer:
        #     def message_func(edges):
        #         # for input layer, matrix multiply can be converted to be
        #         # an embedding lookup using source node id
        #         embed = weight.view(-1, self.out_feat)
        #         index = edges.data['rel_type'] * self.in_feat + edges.src['id']
        #         return {'msg': embed[index] * edges.data['norm']}
        # else:
        def message_func(edges):
            w = weight[edges.data['rel_type']]
            torch.bmm(edges.src['h'].unsqueeze(1), w)
            msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
            # msg = msg * edges.data['norm']
            return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.max(msg='msg', out='h'), apply_func)

class Model(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_rels,
                 num_bases=-1, num_hidden_layers=1):
        super(Model, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers

        # create rgcn layers
        self.build_model()

        # create initial features
        # self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    # initialize feature for each node
    # def create_features(self):
    #     features = torch.arange(self.num_nodes)
    #     return features

    def build_input_layer(self):
        return RGCNLayer(self.in_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True)

    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu)

    def build_output_layer(self):
        return RGCNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases,
                         activation=partial(torch.sigmoid))#, dim=1))

    def forward(self, g):
        g.ndata['h'] = g.ndata['features']
        # g.ndata['id'] = features
        # if self.features is not None:
        #     g.ndata['id'] = self.features
        for layer in self.layers:
            layer(g)
        return g.ndata.pop('h')

g = State.from_str("2/c-eae-b/5.3a1.1ebc1.2d2.2e2").to_dgl_graph()

model = Model(
    in_dim=3,
    h_dim=10,
    out_dim=2,
    num_rels=5,
    num_hidden_layers=1
)

n_hidden = 16 # number of hidden units
n_bases = -1 # use number of relations as number of bases
n_hidden_layers = 0 # use 1 input layer, 1 output layer, no hidden layer
n_epochs = 25 # epochs to train
lr = 0.01 # learning rate
l2norm = 0 # L2 norm coefficient

train_idx = [0,1,2,3,4,5]
val_idx = [0,1,2,3,4,5]

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)

labels = torch.tensor([
    [0.,1.],
    [1.,1.],
    [1.,1.],
    [1.,1.],
    [1.,1.],
    [1.,1.],
],dtype=torch.float)

train_data = [(g, labels)]

print("start training...")
model.train()
for epoch in range(50):
    epoch_loss = 0
    
    for n, (g, labels) in enumerate(train_data):
        optimizer.zero_grad()
        logits = model.forward(g)
        loss_fn = nn.MSELoss()
        loss = loss_fn(logits[train_idx], labels[train_idx])
        loss.backward()

        optimizer.step()
        epoch_loss += loss.detach().item()
    
    epoch_loss /= n+1

    print("Epoch {:05d} | ".format(epoch) +
          "Train Loss: {:.4f} | ".format(epoch_loss))

print(model.forward(g))