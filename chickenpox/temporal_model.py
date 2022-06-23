# Library import (legacy MGVAE)
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import os
import time
import argparse
from torch.nn import MultiheadAttention

# Library import (pytorch-geometric)
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, ClusterLoader, ClusterData
from torch_geometric.nn import MessagePassing, GCNConv, GATConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse

###############################################################
# NOTE: 
# We preferably define our own clustering 
# procedure, rather than using the built-in ClusterLoader
# since there is a chance using ClusterLoader will not
# make the entire net differentiable (separate data process),
# and the net may no longer be isomorphic invariant.
###############################################################

# Fix all random seed
torch_geometric.seed.seed_everything(8)

# Set device to gpu
device = torch.device('cuda')


#######################
# MODEL SUBCOMPONENTS #
#######################


# Define glorot initialization
def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


# Graph encoder block
class GraphEncoder(nn.Module):
    def __init__(self, num_layers, node_dim, edge_dim, hidden_dim, z_dim, use_concat_layer=True, **kwargs):
        super(GraphEncoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.use_concat_layer = use_concat_layer

        self.node_fc1 = nn.Linear(self.node_dim, 64)
        self.node_fc2 = nn.Linear(64, self.hidden_dim)

        if self.edge_dim is not None:
            self.edge_fc1 = nn.Linear(self.edge_dim, 64)
            self.edge_fc2 = nn.Linear(64, self.hidden_dim)

        self.base_net = nn.ModuleList()
        self.combine_net = nn.ModuleList()
        for layer in range(self.num_layers):
            self.base_net.append(GATConv(self.hidden_dim, self.hidden_dim, edge_dim=self.hidden_dim))

        if self.use_concat_layer == True:
            self.latent_fc1 = nn.Linear((self.num_layers + 1) * self.hidden_dim, 128)
            self.latent_fc2 = nn.Linear(128, self.z_dim)
        else:
            self.latent_fc1 = nn.Linear(self.hidden_dim, 128)
            self.latent_fc2 = nn.Linear(128, self.z_dim)

    def forward(self, adj, node_feat, edge_feat=None):
        node_hidden = torch.tanh(self.node_fc1(node_feat))
        node_hidden = torch.tanh(self.node_fc2(node_hidden))

        if edge_feat is not None and self.edge_dim is not None:
            edge_hidden = torch.tanh(self.edge_fc1(edge_feat))
            edge_hidden = torch.tanh(self.edge_fc2(edge_hidden))
        else:
            edge_hidden = None

        all_hidden = [node_hidden]

        if edge_feat is not None:
            for layer in range(len(self.base_net)):
                if layer == 0:
                    hidden = self.base_net[layer](node_hidden, adj, edge_hidden)
                else:
                    hidden = self.base_net[layer](hidden, adj, edge_hidden)
            
                all_hidden.append(hidden)
        else:
            for layer in range(len(self.base_net)):
                if layer == 0:
                    hidden = self.base_net[layer](node_hidden, adj)
                else:
                    hidden = self.base_net[layer](hidden, adj)
            
                all_hidden.append(hidden)

        if self.use_concat_layer == True:
            hidden = torch.cat(all_hidden, dim=1)

        latent = torch.tanh(self.latent_fc1(hidden))
        latent = torch.tanh(self.latent_fc2(latent))
        return latent, edge_hidden


# Graph clustering block
class GraphCluster(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, z_dim, **kwargs):
        super(GraphCluster, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, self.hidden_dim)

        # Option 1: Learnable clustering
        self.base_net = nn.ModuleList()
        
        # Option 2: Fixed clustering
        # self.base_net = []

        for layer in range(self.num_layers):
            self.base_net.append(GCNConv(self.hidden_dim, self.hidden_dim))

        self.assign_net = GCNConv(self.hidden_dim, self.z_dim)

    def forward(self, adj, X):
        hidden = torch.sigmoid(self.fc1(X))
        hidden = torch.sigmoid(self.fc2(hidden))
        for net in self.base_net:
            hidden = net(hidden, adj)
        assign = self.assign_net(hidden, adj)
        return assign


# Multiresolution Graph Network
class MGN(nn.Module):
    def __init__(self, clusters, num_layers, node_dim, edge_dim, hidden_dim, z_dim, num_classes):
        super(MGN, self).__init__()
        self.clusters = clusters
        self.num_layers = num_layers
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.num_classes = num_classes

        self.base_encoder = GraphEncoder(self.num_layers, self.node_dim, self.edge_dim, self.hidden_dim, self.z_dim)

        self.cluster_learner = nn.ModuleList()
        self.global_encoder = nn.ModuleList()
        for i in range(len(self.clusters)):
            N = self.clusters[i]
            self.cluster_learner.append(GraphCluster(self.num_layers, self.z_dim, self.hidden_dim, N))
            if edge_dim is not None:
                self.global_encoder.append(GraphEncoder(self.num_layers, self.z_dim, self.hidden_dim, self.hidden_dim, self.z_dim))
            else:
                self.global_encoder.append(GraphEncoder(self.num_layers, self.z_dim, None, self.hidden_dim, self.z_dim))

        self.vertical_attention = MultiheadAttention(self.z_dim, 1, batch_first=True)

        D = self.z_dim * (len(self.clusters) + 1)
        self.fc1 = nn.Linear(self.z_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, adj, node_feat, edge_feat = None):
        outputs = []

        # Base encoder
        base_latent, base_edge_hidden = self.base_encoder(adj, node_feat, edge_feat)

        outputs.append([base_latent, adj, base_edge_hidden])

        l = 0
        while l < len(self.clusters):
            if l == 0:
                prev_adj = adj
                prev_latent = base_latent
                prev_edge_hidden = base_edge_hidden
            else:
                prev_adj = outputs[len(outputs) - 1][1]
                prev_latent = outputs[len(outputs) - 1][0]
                prev_edge_hidden = outputs[len(outputs) - 1][2]

            # Assignment score
            assign_score = self.cluster_learner[l](prev_adj, prev_latent)

            # Softmax (soft assignment)
            # assign_matrix = F.softmax(assign_score, dim = 2)

            # Gumbel softmax (hard assignment)
            assign_matrix = F.gumbel_softmax(assign_score, tau = 1, hard = True, dim = 1)

            # Print out the cluster assignment matrix
            # print(torch.sum(assign_matrix, dim = 0))

            # Shrinked latent
            shrinked_latent = torch.matmul(assign_matrix.transpose(0, 1), prev_latent)

            # Latent normalization
            shrinked_latent = F.normalize(shrinked_latent, dim = 0)

            # Shrinked adjacency
            if l == 0:
                shrinked_adj = torch.einsum('sdf,st->tdf', torch.matmul(assign_matrix.transpose(0, 1), to_dense_adj(prev_adj, edge_attr = prev_edge_hidden, max_num_nodes=node_feat.size()[0])[0]), assign_matrix)
            else:
                shrinked_adj = torch.einsum('sdf,st->tdf', torch.matmul(assign_matrix.transpose(0, 1), to_dense_adj(prev_adj, edge_attr = prev_edge_hidden, max_num_nodes=self.clusters[l - 1])[0]), assign_matrix)

            # Adjacency normalization
            shrinked_adj = shrinked_adj / torch.sum(shrinked_adj)

            # Reformatting adjacency matrix as edge index
            shrinked_edge_indices = torch.stack(torch.sum(shrinked_adj, dim=2).nonzero(as_tuple=True), dim=0)
            shrinked_edge_hidden = shrinked_adj[shrinked_edge_indices[0], shrinked_edge_indices[1], :]

            # Global encoder
            next_latent, next_edge_hidden = self.global_encoder[l](shrinked_edge_indices, shrinked_latent, shrinked_edge_hidden)

            outputs.append([next_latent, shrinked_edge_indices, next_edge_hidden])
            l += 1

        # Scalar prediction with vertical attention
        latents = torch.stack([torch.sum(output[0], dim = 0) for output in outputs], dim = 0)
        latents = latents[None, :]
        att_latents, _ = self.vertical_attention(latents, latents, latents)
        output_latents = torch.mean(att_latents[0], dim=0)
        hidden = torch.tanh(self.fc1(output_latents))
        predict = self.fc2(hidden)

        return predict, output_latents, outputs


######################################################
# MAIN MODEL: TEMPORAL MULTIRESOLUTION GRAPH NETWORK #
######################################################


class TemporalMGN(nn.Module):
    def __init__(self, clusters, num_layers, node_dim, edge_dim, hidden_dim, z_dim, num_classes):
        super(TemporalMGN, self).__init__()
        self.clusters = clusters
        self.num_layers = num_layers
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.num_classes = num_classes

        self.attention_multires = MGN(self.clusters, self.num_layers, self.node_dim, self.edge_dim, self.hidden_dim, self.z_dim, self.num_classes)
        self.horizontal_attention = MultiheadAttention(self.z_dim, 1, batch_first=True)

        self.fc1 = nn.Linear(self.z_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, time_step: int, graph_seq_embed, adj, node_feat, edge_feat=None):
        _, multires_latents, _ = self.attention_multires(adj, node_feat, edge_feat)
        if time_step is not 0:
            graph_seq_embed = torch.cat([graph_seq_embed, multires_latents[None, :]], dim=0)
        else:
            graph_seq_embed = multires_latents[None, :]

        attn_embed = graph_seq_embed[None, :].to(device)
        horizontal_attn_latents, _ = self.horizontal_attention(attn_embed, attn_embed, attn_embed)
        output_latents = torch.mean(horizontal_attn_latents[0], dim=0)
        hidden = torch.tanh(self.fc1(output_latents))
        predict = self.fc2(hidden)

        return predict, multires_latents[None, :]