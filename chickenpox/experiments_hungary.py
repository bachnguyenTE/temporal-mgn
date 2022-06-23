import torch
import pickle
import os
import time
import argparse
import numpy as np
from torch import optim
from torch.optim import Adam, Adagrad
from torch.utils.data import DataLoader

from temporal_model import TemporalMGN


###############################
# HUNGARY CHICKENPOX DATASETS #
###############################

# Fix all random seed
torch_geometric.seed.seed_everything(8)

# Set device to gpu
device = torch.device('cuda')

from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

loader = ChickenpoxDatasetLoader()

dataset = loader.get_dataset(lags=8)

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)


#############################
# INITIALIZE TEMPORAL MODEL #
#############################

model = TemporalMGN(
    clusters=[8],
    num_layers=4,
    node_dim=8,
    edge_dim=1,
    hidden_dim=24,
    z_dim=24,
    num_classes=20
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


##################
# TRAINING MODEL #
##################

def train():
    model.train()

    cost = 0
    graph_seq_embed_list = []
    for time, snapshot in enumerate(train_dataset):
        snapshot.to(device)
        if time == 0:
            graph_seq_embed = torch.empty((0, 0)).to(device)
        else: 
            graph_seq_embed = torch.cat(graph_seq_embed_list, dim=0).to(device)
        y_hat, cur_graph_seq_embed = model(time, graph_seq_embed, snapshot.edge_index, snapshot.x, snapshot.edge_attr[:, None])
        # print(f"y_hat size: {y_hat.size()}")
        # print(f"y_hat: {y_hat}")
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
        graph_seq_embed_list.append(cur_graph_seq_embed)
    cost = cost / (time+1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()

def test(test_loader):
    model.eval()

    cost = 0
    graph_seq_embed_list = []
    for time, snapshot in enumerate(test_loader):
        snapshot.to(device)
        if time == 0:
            graph_seq_embed = torch.empty((0, 0)).to(device)
        else: 
            graph_seq_embed = torch.cat(graph_seq_embed_list, dim=0).to(device)
        y_hat, cur_graph_seq_embed = model(time, graph_seq_embed, snapshot.edge_index, snapshot.x, snapshot.edge_attr[:, None])
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
        graph_seq_embed_list.append(cur_graph_seq_embed)
    cost = cost / (time+1)
    cost = cost.item()

    return cost

best_test_loss = float('inf')
for epoch in range(150):
    train()
    train_loss = test(train_dataset)
    test_loss = test(test_dataset)
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), "best_t_mgn_test_chickenpox.pth")
        print(f'Saving test loss new best: {test_loss:.4f}')

    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')


####################
# QUICK EVALUATION #
####################

model.load_state_dict(torch.load("best_t_mgn_test_chickenpox.pth"))
model.eval()
cost = 0
graph_seq_embed_list = []
for time, snapshot in enumerate(test_dataset):
    snapshot.to(device)
    if time == 0:
        graph_seq_embed = torch.empty((0, 0)).to(device)
    else: 
        graph_seq_embed = torch.cat(graph_seq_embed_list, dim=0).to(device)
    y_hat, cur_graph_seq_embed = model(time, graph_seq_embed, snapshot.edge_index, snapshot.x, snapshot.edge_attr[:, None])
    # print(f"y_hat size: {y_hat.size()}")
    # print(f"y_hat: {y_hat}")
    cost = cost + torch.mean((y_hat-snapshot.y)**2)
    graph_seq_embed_list.append(cur_graph_seq_embed)
cost = cost / (time)
cost = cost.item()
print("MSE: {:.4f}".format(cost))


################################
# SLOW WINDOW-BASED EVALUATION #
################################


# 1O-window evaluation
device = torch.device('cpu')
time_lags = 8
model.load_state_dict(torch.load("best_t_mgn_test_chickenpox.pth"))
model.to(device)
model.eval()

with torch.no_grad():
    cost = 0
    graph_seq_embed_list = []
    effective_time_count = 0
    for time, snapshot in enumerate(test_dataset):
        snapshot.to(device)
        if time == 0:
            graph_seq_embed = torch.empty((0, 0)).to(device)
        else: 
            graph_seq_embed = torch.cat(graph_seq_embed_list, dim=0).detach().to(device)
        y_hat, cur_graph_seq_embed = model(time, graph_seq_embed, snapshot.edge_index, snapshot.x, snapshot.edge_attr[:, None])
        # print(f"y_hat size: {y_hat.size()}")
        # print(f"y_hat: {y_hat}")
        base_cost = torch.mean((y_hat-snapshot.y)**2)
        graph_seq_embed_list.append(cur_graph_seq_embed)

        # Predict the remaining 39 time steps
        # Assumption: shape of snapshot.x is (nodes, time_lags), shape of y_hat is (1, nodes); last 8 steps are accumulated accordingly
        sub_time_count = 1
        sub_graph_seq_embed_list = graph_seq_embed_list
        sub_cost = base_cost.item()
        sub_x_feat = torch.cat((snapshot.x[:, :(time_lags-1)], y_hat[:, None]), dim=1).detach().to(device)
        for sub_time, sub_snapshot in enumerate(test_dataset):
            if sub_time <= time:
                continue
            if (sub_time - time) > 9:
                break
            sub_time_count += 1
            sub_snapshot.to(device)
            sub_graph_seq_embed = torch.cat(sub_graph_seq_embed_list, dim=0).detach().to(device)
            sub_y_hat, sub_cur_graph_seq_embed = model(sub_time, sub_graph_seq_embed, sub_snapshot.edge_index, sub_x_feat, sub_snapshot.edge_attr[:, None])
            sub_cost = sub_cost + torch.mean((sub_y_hat-sub_snapshot.y)**2).item()
            sub_graph_seq_embed_list.append(sub_cur_graph_seq_embed)
            sub_x_feat = torch.cat((sub_x_feat[:, :(time_lags-1)], sub_y_hat[:, None]), dim=1).detach().to(device)
            
        if sub_time_count == 10:
            cost += sub_cost / 10
            effective_time_count += 1
        else:
            break

    cost = cost / effective_time_count

print("MSE (10-window): {:.4f}".format(cost))


# 20-window evaluation
device = torch.device('cpu')
time_lags = 8
model.to(device)
model.eval()

with torch.no_grad():
    cost = 0
    graph_seq_embed_list = []
    effective_time_count = 0
    for time, snapshot in enumerate(test_dataset):
        snapshot.to(device)
        if time == 0:
            graph_seq_embed = torch.empty((0, 0)).to(device)
        else: 
            graph_seq_embed = torch.cat(graph_seq_embed_list, dim=0).detach().to(device)
        y_hat, cur_graph_seq_embed = model(time, graph_seq_embed, snapshot.edge_index, snapshot.x, snapshot.edge_attr[:, None])
        # print(f"y_hat size: {y_hat.size()}")
        # print(f"y_hat: {y_hat}")
        base_cost = torch.mean((y_hat-snapshot.y)**2)
        graph_seq_embed_list.append(cur_graph_seq_embed)

        # Predict the remaining 39 time steps
        # Assumption: shape of snapshot.x is (nodes, time_lags), shape of y_hat is (1, nodes); last 8 steps are accumulated accordingly
        sub_time_count = 1
        sub_graph_seq_embed_list = graph_seq_embed_list
        sub_cost = base_cost.item()
        sub_x_feat = torch.cat((snapshot.x[:, :(time_lags-1)], y_hat[:, None]), dim=1).detach().to(device)
        for sub_time, sub_snapshot in enumerate(test_dataset):
            if sub_time <= time:
                continue
            if (sub_time - time) > 19:
                break
            sub_time_count += 1
            sub_snapshot.to(device)
            sub_graph_seq_embed = torch.cat(sub_graph_seq_embed_list, dim=0).detach().to(device)
            sub_y_hat, sub_cur_graph_seq_embed = model(sub_time, sub_graph_seq_embed, sub_snapshot.edge_index, sub_x_feat, sub_snapshot.edge_attr[:, None])
            sub_cost = sub_cost + torch.mean((sub_y_hat-sub_snapshot.y)**2).item()
            sub_graph_seq_embed_list.append(sub_cur_graph_seq_embed)
            sub_x_feat = torch.cat((sub_x_feat[:, :(time_lags-1)], sub_y_hat[:, None]), dim=1).detach().to(device)
            
        if sub_time_count == 20:
            cost += sub_cost / 20
            effective_time_count += 1
        else:
            break

    cost = cost / effective_time_count

print("MSE (20-window): {:.4f}".format(cost))


# 40-window evaluation
device = torch.device('cpu')
time_lags = 8
model.to(device)
model.eval()

with torch.no_grad():
    cost = 0
    graph_seq_embed_list = []
    effective_time_count = 0
    for time, snapshot in enumerate(test_dataset):
        snapshot.to(device)
        if time == 0:
            graph_seq_embed = torch.empty((0, 0)).to(device)
        else: 
            graph_seq_embed = torch.cat(graph_seq_embed_list, dim=0).detach().to(device)
        y_hat, cur_graph_seq_embed = model(time, graph_seq_embed, snapshot.edge_index, snapshot.x, snapshot.edge_attr[:, None])
        # print(f"y_hat size: {y_hat.size()}")
        # print(f"y_hat: {y_hat}")
        base_cost = torch.mean((y_hat-snapshot.y)**2)
        graph_seq_embed_list.append(cur_graph_seq_embed)

        # Predict the remaining 39 time steps
        # Assumption: shape of snapshot.x is (nodes, time_lags), shape of y_hat is (1, nodes); last 8 steps are accumulated accordingly
        sub_time_count = 1
        sub_graph_seq_embed_list = graph_seq_embed_list
        sub_cost = base_cost.item()
        sub_x_feat = torch.cat((snapshot.x[:, :(time_lags-1)], y_hat[:, None]), dim=1).detach().to(device)
        for sub_time, sub_snapshot in enumerate(test_dataset):
            if sub_time <= time:
                continue
            if (sub_time - time) > 39:
                break
            sub_time_count += 1
            sub_snapshot.to(device)
            sub_graph_seq_embed = torch.cat(sub_graph_seq_embed_list, dim=0).detach().to(device)
            sub_y_hat, sub_cur_graph_seq_embed = model(sub_time, sub_graph_seq_embed, sub_snapshot.edge_index, sub_x_feat, sub_snapshot.edge_attr[:, None])
            sub_cost = sub_cost + torch.mean((sub_y_hat-sub_snapshot.y)**2).item()
            sub_graph_seq_embed_list.append(sub_cur_graph_seq_embed)
            sub_x_feat = torch.cat((sub_x_feat[:, :(time_lags-1)], sub_y_hat[:, None]), dim=1).detach().to(device)
            
        if sub_time_count == 40:
            cost += sub_cost / 40
            effective_time_count += 1
        else:
            break

    cost = cost / effective_time_count

print("MSE (40-window): {:.4f}".format(cost))