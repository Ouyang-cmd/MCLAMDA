import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import KFold
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
from Model import MCLAMDA, FeatureExtractor, InteractionPredictor
from DataHandler import build_network, extract_samples, fetch_data
from Params import args
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def setup_data_and_model(seed, data_dir):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    network, mappings = build_network(data_dir, seed)
    interactions = extract_samples(data_dir, seed)
    Density, _ = fetch_data(data_dir)
    interactions_df = pd.DataFrame(interactions, columns=['miRNA', 'disease', 'label'])

    disease_map, mirna_map = mappings
    disease_vertices = [disease_map[id_] for id_ in interactions[:, 1]]
    mirna_vertices = [mirna_map[id_] + Density.shape[0] for id_ in interactions[:, 0]]

    return network, interactions_df, disease_vertices, mirna_vertices


def train_and_evaluate(network, interactions_df, disease_vertices, mirna_vertices, computing_device, total_epochs, emb_size, num_layers, dropout_rate, activation_slope, learning_rate, weight_decay):
    splitter = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    for i, (train_indices, test_indices) in enumerate(splitter.split(interactions_df)):
        interactions_df['is_train'] = 0
        interactions_df['is_test'] = 0
        interactions_df.loc[train_indices, 'is_train'] = 1
        interactions_df.loc[test_indices, 'is_test'] = 1

        train_flag = torch.from_numpy(interactions_df['is_train'].values.astype('int32')).to(computing_device)
        test_flag = torch.from_numpy(interactions_df['is_test'].values.astype('int32')).to(computing_device)
        edge_info = {'train': train_flag, 'test': test_flag}
        network.edges[disease_vertices, mirna_vertices].data.update(edge_info)
        network.edges[mirna_vertices, disease_vertices].data.update(edge_info)

        train_edges = network.filter_edges(lambda edges: edges.data['train'].bool()).to(computing_device)
        network_train = network.edge_subgraph(train_edges, preserve_nodes=True)
        network_train.copy_from_parent()
        src_train, dst_train = network_train.all_edges()
        ratings_train = network_train.edata['weight']

        test_edges = network.filter_edges(lambda edges: edges.data['test'].bool()).to(computing_device)
        src_test, dst_test = network.find_edges(test_edges)
        ratings_test = network.edges[test_edges].data['weight']

        model = MCLAMDA(FeatureExtractor(emb_size, num_layers, network_train, dropout_rate, activation_slope, computing_device),
                        InteractionPredictor(emb_size, computing_device), computing_device).to(computing_device)
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_fn = nn.BCELoss()

        fold_epoch_data = []

        for epoch in range(total_epochs):
            model.train()
            optim.zero_grad()
            output, a, b = model(network_train, src_train, dst_train)
            loss = loss_fn(output, ratings_train.float()) + a + b
            loss.backward()
            optim.step()

            model.eval()
            with torch.no_grad():
                test_output, _, _ = model(network, src_test, dst_test)
                test_output_np = test_output.detach().cpu().numpy()
                ratings_test_np = ratings_test.cpu().numpy()
                current_test_auc = metrics.roc_auc_score(ratings_test_np, test_output_np)
                current_test_aupr = metrics.average_precision_score(ratings_test_np, test_output_np)

                fold_epoch_data.append({
                    'epoch': epoch,
                    'auc': current_test_auc,
                    'aupr': current_test_aupr
                })

                print(f'Fold {i + 1}, Epoch {epoch}: AUC = {current_test_auc:.4f}, AUPR = {current_test_aupr:.4f}')


network, interactions_df, disease_vertices, mirna_vertices = setup_data_and_model(args.seed, args.data_dir)
train_and_evaluate(network, interactions_df, disease_vertices, mirna_vertices, torch.device(args.device), args.total_epochs, args.emb_size, args.num_layers, args.dropout_rate, args.activation_slope, args.learning_rate, args.weight_decay)
