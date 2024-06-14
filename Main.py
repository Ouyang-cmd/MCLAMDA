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


def setup_data_and_model(seed, data_dir, test_size=0.2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_val_samples, test_samples = extract_samples(data_dir, seed, test_size)
    network, mappings = build_network(data_dir, train_val_samples, seed)
    Density, _ = fetch_data(data_dir)

    disease_map, mirna_map = mappings
    train_val_df = pd.DataFrame(train_val_samples, columns=['miRNA', 'disease', 'label'])
    test_df = pd.DataFrame(test_samples, columns=['miRNA', 'disease', 'label'])

    disease_vertices = [disease_map[id_] for id_ in train_val_samples[:, 1]]
    mirna_vertices = [mirna_map[id_] + Density.shape[0] for id_ in train_val_samples[:, 0]]

    return network, train_val_df, test_df, disease_vertices, mirna_vertices, mappings, Density.shape[0]


def train_and_evaluate(network, train_val_df, test_df, disease_vertices, mirna_vertices, mappings, density_size,
                       computing_device, total_epochs, emb_size, num_layers, dropout_rate, activation_slope,
                       learning_rate, weight_decay, early_stop_limit):
    results = {
        'fprs': [],
        'tprs': [],
        'auc': [],
        'precisions': [],
        'recalls': [],
        'aupr': [],
        'epoch_details': []
    }

    disease_map, mirna_map = mappings

    splitter = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    for i, (train_indices, val_indices) in enumerate(splitter.split(train_val_df)):
        train_val_df['is_train'] = 0
        train_val_df['is_val'] = 0
        train_val_df.loc[train_indices, 'is_train'] = 1
        train_val_df.loc[val_indices, 'is_val'] = 1

        train_flag = torch.from_numpy(train_val_df['is_train'].values.astype('int32')).to(computing_device)
        val_flag = torch.from_numpy(train_val_df['is_val'].values.astype('int32')).to(computing_device)
        edge_info = {'train': train_flag, 'val': val_flag}
        network.edges[disease_vertices, mirna_vertices].data.update(edge_info)
        network.edges[mirna_vertices, disease_vertices].data.update(edge_info)

        train_edges = network.filter_edges(lambda edges: edges.data['train'].bool()).to(computing_device)
        network_train = network.edge_subgraph(train_edges, preserve_nodes=True)
        network_train.copy_from_parent()
        src_train, dst_train = network_train.all_edges()
        ratings_train = network_train.edata['weight']

        val_edges = network.filter_edges(lambda edges: edges.data['val'].bool()).to(computing_device)
        src_val, dst_val = network.find_edges(val_edges)
        ratings_val = network.edges[val_edges].data['weight']

        model = MCLAMDA(
            FeatureExtractor(emb_size, num_layers, network_train, dropout_rate, activation_slope, computing_device),
            InteractionPredictor(emb_size, computing_device), computing_device).to(computing_device)
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_fn = nn.BCELoss()

        best_auc = 0
        best_val_output = None
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
                val_output, _, _ = model(network, src_val, dst_val)
                val_output_np = val_output.detach().cpu().numpy()
                ratings_val_np = ratings_val.cpu().numpy()
                current_val_auc = metrics.roc_auc_score(ratings_val_np, val_output_np)
                current_val_aupr = metrics.average_precision_score(ratings_val_np, val_output_np)
                predicted_classes = (val_output_np > 0.5).astype(int)
                current_acc = accuracy_score(ratings_val_np, predicted_classes)
                current_prec = precision_score(ratings_val_np, predicted_classes)
                current_rec = recall_score(ratings_val_np, predicted_classes)
                current_f1 = f1_score(ratings_val_np, predicted_classes)

                if current_val_auc > best_auc:
                    best_auc = current_val_auc
                    best_val_output = val_output_np

                fold_epoch_data.append({
                    'epoch': epoch,
                    'loss': loss.item(),
                    'auc': current_val_auc,
                    'aupr': current_val_aupr,
                    'accuracy': current_acc,
                    'precision': current_prec,
                    'recall': current_rec,
                    'f1': current_f1
                })

                print(
                    f'Fold {i + 1}, Epoch {epoch}: Loss = {loss.item()}, AUC = {current_val_auc:.4f}, AUPR = {current_val_aupr:.4f}, Acc = {current_acc:.4f}, Prec = {current_prec:.4f}, Rec = {current_rec:.4f}, F1 = {current_f1:.4f}')

        results['epoch_details'].append(fold_epoch_data)

        fpr, tpr, _ = metrics.roc_curve(ratings_val_np, best_val_output)
        precision, recall, _ = metrics.precision_recall_curve(ratings_val_np, best_val_output)
        results['fprs'].append(fpr)
        results['tprs'].append(tpr)
        results['auc'].append(best_auc)
        results['precisions'].append(precision)
        results['recalls'].append(recall)
        results['aupr'].append(metrics.average_precision_score(ratings_val_np, best_val_output))

        model_path = os.path.join(args.data_dir, f'model_fold_{i + 1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f'Model saved for Fold {i + 1} at {model_path}')

    test_disease_vertices = [disease_map[id_] for id_ in test_df['disease']]
    test_mirna_vertices = [mirna_map[id_] + density_size for id_ in test_df['miRNA']]
    src_test = torch.tensor(test_disease_vertices, dtype=torch.int64)
    dst_test = torch.tensor(test_mirna_vertices, dtype=torch.int64)
    ratings_test = torch.tensor(test_df['label'], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        test_output, _, _ = model(network, src_test, dst_test)
        test_output_np = test_output.detach().cpu().numpy()
        ratings_test_np = ratings_test.cpu().numpy()
        test_auc = metrics.roc_auc_score(ratings_test_np, test_output_np)
        test_aupr = metrics.average_precision_score(ratings_test_np, test_output_np)
        predicted_classes = (test_output_np > 0.5).astype(int)
        test_acc = accuracy_score(ratings_test_np, predicted_classes)
        test_prec = precision_score(ratings_test_np, predicted_classes)
        test_rec = recall_score(ratings_test_np, predicted_classes)
        test_f1 = f1_score(ratings_test_np, predicted_classes)

        print(
            f'Independent Test Set: AUC = {test_auc:.4f}, AUPR = {test_aupr:.4f}, Acc = {test_acc:.4f}, Prec = {test_prec:.4f}, Rec = {test_rec:.4f}, F1 = {test_f1:.4f}')

    return results, model


def save_mirna_predictions(model_path, network, specified_diseases, density_size, output_file, device):
    # Load the model
    model = MCLAMDA(
        FeatureExtractor(args.emb_size, args.num_layers, network, args.dropout_rate, args.activation_slope, device),
        InteractionPredictor(args.emb_size, device),
        device
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    rna_name_path = 'data/rna_name.csv'
    rna_names = pd.read_csv(rna_name_path)['miRNA'].tolist()

    results = []
    with torch.no_grad():
        for disease in specified_diseases:
            print("disease11: ", disease)
            miRNA_neighbors = network.successors(disease).tolist()
            print("miRNA_neighbors:", miRNA_neighbors)

            if not miRNA_neighbors:
                print(f"No miRNA neighbors found for disease {disease}.")
                continue

            miRNA_neighbors = [node for node in miRNA_neighbors if node >= density_size]

            if not miRNA_neighbors:
                print(f"No valid miRNA neighbors found for disease {disease}.")
                continue

            disease_indices = [disease] * len(miRNA_neighbors)
            disease_tensor = torch.tensor(disease_indices, dtype=torch.int64).to(device)
            mirna_tensor = torch.tensor(miRNA_neighbors, dtype=torch.int64).to(device)

            outputs, _, _ = model(network, disease_tensor, mirna_tensor)
            outputs_np = outputs.cpu().numpy()

            scores = [(disease, mirna - density_size, score) for mirna, score in zip(miRNA_neighbors, outputs_np)]

            scores.sort(key=lambda x: x[2], reverse=True)
            top_scores = scores[:50]
            results.extend(top_scores)

    results_named = [(disease, mirna, rna_names[mirna], score) for disease, mirna, score in results]

    results_df = pd.DataFrame(results_named, columns=['disease', 'miRNA_index', 'miRNA_name', 'score'])
    results_df.sort_values(by=['disease', 'score'], ascending=[True, False], inplace=True)
    results_df.to_csv(output_file, index=False)
    print(f'Saved miRNA predictions to {output_file}')


def plot_final_auc_aupr(fprs, tprs, auc_scores, precisions, recalls, aupr_scores):

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    for i, (fpr, tpr, auc_score) in enumerate(zip(fprs, tprs, auc_scores)):
        ax.plot(fpr, tpr, label=f'Fold {i + 1} (AUC = {auc_score:.4f})')

    ax_inset = ax.inset_axes([0.3, 0.1, 0.37, 0.37])
    for i, (fpr, tpr, auc_score) in enumerate(zip(fprs, tprs, auc_scores)):
        ax_inset.plot(fpr, tpr)
    ax_inset.set_xlim([0.15, 0.25])
    ax_inset.set_ylim([0.85, 0.95])
    ax_inset.set_xticklabels([])
    ax_inset.set_yticklabels([])
    ax_inset.spines['top'].set_linestyle('dashed')
    ax_inset.spines['bottom'].set_linestyle('dashed')
    ax_inset.spines['left'].set_linestyle('dashed')
    ax_inset.spines['right'].set_linestyle('dashed')
    rect = patches.Rectangle((0.15, 0.85), 0.1, 0.1, linewidth=1, edgecolor='r', linestyle='dashed', facecolor='none')
    ax.add_patch(rect)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Fold')
    plt.legend(loc='lower right')
    plt.savefig('roc_curves_optimized.png')
    plt.show()

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    for i, (precision, recall, aupr_score) in enumerate(zip(precisions, recalls, aupr_scores)):
        ax.plot(recall, precision, label=f'Fold {i + 1} (AUPR = {aupr_score:.4f})')

    ax_inset = ax.inset_axes([0.2, 0.2, 0.37, 0.37])
    for i, (precision, recall, aupr_score) in enumerate(zip(precisions, recalls, aupr_scores)):
        ax_inset.plot(recall, precision)
    ax_inset.set_xlim([0.8, 1.0])
    ax_inset.set_ylim([0.8, 1.0])
    ax_inset.set_xticklabels([])
    ax_inset.set_yticklabels([])
    ax_inset.spines['top'].set_linestyle('dashed')
    ax_inset.spines['bottom'].set_linestyle('dashed')
    ax_inset.spines['left'].set_linestyle('dashed')
    ax_inset.spines['right'].set_linestyle('dashed')
    rect = patches.Rectangle((0.8, 0.8), 0.2, 0.2, linewidth=1, edgecolor='r', linestyle='dashed', facecolor='none')
    ax.add_patch(rect)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Each Fold')
    plt.legend(loc='lower left')
    plt.savefig('precision_recall_curves_optimized.png')
    plt.show()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    network, train_val_df, test_df, disease_vertices, mirna_vertices, mappings, density_size = setup_data_and_model(
        args.seed, args.data_dir)
    results, model = train_and_evaluate(network, train_val_df, test_df, disease_vertices, mirna_vertices, mappings,
                                        density_size, torch.device(args.device), args.total_epochs, args.emb_size,
                                        args.num_layers, args.dropout_rate, args.activation_slope, args.learning_rate,
                                        args.weight_decay, args.patience)

    plot_final_auc_aupr(results['fprs'], results['tprs'], results['auc'], results['precisions'], results['recalls'],
                        results['aupr'])

    specified_diseases = [125, 187, 569]

    model_path = 'data/model_fold_3.pth'
    save_mirna_predictions(model_path, network, specified_diseases, density_size, 'miRNA_predictions.csv',
                           torch.device(args.device))
