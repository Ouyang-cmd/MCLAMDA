import numpy as np
import pandas as pd
import torch
import dgl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def initialize_graph(num_nodes):
    graph = dgl.DGLGraph(multigraph=True)
    graph.add_nodes(num_nodes)
    return graph


def set_node_classes(graph, boundary):
    node_classes = np.zeros(graph.number_of_nodes(), dtype='float32')
    node_classes[:boundary] = 1
    graph.ndata['node_type'] = node_classes


def set_features(graph, features, feature_name, start_idx, end_idx):
    feature_space = np.zeros((graph.number_of_nodes(), features.shape[1]), dtype='float32')
    feature_space[start_idx:end_idx, :] = features
    graph.ndata[feature_name] = feature_space


def create_index_maps(disease_count, mirna_count):
    disease_map = {i + 1: i for i in range(disease_count)}
    mirna_map = {i + 1: i for i in range(mirna_count)}
    return disease_map, mirna_map


def fetch_data(path):
    d_gsm = np.genfromtxt(f'{path}/d_gs.csv', delimiter=',')
    d_ssm = np.genfromtxt(f'{path}/d_ss.csv', delimiter=',')
    d_tsm = np.genfromtxt(f'{path}/d_ts.csv', delimiter=',')

    m_fsm = np.genfromtxt(f'{path}/m_fs.csv', delimiter=',')
    m_gsm = np.genfromtxt(f'{path}/m_gs.csv', delimiter=',')
    m_ssm = np.genfromtxt(f'{path}/m_ss.csv', delimiter=',')

    combined_disease_features = np.concatenate([d_gsm, d_ssm, d_tsm], axis=1)  # 沿着列方向拼接

    combined_miRNA_features = np.concatenate([m_fsm, m_gsm, m_ssm], axis=1)  # 沿着列方向拼接

    # 将 NumPy 数组转换为 PyTorch 张量
    disease_features_tensor = torch.tensor(combined_disease_features, dtype=torch.float32)
    miRNA_features_tensor = torch.tensor(combined_miRNA_features, dtype=torch.float32)

    print("disease_features_tensor: ",  disease_features_tensor.shape)
    print("miRNA_features_tensor: ", miRNA_features_tensor.shape)

    # disease_features_tensor: torch.Size([591, 1773])
    # miRNA_features_tensor: torch.Size([853, 2559])

    return disease_features_tensor, miRNA_features_tensor


def build_network(folder, interaction_samples, seed):
    density_matrix, mass_matrix = fetch_data(folder)

    network_graph = initialize_graph(density_matrix.shape[0] + mass_matrix.shape[0])
    set_node_classes(network_graph, density_matrix.shape[0])

    set_features(network_graph, density_matrix, 'd_features', 0, density_matrix.shape[0])
    set_features(network_graph, mass_matrix, 'm_features', density_matrix.shape[0], network_graph.number_of_nodes())

    disease_map, mirna_map = create_index_maps(density_matrix.shape[0], mass_matrix.shape[0])

    disease_nodes = [disease_map[id_] for id_ in interaction_samples[:, 1]]
    mirna_nodes = [mirna_map[id_] + density_matrix.shape[0] for id_ in interaction_samples[:, 0]]

    add_interaction_edges(network_graph, disease_nodes, mirna_nodes, interaction_samples[:, 2])

    network_graph.readonly()

    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(density_matrix)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c='blue', marker='o')
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig("disease.png")
    plt.show()

    tsne = TSNE(n_components=2, random_state=42)
    Y_embedded = tsne.fit_transform(mass_matrix)
    plt.figure()
    plt.scatter(Y_embedded[:, 0], Y_embedded[:, 1], c='red', marker='o')
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig("miRNA.png")
    plt.show()

    # Y_embedded = tsne.fit_transform(mass_matrix)
    # plt.scatter(Y_embedded[:, 0], Y_embedded[:, 1], c='red', marker='o')

    # combined_features = np.vstack((density_matrix, mass_matrix))
    # node_labels = np.concatenate((np.zeros(density_matrix.shape[0]), np.ones(mass_matrix.shape[0])))
    # plot_tsne(combined_features, node_labels, "t-SNE Visualization of Disease and miRNA Features")

    return network_graph, create_index_maps(density_matrix.shape[0], mass_matrix.shape[0])


def extract_samples(folder, seed, test_size=0.2):
    positive_file = f'{folder}/all_md_pairs.csv'
    negative_file = f'{folder}/reliable_negative_pairs3.csv'
    association_data = pd.read_csv(positive_file, names=['miRNA', 'disease', 'label'])

    positive_samples = association_data[association_data['label'] == 1]
    negative_samples = pd.read_csv(negative_file, names=['miRNA', 'disease', 'label'], skiprows=1)
    negative_samples['miRNA'] += 1
    negative_samples['disease'] += 1

    mixed_sample_set = positive_samples.append(negative_samples)
    mixed_sample_set.reset_index(drop=True, inplace=True)

    # Shuffle and split the data into training/validation and test sets
    mixed_sample_set = mixed_sample_set.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_split_idx = int(len(mixed_sample_set) * test_size)
    test_samples = mixed_sample_set[:test_split_idx]
    train_val_samples = mixed_sample_set[test_split_idx:]

    return train_val_samples.values, test_samples.values


def add_interaction_edges(graph, disease_nodes, mirna_nodes, labels):
    label_tensor = torch.from_numpy(labels.astype('float32'))

    graph.add_edges(disease_nodes, mirna_nodes, data={'weight': label_tensor})
    graph.add_edges(mirna_nodes, disease_nodes, data={'weight': label_tensor})


