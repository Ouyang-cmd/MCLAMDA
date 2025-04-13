import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class MCLAMDA(nn.Module):

    def __init__(self, featureExtractor, interactionPredictor, device):
        super(MCLAMDA, self).__init__()
        self.featureExtractor = featureExtractor
        self.interactionPredictor = interactionPredictor
        self.device = device
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, graph, diseases, mirnas):
        graph = graph.to(self.device)

        h, src_train_initial_features, dst_train_initial_features = self.featureExtractor(graph, diseases, mirnas)
        h_diseases = h[diseases].to(self.device)
        h_mirnas = h[mirnas].to(self.device)

        pos_sim = F.cosine_similarity(h_diseases, src_train_initial_features, dim=1)
        neg_sim = F.cosine_similarity(h_diseases, dst_train_initial_features, dim=1)

        contrastive_loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim))).mean()

        feature = graph.ndata['emb']

        k = 50
        updated_h_diseases = self.knn_feature_update(feature[:591], k)
        updated_h_mirnas = self.knn_feature_update(feature[591:], k)

        feature_contrastive_loss = self.compute_knn_contrastive_loss(updated_h_diseases, updated_h_mirnas)

        return self.interactionPredictor(h_diseases, h_mirnas), contrastive_loss, feature_contrastive_loss

    def knn_feature_update(self, features, k):
        sim_matrix = torch.matmul(features, features.T)
        _, indices = torch.topk(sim_matrix, k=k + 1, dim=1)
        indices = indices[:, 1:]
        neighbors_features = features[indices]
        updated_features = neighbors_features.mean(dim=1)
        return updated_features

    def compute_knn_contrastive_loss(self, features_diseases, features_mirnas):
        center_vector_A = features_diseases.mean(dim=0)
        center_vector_B = features_mirnas.mean(dim=0)

        sim_to_A_disease = F.cosine_similarity(features_diseases, center_vector_A.unsqueeze(0), dim=1)
        sim_to_B_disease = F.cosine_similarity(features_diseases, center_vector_B.unsqueeze(0), dim=1)
        loss_disease = -torch.log(
            torch.exp(sim_to_A_disease) / (torch.exp(sim_to_A_disease) + torch.exp(sim_to_B_disease))).mean()

        sim_to_B_miRNA = F.cosine_similarity(features_mirnas, center_vector_B.unsqueeze(0), dim=1)
        sim_to_A_miRNA = F.cosine_similarity(features_mirnas, center_vector_A.unsqueeze(0), dim=1)
        loss_miRNA = -torch.log(
            torch.exp(sim_to_B_miRNA) / (torch.exp(sim_to_B_miRNA) + torch.exp(sim_to_A_miRNA))).mean()

        return (loss_disease + loss_miRNA) / 2


class FeatureExtractor(nn.Module):
    def __init__(self, embedding_size, n_layers, graph, dropout, slope, ctx):
        super(FeatureExtractor , self).__init__()
        self.graph = graph.to(ctx)
        self.disease_nodes = graph.nodes()[graph.ndata['node_type'] == 1]
        self.mirna_nodes = graph.nodes()[graph.ndata['node_type'] == 0]
        self.ctx = ctx
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.layers = nn.ModuleList([
            Aggregator(embedding_size, graph, self.disease_nodes, self.mirna_nodes, dropout, slope, ctx)
            for _ in range(n_layers)
        ])
        self.init_embedding_and_attention()

    def init_embedding_and_attention(self):
        self.embedding_proj = {
            'disease': nn.Sequential(
                nn.Linear(1773, self.embedding_size, bias=False),
                nn.Dropout(self.dropout)
            ).to(self.ctx),
            'mirna': nn.Sequential(
                nn.Linear(2559, self.embedding_size, bias=False),
                nn.Dropout(self.dropout)
            ).to(self.ctx)
        }
        self.attention = {
            'disease': nn.MultiheadAttention(self.embedding_size, num_heads=4).to(self.ctx),
            'mirna': nn.MultiheadAttention(self.embedding_size, num_heads=4).to(self.ctx)
        }

    def process_nodes(self, node_type, features):
        proj = self.embedding_proj[node_type](features).unsqueeze(0)
        attn_output, _ = self.attention[node_type](proj, proj, proj)
        return attn_output.squeeze(0)

    def forward(self, graph, src_train=None, dst_train=None):
        graph = graph.to(self.ctx)
        graph.ndata['emb'] = torch.zeros((graph.number_of_nodes(), self.embedding_size), device=self.ctx)
        graph.ndata['emb'][self.disease_nodes] = self.process_nodes('disease',
                                                              graph.ndata['d_features'][self.disease_nodes])
        graph.ndata['emb'][self.mirna_nodes] = self.process_nodes('mirna', graph.ndata['m_features'][self.mirna_nodes])

        initial_features = graph.ndata['emb'].clone()

        src_initial_features = initial_features[src_train] if src_train is not None else None
        dst_initial_features = initial_features[dst_train] if dst_train is not None else None

        for layer in self.layers:
            layer(graph)

        return graph.ndata['emb'], src_initial_features, dst_initial_features


class InteractionPredictor(nn.Module):
    def __init__(self, size, device):

        super(InteractionPredictor, self).__init__()

        self.transformation_matrix = nn.Parameter(torch.randn(size, size, device=device))

        self.sigmoid_activation = nn.Sigmoid()

    def forward(self, features_diseases, features_miRNAs):

        interaction_matrix = torch.matmul(features_diseases, self.transformation_matrix)

        pre_activation_results = torch.sum(interaction_matrix * features_miRNAs, dim=1)

        interaction_probabilities = self.sigmoid_activation(pre_activation_results)

        return interaction_probabilities


class Aggregator(nn.Module):
    def __init__(self, feature_size, graph, disease_nodes, mirna_nodes, dropout, slope, device):
        super(Aggregator, self).__init__()
        self.feature_size = feature_size
        self.graph = graph
        self.disease_nodes = disease_nodes
        self.mirna_nodes = mirna_nodes
        self.device = device
        self.dropout = dropout
        self.slope = slope
        self.leakyrelu = nn.LeakyReLU(slope).to(device)
        self.W_self = nn.Linear(feature_size, feature_size).to(device)
        self.W_neigh = nn.Linear(feature_size, feature_size).to(device)
        self.dropout_layer = nn.Dropout(dropout).to(device)
        self.residual = nn.Linear(feature_size, feature_size).to(device) # 残差连接
        all_nodes = torch.arange(graph.number_of_nodes(), dtype=torch.int64, device=device)
        self.degree = graph.in_degrees(all_nodes).float().to(device)
        self.message_func = fn.copy_src(src='emb', out='msg')
        self.reduce_func = fn.mean(msg='msg', out='update')

    def forward(self, graph):
        assert graph.number_of_nodes() == self.graph.number_of_nodes()
        graph.ndata['degree'] = self.degree
        graph.update_all(self.message_func, self.reduce_func)
        graph.ndata['emb'] = self.apply_node_updates(graph)

    def apply_node_updates(self, graph):
        updated_features  = graph.ndata['emb'].clone()
        for ntype in [self.disease_nodes, self.mirna_nodes]:
            h = graph.ndata['emb'][ntype]
            update = graph.ndata['update'][ntype]
            degree = graph.ndata['degree'][ntype].unsqueeze(1)
            self_h = self.dropout_layer(self.leakyrelu(self.W_self(h)))
            neigh_h = self.dropout_layer(self.leakyrelu(self.W_neigh(update / torch.clamp(degree, min=1e-6))))
            aggregated_features  = self_h + neigh_h
            aggregated_features  = self.leakyrelu(self.residual(aggregated_features ))
            updated_features [ntype] = aggregated_features
        return updated_features




