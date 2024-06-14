import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import random


d_gs = pd.read_csv('data/d_gs.csv', header=None)
d_ss = pd.read_csv('data/d_ss.csv', header=None)
d_ts = pd.read_csv('data/d_ts.csv', header=None)
m_fs = pd.read_csv('data/m_fs.csv', header=None)
m_gs = pd.read_csv('data/m_gs.csv', header=None)
m_ss = pd.read_csv('data/m_ss.csv', header=None)
all_md_pairs = pd.read_csv('data/all_md_pairs.csv', names=['miRNA', 'disease', 'label'])


all_md_pairs['miRNA'] -= 1
all_md_pairs['disease'] -= 1


def get_feature_vector(m_index, d_index):
    miRNA_features = np.hstack([m_fs.iloc[m_index].values, m_gs.iloc[m_index].values, m_ss.iloc[m_index].values])
    disease_features = np.hstack([d_gs.iloc[d_index].values, d_ss.iloc[d_index].values, d_ts.iloc[d_index].values])
    return np.hstack([miRNA_features, disease_features])


known_positive = all_md_pairs[all_md_pairs['label'] == 1]
print("known_positive: ", len(known_positive))
unknown_pairs = all_md_pairs[all_md_pairs['label'] == 0]

positive_features = np.array([get_feature_vector(row['miRNA'], row['disease']) for _, row in known_positive.iterrows()])
unlabeled_pairs = [(row['miRNA'], row['disease'], get_feature_vector(row['miRNA'], row['disease'])) for _, row in unknown_pairs.iterrows()]
unlabeled_features = np.array([pair[2] for pair in unlabeled_pairs])

kmeans_positive = KMeans(n_clusters=1).fit(positive_features)
Cp = kmeans_positive.cluster_centers_[0]

kmeans_unlabeled = KMeans(n_clusters=1).fit(unlabeled_features)
Cu = kmeans_unlabeled.cluster_centers_[0]


def cosine_euclidean_similarity(vec, center):
    cos_sim = cosine_similarity([vec], [center])[0][0]
    eucl_sim = 1 / (1 + euclidean(vec, center))
    return cos_sim + eucl_sim

tolerance = 1e-5
while True:
    likely_positive = []
    likely_negative = []

    for m_index, d_index, vec in unlabeled_pairs:
        similarity_to_Cp = cosine_euclidean_similarity(vec, Cp)
        similarity_to_Cu = cosine_euclidean_similarity(vec, Cu)

        if similarity_to_Cp > similarity_to_Cu:
            likely_positive.append((m_index, d_index, vec))
        else:
            likely_negative.append((m_index, d_index, vec))

    new_Cp = np.mean([vec for _, _, vec in likely_positive], axis=0)
    new_Cu = np.mean([vec for _, _, vec in likely_negative], axis=0)

    if np.linalg.norm(new_Cp - Cp) <= tolerance and np.linalg.norm(new_Cu - Cu) <= tolerance:
        break

    Cp, Cu = new_Cp, new_Cu


num_positive = len(known_positive)
print("likely_negative: ", len(likely_negative))
reliable_negative_pairs = random.sample(likely_negative, num_positive)
print("reliable_negative_pairs: ", len(reliable_negative_pairs))
reliable_negative_pairs_df = pd.DataFrame(reliable_negative_pairs, columns=['miRNA', 'disease', 'feature'])
reliable_negative_pairs_df = reliable_negative_pairs_df.drop(columns='feature')
reliable_negative_pairs_df['label'] = 0
reliable_negative_pairs_df.to_csv('reliable_negative_pairs3.csv', index=False)

