import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def gini_coefficient(embedding):
    # embedding = embedding.cpu()
    embedding = torch.sort(embedding)[0]
    n = embedding.shape[0]

    # # Calculate the Gini coefficient
    index = torch.arange(1, n + 1, dtype=torch.float32)
    gini = (torch.sum((2 * index - n - 1) * embedding)) / (n * torch.sum(embedding))

    return gini


def measure_diversity(embeddings, diversity_type):
    if diversity_type == "gini":
        gini_values = torch.zeros(embeddings.shape[0])
        for i in range(embeddings.shape[0]):
            gini_values[i] = gini_coefficient(embeddings[i])

        return gini_values
    elif diversity_type == "euclidean":
        embeddings = embeddings.detach().numpy()
        kmeans = KMeans(n_clusters=1, random_state=42, n_init="auto")
        kmeans.fit(embeddings)
        centroid = kmeans.cluster_centers_[0]
        euclidean_distances = torch.Tensor(
            pairwise_distances(embeddings, [centroid])
        ).reshape(-1)

        distances_min = torch.min(euclidean_distances)
        distances_max = torch.max(euclidean_distances)
        normalized_distances = (euclidean_distances - distances_min) / (
            distances_max - distances_min
        )

        return normalized_distances
    elif diversity_type == "cosine":
        embeddings = embeddings.detach().numpy()
        kmeans = KMeans(n_clusters=1, random_state=42, n_init="auto")
        kmeans.fit(embeddings)
        centroid = kmeans.cluster_centers_[0]
        embeddings = torch.Tensor(embeddings)
        centroid = torch.Tensor(centroid)
        cosine_similarity = torch.nn.functional.cosine_similarity(embeddings, centroid)

        return 1 - cosine_similarity
    else:
        raise NotImplementedError


def compute_perturbation_weight(
    diversity, lower_bound=0, upper_bound=0.3, individual_factor=False
):
    lmda = 1 - diversity
    normalized_lmda = lower_bound + (
        (lmda - torch.min(lmda)) * (upper_bound - lower_bound)
    ) / (torch.max(lmda) - torch.min(lmda))

    if individual_factor:
        return normalized_lmda
    else:
        return normalized_lmda.mean().item()
