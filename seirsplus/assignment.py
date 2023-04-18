from typing import Any, Dict, Optional, Tuple

import numpy as np
from gensim.models import Word2Vec
from networkx import Graph
from node2vec import Node2Vec
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def embed_nodes(
    graph: Graph,
    dimensions: int = 32,
    walk_length: int = 20,
    num_walks: int = 10,
    workers: int = 4,
    window: int = 10,
    min_count: int = 1,
) -> Tuple[np.ndarray, Word2Vec]:
    """Embed nodes in a graph using node2vec.

    Args:
        graph: A networkx Graph object.
        dimensions: The embedding dimension. 
        walk_length, num_walks, workers: Parameters for node2vec.
        window, min_count: Parameters for fitting the node2vec model.
        
    Returns:
        np.ndarray: embedding matrix of shape `(num_nodes, dimensions)`.
    """
    # Generate walks
    node2vec = Node2Vec(
        graph=graph,
        dimensions=dimensions, 
        walk_length=walk_length, 
        num_walks=num_walks, 
        workers=workers, 
        quiet=False,
    )
    
    # Learn embeddings 
    model = node2vec.fit(window=window, min_count=min_count)
    return model.wv.vectors, model


def get_equal_sized_clusters(
    X: np.ndarray, 
    model: Word2Vec,
    graph: Graph, 
    cluster_size: int,
    key_to_index: Optional[Dict[Any, int]] = None,
) -> Dict[Any, int]:
    """Get clusters of equal size for a graph.
    
    Running KMeans then finding the minimal matching of points to clusters 
    under the constraint of maximal points assigned to cluster.
    
    Args:
        X: A `num_samples x num_features`-shaped numpy array of node embeddings.
        model: A `Word2Vec` model.
        graph: A networkx Graph object representing the social network.
        cluster_size: The size of each cluster. 
        key_to_index: An optional dictionary mapping node to index in the 
            embedding matrix.
        
    Returns:
        A dictionary mapping node to cluster id.
    """
    n_clusters = int(np.ceil(len(X) / cluster_size))
    kmeans = KMeans(n_clusters)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    centers = centers.reshape(-1, 1, X.shape[-1]).repeat(cluster_size, 1).reshape(-1, X.shape[-1])
    
    distance_matrix = cdist(X, centers)
    clusters = linear_sum_assignment(distance_matrix)[1]//cluster_size
    clusters = [int(x) for x in clusters]
    # print(model.wv.key_to_index)
    if key_to_index is None:
        clusters = {i: clusters[model.wv.key_to_index[str(i)]] for i in graph.nodes()}
    else:
        clusters = {i: clusters[key_to_index[str(i)]] for i in graph.nodes()}

    return clusters

# def prepare_for_pooled_test(
#     ids
#     assignment,
#     viral_loads, # log10
# ) -> list of lists of viral loads 
# # assume viral load stored in log10 scale
# # log10 viral load

# # TODO: final output: decide on the output, list of ids that are positive
# # how many test were consumed
# # metrics

# def assign(
#     network,
#     cluster_size, 
#     params, # such as dim, num_iter, etc.
#     ):
#     pass # return a single array of cluster assignments
#     # check screening_assignment in google doc
    
    