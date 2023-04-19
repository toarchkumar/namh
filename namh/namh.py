# base namh code
import numpy as np
import itertools
import networkx as nx
from scipy.spatial.distance import cosine

### neighborhood aware metropolis-hastings (namh) code ###
def Initialize(G, k):
    V = len(G.nodes())
    Z = np.zeros((V, k))
    nodes = list(G.nodes())
    unassigned_nodes = set(nodes)

    for i in range(k):
        node = np.random.choice(list(unassigned_nodes))
        unassigned_nodes.remove(node)
        neighbors = list(G.neighbors(node))
        unassigned_nodes -= set(neighbors)
        node_idx = nodes.index(node)
        neighbor_idxs = [nodes.index(neighbor) for neighbor in neighbors]
        Z[neighbor_idxs + [node_idx], i] = 1

        if not unassigned_nodes:
            break

    for node_idx, node in enumerate(nodes):
        community = np.random.randint(0, k)
        Z[node_idx, community] = 1
        
    return Z

def Proposal(u_idx, G, Z, k, l, delta):
    nodes = list(G.nodes())
    u = nodes[u_idx]
    neighbors = list(G.neighbors(u))
    neighbor_idxs = [nodes.index(neighbor) for neighbor in neighbors]
    S = np.where(np.sum(Z[neighbor_idxs, :], axis=0) > 0)[0]
    subsets = []
    for i in range(1, l + 1):
        subsets += itertools.combinations(S, i)

    if not subsets:
        return Z.copy()

    f_map = {i: ObjectiveFunction(u_idx, Z, subset, delta) for i, subset in enumerate(subsets)}
    softmax = {}
    max_f = max(f_map.values())
    total = sum(np.exp(value - max_f) for value in f_map.values()) + 1e-12  # Add a small constant to avoid division by zero
    for key in f_map:
        softmax[key] = np.exp(f_map[key] - max_f) / total
    idx = np.random.choice(list(softmax.keys()), p=list(softmax.values()))
    s = subsets[idx]
    z_new = np.zeros(k)
    z_new[list(s)] = 1
    Z_proposal = Z.copy()
    Z_proposal[u_idx, :] = z_new
    return Z_proposal

def ObjectiveFunction(u_idx, Z, subset, delta):
    k = Z.shape[1]
    z_temp = Z[u_idx, :].copy()
    z_temp.fill(0)
    z_temp[list(subset)] = 1
    neighbors = np.where(z_temp == 1)[0]
    sizes = np.sum(Z, axis=0)
    sizes[neighbors] -= 1
    sizes[sizes == 0] = 1
    objective = 0
    for i in range(k):
        if i in neighbors:
            size = sizes[i] + 1
        else:
            size = sizes[i]
        if size == 0:
            continue
        objective += np.log(size + 1e-12) - np.log(Z.shape[0] - size + 1e-12)
    objective *= delta
    return objective

def AcceptanceProbability(u_idx, G, Z, k, l, T, delta):
    Z_proposal = Proposal(u_idx, G, Z, k, l, delta)
    f_u_Z = ObjectiveFunction(u_idx, Z, (), delta)
    f_u_Z_proposal = ObjectiveFunction(u_idx, Z_proposal, (), delta)
    diff = f_u_Z_proposal - f_u_Z
    if diff > 0:
        acceptance_prob = 1
    else:
        acceptance_prob = np.exp(diff / T)  # Divide by T instead of multiplying
    return acceptance_prob

# Final function to calculate NAMH
def NAMH(G, k, T, epochs, l, delta):
    Z = Initialize(G, k)
    community_count = np.zeros_like(Z)  # Initialize the community_count matrix
    V = np.arange(len(G.nodes()))
    nodes = list(G.nodes())
    for t in range(epochs):
        np.random.shuffle(V)
        for u_idx in V:
            Z_prime = Proposal(u_idx, G, Z, k, l, delta)
            p = AcceptanceProbability(u_idx, G, Z, k, l, T, delta)
            if np.random.uniform() < p:
                Z[u_idx] = Z_prime[u_idx]
        community_count += Z  # Update the community_count matrix after every epoch
    community_prob = community_count / epochs  # Normalize the community_count matrix
    return community_prob

### similarity code ###
def similarity(v1, v2):
    return 1 - cosine(v1, v2)

def build_graph(vector_database, input_prompt):
    G = nx.Graph()
    n = len(vector_database)
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=similarity(vector_database[i], vector_database[j]))
    
    G.add_node(n)
    for i in range(n):
        G.add_edge(n, i, weight=similarity(input_prompt, vector_database[i]))
    
    return G

def find_closest_match(input_prompt, vector_database, k=2, T=1.0, epochs=100, l=2, delta=0.5):
    G = build_graph(vector_database, input_prompt)
    result = NAMH(G, k, T, epochs, l, delta)
    input_community = np.argmax(result[-1])
    node_indices = np.where(result[:, input_community] == 1)[0]
    node_indices = node_indices[node_indices != len(vector_database)]  # Exclude the input prompt node

    closest_match = None
    highest_similarity = -1
    community_vectors = []
    for idx in node_indices:
        sim = similarity(input_prompt, vector_database[idx])
        if sim > highest_similarity:
            highest_similarity = sim
            closest_match = idx
        community_vectors.append(vector_database[idx])

    return closest_match, highest_similarity, community_vectors


if __name__ == '__main__':
    ### test the code ###

    # Create a vector database with 5 sample vectors
    vector_database = np.array([
        [0.8, 0.2, 0.3],
        [0.9, 0.1, 0.2],
        [0.1, 0.8, 0.7],
        [0.2, 0.9, 0.6],
        [0.5, 0.5, 0.5]
    ])

    # Create an input prompt vector
    input_prompt = np.array([0.2, 0.4, 0.3])

    # Set parameters for the NAMH algorithm
    k = 2
    T = 1
    epochs = 1000
    l = 1
    delta = 0.5

    # Find the closest match in the vector database for the input prompt
    closest_match, highest_similarity, community_vectors = find_closest_match(input_prompt, vector_database, k, T, epochs, l, delta)

    print(f"Closest match: Vector index {closest_match}, Similarity: {highest_similarity}")
    print(f"Vectors in community: {community_vectors}")