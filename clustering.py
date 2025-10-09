import networkx as nx
import community as community

# Create a graph
G = nx.Graph()

# Add edges with specific weights
G.add_edges_from([
    (1, 2, {'weight': 1.0}),
    (1, 3, {'weight': 1.0}),
    (2, 3, {'weight': 2.0}),
    (3, 4, {'weight': 3.0}),
    (4, 5, {'weight': 1.0}),
    (4, 6, {'weight': 1.0})
])

# Compute the best partition using the edge weights
partition = community.best_partition(G)

# Print the resulting communities
for node, community_id in partition.items():
    print(f"Node {node} belongs to Community {community_id}")