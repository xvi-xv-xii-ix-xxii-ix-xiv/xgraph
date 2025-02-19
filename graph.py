import matplotlib.pyplot as plt
import networkx as nx

# Define the adjacency matrix representing the graph.
# A value of 1 indicates an edge between the nodes, while 0 indicates no edge.
matrix = [
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # Node 0 is connected to Nodes 1 and 2
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Node 1 is connected to Nodes 0 and 2
    [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # Node 2 is connected to Nodes 0, 1, and 3
    [0, 0, 1, 0, 1, 1, 0, 0, 0, 0],  # Node 3 is connected to Nodes 2, 4, and 5
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # Node 4 is connected to Nodes 3 and 5
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Node 5 is connected to Nodes 3 and 4
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],  # Node 6 is connected to Nodes 7 and 8
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # Node 7 is connected to Nodes 6 and 8
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # Node 8 is connected to Nodes 6, 7, and 9
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Node 9 is connected to Node 8
]

# Create an undirected graph using NetworkX
G = nx.Graph()

# Iterate through the matrix and add edges to the graph
# For each pair of nodes i and j, if there is an edge (matrix[i][j] == 1), add it to the graph.
for i in range(len(matrix)):
    for j in range(i + 1, len(matrix[i])):  # Only check upper triangle to avoid duplicate edges
        if matrix[i][j] == 1:  # If there's an edge, add it to the graph
            G.add_edge(i, j)

# Set up the plot for graph visualization
plt.figure(figsize=(8, 6))  # Set the figure size for the plot

# Draw the graph with customized settings
# - `with_labels=True` ensures that the nodes are labeled
# - `node_size=700` adjusts the size of the nodes
# - `node_color='lightblue'` sets the color of the nodes
# - `font_size=10` and `font_weight='bold'` customize the font of node labels
# - `edge_color='gray'` makes the edges appear in gray
nx.draw(G, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_weight='bold', edge_color='gray')

# Set the title of the graph
plt.title("Graph Representation", fontsize=15)

# Display the graph
plt.show()
