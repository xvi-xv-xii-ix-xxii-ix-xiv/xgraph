"""
This script visualizes a heterogeneous multigraph representing a network (e.g., Viking trade routes)
using NetworkX and Matplotlib. It loads node and edge data from CSV files, constructs a multigraph,
and draws it with customized node colors, edge styles, and labels. Nodes represent entities like
ports or settlements, while edges represent connections with attributes like route type and weight.

Requirements:
- pandas: For loading and processing CSV data.
- networkx: For creating and manipulating the graph.
- matplotlib: For visualizing the graph.

Input:
- nodes_file: CSV file with node data (node_id, data, region, type, population).
- edges_file: CSV file with edge data (from, to, weight, data, priority).
"""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

def load_graph(nodes_file, edges_file):
    """
    Load node and edge data from CSV files and construct a NetworkX MultiGraph.

    Parameters:
    - nodes_file (str): Path to the CSV file containing node data.
    - edges_file (str): Path to the CSV file containing edge data.

    Returns:
    - G (nx.MultiGraph): A NetworkX MultiGraph object with nodes and edges.
    """
    # Load data from CSV files into pandas DataFrames
    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)

    # Ensure node_id, from, and to columns are integers
    nodes_df["node_id"] = nodes_df["node_id"].astype(int)
    edges_df["from"] = edges_df["from"].astype(int)
    edges_df["to"] = edges_df["to"].astype(int)

    # Handle missing values (NaN) in node attributes by filling with "unknown" and converting to strings
    nodes_df[["data", "region", "type"]] = nodes_df[["data", "region", "type"]].fillna("unknown").astype(str)
    nodes_df["population"] = nodes_df["population"].astype(str)  # Population as string for flexibility

    # Handle missing values in edge attributes
    edges_df["weight"] = edges_df["weight"].astype(float)  # Ensure weight is a float
    edges_df[["data", "priority"]] = edges_df[["data", "priority"]].fillna("unknown").astype(str)

    # Initialize an empty MultiGraph (allows multiple edges between the same nodes)
    G = nx.MultiGraph()

    # Add nodes to the graph with all their attributes
    for _, row in nodes_df.iterrows():
        G.add_node(row["node_id"], **row.to_dict())

    # Add edges to the graph with their attributes (weight, route, priority)
    for _, row in edges_df.iterrows():
        edge_attributes = {
            "weight": row["weight"],
            "route": row["data"],  # 'data' column repurposed as route type
            "priority": row["priority"],
        }
        G.add_edge(row["from"], row["to"], **edge_attributes)

    return G

def draw_graph(G):
    """
    Visualize the MultiGraph with customized node colors, edge styles, and labels.

    Parameters:
    - G (nx.MultiGraph): The graph to visualize.
    """
    # Set up the figure size for the plot
    plt.figure(figsize=(10, 8))

    # Generate a layout for the graph using the spring layout algorithm with a fixed seed for reproducibility
    pos = nx.spring_layout(G, seed=42, k=0.3)  # k controls node spacing

    # Define a color mapping for different node types
    node_color_map = {
        "port": "#1f77b4",          # Blue
        "trade_hub": "#ff7f0e",     # Orange
        "settlement": "#2ca02c",    # Green
        "fortress": "#d62728",      # Red
        "religious_center": "#9467bd",  # Purple
        "unknown": "#7f7f7f",       # Gray (default)
    }
    # Assign colors to nodes based on their type
    node_colors = [
        node_color_map.get(G.nodes[n].get("type", "unknown"), "#7f7f7f")
        for n in G.nodes
    ]

    # Draw the nodes with specified colors, size, and edge outlines
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, edgecolors="black", node_size=100, alpha=0.9
    )

    # Create labels for nodes showing node_id, data, type, and population
    labels = {
        n: f"{n}: {G.nodes[n]['data']}\n({G.nodes[n].get('type', 'unknown')}, Pop: {G.nodes[n].get('population', 'N/A')})"
        for n in G.nodes
    }
    # Offset label positions slightly above nodes for better readability
    label_pos = {k: (v[0], v[1] + 0.05) for k, v in pos.items()}
    nx.draw_networkx_labels(G, label_pos, labels, font_size=8, font_weight="bold")

    # Define color and style mappings for different edge route types
    edge_color_map = {
        "sea": "blue",
        "river": "green",
        "trade": "orange",
        "mountain": "brown",
        "unknown": "gray",
    }
    edge_style_map = {
        "sea": "solid",
        "river": "dashed",
        "trade": "dotted",
        "mountain": "dashdot",
        "unknown": "solid",
    }

    # Draw edges with curvature for multiple edges between the same nodes
    for u, v, key, data in G.edges(keys=True, data=True):
        route_type = data.get("route", "unknown")
        edge_color = edge_color_map.get(route_type, "gray")
        edge_style = edge_style_map.get(route_type, "solid")

        # Calculate curvature (rad) for multiple edges between the same nodes
        num_edges = len(G[u][v])
        if num_edges > 1:
            rad = 0.1 * key  # Curvature increases with edge key
        else:
            rad = 0.0

        # Draw the edge with specified style, color, and curvature
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            edge_color=edge_color,
            style=edge_style,
            width=2,
            alpha=0.7,
            arrows=True,  # Show direction of edges
            connectionstyle=f"arc3,rad={rad}",  # Apply curvature
        )

    # Add a legend for edge types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=edge_color_map["sea"], lw=2, label="Sea"),
        Line2D([0], [0], color=edge_color_map["river"], lw=2, linestyle="--", label="River"),
        Line2D([0], [0], color=edge_color_map["trade"], lw=2, linestyle=":", label="Trade"),
        Line2D([0], [0], color=edge_color_map["mountain"], lw=2, linestyle="-.", label="Mountain"),
        Line2D([0], [0], color=edge_color_map["unknown"], lw=2, label="Unknown"),
    ]
    plt.legend(handles=legend_elements, loc="upper left", fontsize=8)

    # Set the plot title and adjust layout
    plt.title("Heterogeneous MultiGraph Visualization", fontsize=12, fontweight="bold")
    plt.tight_layout()  # Automatically adjust spacing to prevent overlap
    plt.axis("off")  # Hide axes
    plt.show()

if __name__ == "__main__":
    """
    Main execution block: Load the graph from Viking-themed CSV files and visualize it.
    """
    graph = load_graph("viking_nodes.csv", "viking_edges.csv")
    draw_graph(graph)