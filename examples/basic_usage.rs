use std::collections::HashMap;
use xgraph::algorithms::connectivity::Connectivity;
use xgraph::algorithms::leiden_clustering::{CommunityConfig, CommunityDetection};
use xgraph::algorithms::wiedemann_ford::DominatingSetFinder;
<<<<<<< HEAD
use xgraph::io::{
    csv_io::CsvIO,
};
use xgraph::prelude::*;

type WeightType = u32; // Main type for edge weights

/// Function to create a graph from an adjacency matrix.
///
/// # Arguments
/// * `matrix` - A vector of vectors representing the adjacency matrix of the graph.
/// * `directed` - A boolean indicating whether the graph is directed.
///
/// # Returns
/// A `Graph<WeightType, (), ()>` object created from the given matrix.
=======
use xgraph::io::csv_io::CsvIO;
use xgraph::prelude::*;

/// Type alias for edge weights used throughout the graph analysis.
type WeightType = u32;

/// Creates a graph from an adjacency matrix representation.
///
/// This function constructs a graph where edges are defined by non-zero values in the matrix.
/// The resulting graph uses `WeightType` for edge weights and empty tuples for node and edge data.
///
/// # Arguments
/// * `matrix` - A 2D vector where `matrix[i][j]` represents the weight of an edge from node `i` to node `j`. A value of 0 indicates no edge.
/// * `directed` - A boolean indicating whether the graph is directed (true) or undirected (false).
///
/// # Returns
/// A `Graph<WeightType, (), ()>` instance initialized from the matrix.
///
/// # Panics
/// Panics if the matrix is malformed or cannot be converted into a valid graph (e.g., non-square matrix).
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)
fn create_graph_from_matrix(
    matrix: Vec<Vec<WeightType>>,
    directed: bool,
) -> Graph<WeightType, (), ()> {
    Graph::from_adjacency_matrix(&matrix, directed, (), ())
        .expect("Failed to create graph from matrix")
}

<<<<<<< HEAD
/// Function to print details of the graph nodes and edges.
///
/// # Arguments
/// * `graph` - A reference to the `Graph<WeightType, (), ()>` object.
fn print_graph_details(graph: &Graph<WeightType, (), ()>) {
    let nodes: Vec<(usize, &())> = graph.all_nodes().collect(); // Explicitly specify the type of vertices
    let edges = graph.get_all_edges();
=======
/// Prints detailed information about the graph's nodes and edges.
///
/// This function provides a summary of the graph structure, including the number of nodes and edges,
/// and lists their respective IDs and connections.
///
/// # Arguments
/// * `graph` - A reference to the `Graph<WeightType, (), ()>` to analyze.
fn print_graph_details(graph: &Graph<WeightType, (), ()>) {
    let nodes: Vec<(usize, &())> = graph.all_nodes().collect(); // Collect all nodes into a vector
    let edges = graph.get_all_edges(); // Retrieve all edges
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)

    println!("\n================== Graph Details ==================");
    println!(
        "Nodes ({}): {:?}",
        nodes.len(),
<<<<<<< HEAD
        nodes.iter().map(|(id, _)| id).collect::<Vec<_>>()
    );
    println!("Edges ({}): {:?}", edges.len(), edges);
}

/// Function to demonstrate IO.
fn demonstrate_io(graph: &mut Graph<WeightType, (), ()>) {
    println!("\n================== IO and Format Demonstration ==================");

    let string_graph = graph.to_string_graph();
=======
        nodes.iter().map(|(id, _)| id).collect::<Vec<_>>() // List node IDs
    );
    println!("Edges ({}): {:?}", edges.len(), edges); // List edge details
}

/// Demonstrates input/output operations with the graph.
///
/// Saves the graph to CSV files and then loads it back to verify the IO process.
/// Prints the steps and results for clarity.
///
/// # Arguments
/// * `graph` - A mutable reference to the `Graph<WeightType, (), ()>` to save and load.
fn demonstrate_io(graph: &mut Graph<WeightType, (), ()>) {
    println!("\n================== IO and Format Demonstration ==================");

    let string_graph = graph.to_string_graph(); // Convert to a string-based graph for IO
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)

    // 1. Save to CSV
    println!("\n[Saving to CSV]");
    string_graph
        .save_to_csv("nodes.csv", "edges.csv")
        .expect("Failed to save to CSV");
    println!("Graph saved to nodes.csv and edges.csv");

    // 2. Load from CSV
    println!("\n[Loading from CSV]");
    let loaded_graph =
        Graph::<WeightType, String, String>::load_from_csv("nodes.csv", "edges.csv", true)
            .expect("Failed to load from CSV");
    println!(
        "Loaded graph: {} nodes, {} edges",
        loaded_graph.nodes.len(),
        loaded_graph.edges.len()
    );
}

<<<<<<< HEAD
/// Function to analyze the graph and print the results.
///
/// # Arguments
/// * `graph` - A mutable reference to the `Graph<WeightType, (), ()>` object.
fn analyze_graph(graph: &mut Graph<WeightType, (), ()>) {
    println!("\n================== Graph Analysis ==================");

    // 1. Basic metrics of the graph
=======
/// Analyzes the graph and prints various structural properties.
///
/// This function performs a comprehensive analysis, including metrics, connectivity,
/// centrality, attributes, bridges, components, density, and dominating sets.
///
/// # Arguments
/// * `graph` - A mutable reference to the `Graph<WeightType, (), ()>` to analyze.
fn analyze_graph(graph: &mut Graph<WeightType, (), ()>) {
    println!("\n================== Graph Analysis ==================");

    // 1. Basic metrics
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)
    let num_nodes = graph.nodes.len();
    let num_edges = graph.get_all_edges().len();
    print_metrics(num_nodes, num_edges, graph.directed);

    // 2. Connectivity and paths
    print_connectivity(graph, num_nodes);

    // 3. Centrality
    print_centrality(graph);

    // 4-5. Node and edge attributes
    print_attributes(graph);

    // 6. Bridges
    print_bridges(graph);

    // 7. Connected components
    print_connected_components(graph);

    // 8. Graph density
    print_density(num_nodes, num_edges, graph.directed);

<<<<<<< HEAD
    // 9. Example usage of the Wiedemann-Ford algorithm
    print_wiedemann_ford(graph);
}

// Separate functions to improve readability

/// Function to print basic metrics of the graph.
///
/// # Arguments
/// * `nodes` - Number of nodes in the graph.
/// * `edges` - Number of edges in the graph.
/// * `directed` - A boolean indicating whether the graph is directed.
=======
    // 9. Wiedemann-Ford dominating set
    print_wiedemann_ford(graph);
}

/// Prints basic metrics about the graph structure.
///
/// Displays the number of nodes, edges, and graph type (directed or undirected).
///
/// # Arguments
/// * `nodes` - The number of nodes in the graph.
/// * `edges` - The number of edges in the graph.
/// * `directed` - A boolean indicating if the graph is directed.
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)
fn print_metrics(nodes: usize, edges: usize, directed: bool) {
    println!("\n[Basic Metrics]");
    println!("Number of nodes: {}", nodes);
    println!("Number of edges: {}", edges);
    println!(
        "Type of graph: {}",
        if directed { "directed" } else { "undirected" }
    );
}

<<<<<<< HEAD
/// Function to print connectivity and path information.
///
/// # Arguments
/// * `graph` - A mutable reference to the `Graph<WeightType, (), ()>` object.
/// * `node_count` - Number of nodes in the graph.
fn print_connectivity(graph: &mut Graph<WeightType, (), ()>, node_count: usize) {
    println!("\n[Connectivity and Paths]");
    if node_count >= 6 {
        println!("Path from 0 to 5 exists: {}", graph.has_path(0, 5));
        println!("Shortest path 0->5: {:?}", graph.bfs_path(0, 5));
    }
}

/// Function to print centrality information of the graph.
///
/// # Arguments
/// * `graph` - A mutable reference to the `Graph<WeightType, (), ()>` object.
fn print_centrality(graph: &mut Graph<WeightType, (), ()>) {
    println!("\n[Centrality]");
    let centrality = graph.degree_centrality();
=======
/// Prints connectivity information and example paths in the graph.
///
/// Checks for paths between specific nodes (if enough nodes exist) and displays shortest paths.
///
/// # Arguments
/// * `graph` - A mutable reference to the `Graph<WeightType, (), ()>` to analyze.
/// * `node_count` - The total number of nodes in the graph.
fn print_connectivity(graph: &mut Graph<WeightType, (), ()>, node_count: usize) {
    println!("\n[Connectivity and Paths]");
    if node_count >= 6 {
        // Example path check between nodes 0 and 5
        println!("Path from 0 to 5 exists: {}", graph.has_path(0, 5));
        println!("Shortest path 0->5: {:?}", graph.bfs_path(0, 5));
    } else {
        println!("Not enough nodes for path example (need at least 6)");
    }
}

/// Prints centrality metrics for the graph's nodes.
///
/// Computes and displays the degree centrality for each node.
///
/// # Arguments
/// * `graph` - A mutable reference to the `Graph<WeightType, (), ()>` to analyze.
fn print_centrality(graph: &mut Graph<WeightType, (), ()>) {
    println!("\n[Centrality]");
    let centrality = graph.degree_centrality(); // Calculate degree centrality
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)
    println!("Degree centrality:");
    centrality
        .iter()
        .for_each(|(node, val)| println!("  Node {}: {:.2}", node, val));
}

<<<<<<< HEAD
/// Function to print attributes of nodes and edges.
///
/// # Arguments
/// * `graph` - A mutable reference to the `Graph<WeightType, (), ()>` object.
fn print_attributes(graph: &mut Graph<WeightType, (), ()>) {
    // Node attributes (unchanged)
=======
/// Prints node and edge attributes present in the graph.
///
/// Aggregates and displays all attributes, grouping them by key and value.
///
/// # Arguments
/// * `graph` - A mutable reference to the `Graph<WeightType, (), ()>` to analyze.
fn print_attributes(graph: &mut Graph<WeightType, (), ()>) {
    // Aggregate node attributes
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)
    let node_attrs = graph
        .nodes
        .iter()
        .flat_map(|(id, node)| node.attributes.iter().map(move |(k, v)| (k, v, id)))
        .fold(HashMap::new(), |mut acc, (k, v, id)| {
            acc.entry(k)
                .or_insert(HashMap::new())
                .entry(v)
                .or_insert(Vec::new())
                .push(id);
            acc
        });

    println!("\n[Node Attributes]");
    if node_attrs.is_empty() {
        println!("No node attributes");
    } else {
        node_attrs.iter().for_each(|(attr, values)| {
            println!("Attribute '{}':", attr);
            values
                .iter()
                .for_each(|(val, ids)| println!("  {}: {} nodes ({:?})", val, ids.len(), ids));
        });
    }

<<<<<<< HEAD
    // Edge attributes (fixed version)
=======
    // Aggregate edge attributes
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)
    let edge_attrs = graph
        .get_all_edges()
        .iter()
        .flat_map(|(from, to, weight, _)| {
            graph
                .get_all_edge_attributes(*from, *to)
                .into_iter()
                .flat_map(|attrs| attrs.iter())
                .map(move |(k, v)| (k.clone(), v.clone(), (*from, *to, *weight)))
        })
        .fold(HashMap::new(), |mut acc, (k, v, edge)| {
            acc.entry(k)
                .or_insert(HashMap::new())
                .entry(v)
                .or_insert(Vec::new())
                .push(edge);
            acc
        });
    println!("\n[Edge Attributes]");
    if edge_attrs.is_empty() {
        println!("No edge attributes");
    } else {
        edge_attrs.iter().for_each(|(attr, values)| {
            println!("Attribute '{}':", attr);
            values.iter().for_each(|(val, edges)| {
                println!("  {}: {} edges ({:?})", val, edges.len(), edges)
            });
        });
    }
}

<<<<<<< HEAD
/// Function to print bridges (critical edges) of the graph.
///
/// # Arguments
/// * `graph` - A mutable reference to the `Graph<WeightType, (), ()>` object.
fn print_bridges(graph: &mut Graph<WeightType, (), ()>) {
    println!("\n[Bridges]");
    let bridges = graph.find_bridges();
    println!("Bridges (critical edges): {:?}", bridges);
}

/// Function to print connected components of the graph.
///
/// # Arguments
/// * `graph` - A mutable reference to the `Graph<WeightType, (), ()>` object.
=======
/// Prints bridges (critical edges) in the graph.
///
/// Identifies and lists edges whose removal would disconnect the graph.
///
/// # Arguments
/// * `graph` - A mutable reference to the `Graph<WeightType, (), ()>` to analyze.
fn print_bridges(graph: &mut Graph<WeightType, (), ()>) {
    println!("\n[Bridges]");
    let bridges = graph.find_bridges(); // Find all bridges
    println!("Bridges (critical edges): {:?}", bridges);
}

/// Prints information about connected components in the graph.
///
/// For directed graphs, shows both strongly and weakly connected components.
/// For undirected graphs, shows regular connected components.
///
/// # Arguments
/// * `graph` - A mutable reference to the `Graph<WeightType, (), ()>` to analyze.
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)
fn print_connected_components(graph: &mut Graph<WeightType, (), ()>) {
    println!("\n[Connected Components]");

    let components = if graph.is_directed() {
<<<<<<< HEAD
        // For directed graphs, show both types of components
=======
        // Directed graph: show both strong and weak components
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)
        println!("Strongly connected components:");
        let scc = graph.find_strongly_connected_components();
        println!("  Count: {}", scc.len());

        println!("Weakly connected components:");
        let wcc = graph.find_weakly_connected_components();
        println!("  Count: {}", wcc.len());
        wcc
    } else {
<<<<<<< HEAD
        // For undirected graphs, regular components
=======
        // Undirected graph: regular components
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)
        graph.find_connected_components()
    };

    components
        .iter()
        .enumerate()
        .for_each(|(i, c)| println!("  Component {}: {} nodes", i, c.len()));

    println!("Overall connectivity: {}", graph.is_connected());

    // Additional connectivity checks
    println!("\nAdditional checks:");
    println!("Weakly connected: {}", graph.is_weakly_connected());
    if graph.is_directed() {
        println!("Strongly connected: {}", graph.is_strongly_connected());
    }
}

<<<<<<< HEAD
/// Function to print density of the graph.
///
/// # Arguments
/// * `nodes` - Number of nodes in the graph.
/// * `edges` - Number of edges in the graph.
/// * `directed` - Indicates if the graph is directed.
=======
/// Prints the density of the graph.
///
/// Calculates and displays the graph's density as the ratio of actual to possible edges.
///
/// # Arguments
/// * `nodes` - The number of nodes in the graph.
/// * `edges` - The number of edges in the graph.
/// * `directed` - A boolean indicating if the graph is directed.
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)
fn print_density(nodes: usize, edges: usize, directed: bool) {
    println!("\n[Density]");
    let density = calculate_density(nodes, edges, directed);
    println!("Density: {:.4}\n", density);
}

<<<<<<< HEAD
/// Function to calculate the density of the graph.
///
/// # Arguments
/// * `nodes` - Number of nodes in the graph.
/// * `edges` - Number of edges in the graph.
/// * `directed` - Indicates if the graph is directed.
///
/// Returns:
/// The calculated density as a float.
fn calculate_density(nodes: usize, edges: usize, directed: bool) -> f64 {
    if nodes <= 1 {
        return 0.0;
    }
    let possible_edges = if directed {
        nodes * (nodes - 1)
    } else {
        nodes * (nodes - 1) / 2
=======
/// Calculates the density of the graph.
///
/// Density is the ratio of existing edges to the maximum possible edges in the graph.
///
/// # Arguments
/// * `nodes` - The number of nodes in the graph.
/// * `edges` - The number of edges in the graph.
/// * `directed` - A boolean indicating if the graph is directed.
///
/// # Returns
/// The density as a floating-point value between 0.0 and 1.0.
fn calculate_density(nodes: usize, edges: usize, directed: bool) -> f64 {
    if nodes <= 1 {
        return 0.0; // No edges possible with 0 or 1 node
    }
    let possible_edges = if directed {
        nodes * (nodes - 1) // Directed: n * (n-1) possible edges
    } else {
        nodes * (nodes - 1) / 2 // Undirected: n * (n-1) / 2 possible edges
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)
    };
    edges as f64 / possible_edges as f64
}

<<<<<<< HEAD
/// Function to print results of the Wiedemann-Ford algorithm on the graph.
///
/// # Arguments
/// * `graph` - A mutable reference to the `Graph<WeightType, (), ()>` object.
fn print_wiedemann_ford(graph: &mut Graph<WeightType, (), ()>) {
    let dominating_set = graph.find_dominating_set();
=======
/// Prints the dominating set found using the Wiedemann-Ford algorithm.
///
/// Identifies a minimal set of nodes that dominate all others in the graph.
///
/// # Arguments
/// * `graph` - A mutable reference to the `Graph<WeightType, (), ()>` to analyze.
fn print_wiedemann_ford(graph: &mut Graph<WeightType, (), ()>) {
    let dominating_set = graph.find_dominating_set(); // Apply Wiedemann-Ford algorithm
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)
    println!("\n[Wiedemann-Ford: Dominating Set]");
    println!("Dominating set: {:?}", dominating_set);
}

<<<<<<< HEAD
/// Function to perform Leiden clustering on the graph.
///
/// # Arguments
/// * `graph` - A reference to the `Graph<WeightType, (), ()>` object.
fn perform_clustering(graph: &Graph<WeightType, (), ()>) {
    let resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5];
    let gammas = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
    const LEIDEN_ITERATIONS: usize = 10;
=======
/// Performs Leiden clustering on the graph and prints the results.
///
/// Runs clustering experiments with varying resolution and gamma parameters,
/// comparing deterministic and non-deterministic outcomes.
///
/// # Arguments
/// * `graph` - A reference to the `Graph<WeightType, (), ()>` to cluster.
fn perform_clustering(graph: &Graph<WeightType, (), ()>) {
    let resolutions = [0.2]; // Single resolution for simplicity
    let gammas = [0.2]; // Single gamma for simplicity
    const LEIDEN_ITERATIONS: usize = 50; // Maximum iterations for clustering
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)

    println!("\n================ Leiden Clustering Experiments ================");

    for (i, &resolution) in resolutions.iter().enumerate() {
        for (j, &gamma) in gammas.iter().enumerate() {
<<<<<<< HEAD
            let config = CommunityConfig {
                gamma,
                resolution,
                iterations: LEIDEN_ITERATIONS,
            };

            let communities = graph.detect_communities_with_config(config);

            println!("\nExperiment #{}-{}", i + 1, j + 1);
=======
            let fixed_seed = 42;

            // Deterministic clustering configuration
            let config_det = CommunityConfig {
                gamma,
                resolution,
                iterations: LEIDEN_ITERATIONS,
                deterministic: true,
                seed: None,
            };

            let communities_det = graph.detect_communities_with_config(config_det.clone());

            println!("\nExperiment #{}-{} (Deterministic)", i + 1, j + 1);
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)
            println!(
                "Parameters: γ = {:.1}, resolution = {:.1}",
                gamma, resolution
            );
<<<<<<< HEAD
            println!("Found {} communities:", communities.len());

            communities
=======
            println!("Found {} communities:", communities_det.len());
            communities_det
                .iter()
                .enumerate()
                .for_each(|(idx, (comm_id, nodes))| {
                    println!(
                        "  Cluster {} (ID: {}): {} nodes : {:?}",
                        idx + 1,
                        comm_id,
                        nodes.len(),
                        nodes
                    );
                });

            // Non-deterministic clustering configuration
            let config_non_det = CommunityConfig {
                deterministic: false,
                seed: None,
                ..config_det
            };

            let communities_non_det = graph.detect_communities_with_config(config_non_det);

            println!(
                "\nExperiment #{}-{} (Non-Deterministic with seed {})",
                i + 1,
                j + 1,
                fixed_seed
            );
            println!(
                "Parameters: γ = {:.1}, resolution = {:.1}",
                gamma, resolution
            );
            println!("Found {} communities:", communities_non_det.len());
            communities_non_det
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)
                .iter()
                .enumerate()
                .for_each(|(idx, (comm_id, nodes))| {
                    println!(
                        "  Cluster {} (ID: {}): {} nodes : {:?}",
                        idx + 1,
                        comm_id,
                        nodes.len(),
                        nodes
                    );
                });
        }
    }
}

<<<<<<< HEAD
fn main() {
=======
/// Main entry point for the graph analysis demonstration.
///
/// Creates a sample graph from an adjacency matrix, adds attributes,
/// and runs various analyses and clustering experiments.
fn main() {
    // Define a sample adjacency matrix representing an undirected graph
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)
    let matrix = vec![
        vec![0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        vec![1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        vec![1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
        vec![0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        vec![0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        vec![0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        vec![0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
        vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    ];

<<<<<<< HEAD
    let mut graph = create_graph_from_matrix(matrix, true);

=======
    // Create and initialize the graph
    let mut graph = create_graph_from_matrix(matrix, false);

    // Add sample attributes to nodes and edges
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)
    let _ = graph.set_node_attribute(1, "color".into(), "red".into());
    let _ = graph.set_node_attribute(2, "color".into(), "blue".into());
    let _ = graph.set_edge_attribute(1, 2, "type".into(), "road".into());
    let _ = graph.set_edge_attribute(2, 3, "type".into(), "rail".into());

<<<<<<< HEAD
=======
    // Run the analysis and clustering
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)
    print_graph_details(&graph);
    analyze_graph(&mut graph);
    perform_clustering(&graph);
    demonstrate_io(&mut graph);
<<<<<<< HEAD
}
=======
}
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)
