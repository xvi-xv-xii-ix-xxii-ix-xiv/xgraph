use std::collections::HashMap;
use xgraph::algorithms::connectivity::Connectivity;
use xgraph::algorithms::leiden_clustering::{CommunityConfig, CommunityDetection};
use xgraph::algorithms::wiedemann_ford::DominatingSetFinder;
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
fn create_graph_from_matrix(
    matrix: Vec<Vec<WeightType>>,
    directed: bool,
) -> Graph<WeightType, (), ()> {
    Graph::from_adjacency_matrix(&matrix, directed, (), ())
        .expect("Failed to create graph from matrix")
}

/// Function to print details of the graph nodes and edges.
///
/// # Arguments
/// * `graph` - A reference to the `Graph<WeightType, (), ()>` object.
fn print_graph_details(graph: &Graph<WeightType, (), ()>) {
    let nodes: Vec<(usize, &())> = graph.all_nodes().collect(); // Explicitly specify the type of vertices
    let edges = graph.get_all_edges();

    println!("\n================== Graph Details ==================");
    println!(
        "Nodes ({}): {:?}",
        nodes.len(),
        nodes.iter().map(|(id, _)| id).collect::<Vec<_>>()
    );
    println!("Edges ({}): {:?}", edges.len(), edges);
}

/// Function to demonstrate IO.
fn demonstrate_io(graph: &mut Graph<WeightType, (), ()>) {
    println!("\n================== IO and Format Demonstration ==================");

    let string_graph = graph.to_string_graph();

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

/// Function to analyze the graph and print the results.
///
/// # Arguments
/// * `graph` - A mutable reference to the `Graph<WeightType, (), ()>` object.
fn analyze_graph(graph: &mut Graph<WeightType, (), ()>) {
    println!("\n================== Graph Analysis ==================");

    // 1. Basic metrics of the graph
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
fn print_metrics(nodes: usize, edges: usize, directed: bool) {
    println!("\n[Basic Metrics]");
    println!("Number of nodes: {}", nodes);
    println!("Number of edges: {}", edges);
    println!(
        "Type of graph: {}",
        if directed { "directed" } else { "undirected" }
    );
}

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
    println!("Degree centrality:");
    centrality
        .iter()
        .for_each(|(node, val)| println!("  Node {}: {:.2}", node, val));
}

/// Function to print attributes of nodes and edges.
///
/// # Arguments
/// * `graph` - A mutable reference to the `Graph<WeightType, (), ()>` object.
fn print_attributes(graph: &mut Graph<WeightType, (), ()>) {
    // Node attributes (unchanged)
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

    // Edge attributes (fixed version)
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
fn print_connected_components(graph: &mut Graph<WeightType, (), ()>) {
    println!("\n[Connected Components]");

    let components = if graph.is_directed() {
        // For directed graphs, show both types of components
        println!("Strongly connected components:");
        let scc = graph.find_strongly_connected_components();
        println!("  Count: {}", scc.len());

        println!("Weakly connected components:");
        let wcc = graph.find_weakly_connected_components();
        println!("  Count: {}", wcc.len());
        wcc
    } else {
        // For undirected graphs, regular components
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

/// Function to print density of the graph.
///
/// # Arguments
/// * `nodes` - Number of nodes in the graph.
/// * `edges` - Number of edges in the graph.
/// * `directed` - Indicates if the graph is directed.
fn print_density(nodes: usize, edges: usize, directed: bool) {
    println!("\n[Density]");
    let density = calculate_density(nodes, edges, directed);
    println!("Density: {:.4}\n", density);
}

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
    };
    edges as f64 / possible_edges as f64
}

/// Function to print results of the Wiedemann-Ford algorithm on the graph.
///
/// # Arguments
/// * `graph` - A mutable reference to the `Graph<WeightType, (), ()>` object.
fn print_wiedemann_ford(graph: &mut Graph<WeightType, (), ()>) {
    let dominating_set = graph.find_dominating_set();
    println!("\n[Wiedemann-Ford: Dominating Set]");
    println!("Dominating set: {:?}", dominating_set);
}

/// Function to perform Leiden clustering on the graph.
///
/// # Arguments
/// * `graph` - A reference to the `Graph<WeightType, (), ()>` object.
fn perform_clustering(graph: &Graph<WeightType, (), ()>) {
    let resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5];
    let gammas = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2];
    const LEIDEN_ITERATIONS: usize = 10;

    println!("\n================ Leiden Clustering Experiments ================");

    for (i, &resolution) in resolutions.iter().enumerate() {
        for (j, &gamma) in gammas.iter().enumerate() {
            let config = CommunityConfig {
                gamma,
                resolution,
                iterations: LEIDEN_ITERATIONS,
            };

            let communities = graph.detect_communities_with_config(config);

            println!("\nExperiment #{}-{}", i + 1, j + 1);
            println!(
                "Parameters: γ = {:.1}, resolution = {:.1}",
                gamma, resolution
            );
            println!("Found {} communities:", communities.len());

            communities
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

fn main() {
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

    let mut graph = create_graph_from_matrix(matrix, true);

    let _ = graph.set_node_attribute(1, "color".into(), "red".into());
    let _ = graph.set_node_attribute(2, "color".into(), "blue".into());
    let _ = graph.set_edge_attribute(1, 2, "type".into(), "road".into());
    let _ = graph.set_edge_attribute(2, 3, "type".into(), "rail".into());

    print_graph_details(&graph);
    analyze_graph(&mut graph);
    perform_clustering(&graph);
    demonstrate_io(&mut graph);
}
