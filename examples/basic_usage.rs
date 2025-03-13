//! Graph analysis demonstration program
//!
//! This module provides a comprehensive demonstration of graph analysis techniques, showcasing
//! various structural and algorithmic operations on graphs. It includes robust error handling
//! to ensure reliable execution and serves as an example for users of the `xgraph` library.
//!
//! # Features
//! - Basic graph metrics (node and edge counts, graph type)
//! - Connectivity analysis (path finding, connected components)
//! - Centrality measures (degree centrality)
//! - Bridge detection
//! - Community detection using the Leiden algorithm
//! - Input/output operations with CSV files
//!
//! # Examples
//!
//! Running the full analysis:
//! ```rust
//! fn main() -> Result<(), GraphAnalysisError> {
//!     let matrix = vec![
//!         vec![0, 1, 1, 0],
//!         vec![1, 0, 0, 1],
//!         vec![1, 0, 0, 1],
//!         vec![0, 1, 1, 0],
//!     ];
//!     let mut graph = create_graph_from_matrix(matrix, false)?;
//!     print_graph_details(&graph);
//!     analyze_graph(&mut graph)?;
//!     perform_clustering(&graph)?;
//!     demonstrate_io(&mut graph)?;
//!     Ok(())
//! }
//! ```

use std::collections::HashMap;
use xgraph::bridges::Bridges;
use xgraph::centrality::Centrality;
use xgraph::graph::algorithms::connectivity::Connectivity;
use xgraph::graph::algorithms::leiden_clustering::{CommunityConfig, CommunityDetection};
use xgraph::graph::algorithms::wiedemann_ford::DominatingSetFinder;
use xgraph::graph::conversion::GraphConversion;
use xgraph::graph::CsvIO;
use xgraph::search::Search;
use xgraph::Graph;

/// Type alias for edge weights used throughout the graph analysis.
///
/// Defines the weight type as `u32` for consistency across the demonstration.
type WeightType = u32;

/// Error type for graph analysis operations.
///
/// Encapsulates various errors that may occur during graph analysis, providing detailed feedback
/// to the library user.
#[derive(Debug)]
pub enum GraphAnalysisError {
    /// Indicates a failure in bridge detection.
    BridgeError(xgraph::bridges::BridgeError),
    /// Indicates a failure in centrality computation.
    CentralityError(xgraph::centrality::CentralityError),
    /// Indicates a failure in connectivity analysis.
    ConnectivityError(xgraph::connectivity::ConnectivityError),
    /// Indicates a failure in IO operations.
    IOError(std::io::Error),
    /// Indicates a failure in community detection.
    CommunityDetectionError(xgraph::graph::algorithms::leiden_clustering::CommunityDetectionError),
    /// Indicates a failure in search operations (e.g., path finding).
    SearchError(xgraph::search::SearchError),
    /// Indicates a failure in graph conversion (e.g., during IO operations).
    ConversionError(xgraph::graph::conversion::graph_conversion::GraphConversionError),
    /// Indicates a failure in dominating set computation.
    DominatingSetError(xgraph::graph::algorithms::wiedemann_ford::DominatingSetError),
}

impl std::fmt::Display for GraphAnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphAnalysisError::BridgeError(e) => write!(f, "Bridge computation error: {}", e),
            GraphAnalysisError::CentralityError(e) => {
                write!(f, "Centrality computation error: {}", e)
            }
            GraphAnalysisError::ConnectivityError(e) => {
                write!(f, "Connectivity analysis error: {}", e)
            }
            GraphAnalysisError::IOError(e) => write!(f, "IO error: {}", e),
            GraphAnalysisError::CommunityDetectionError(e) => {
                write!(f, "Community detection error: {}", e)
            }
            GraphAnalysisError::SearchError(e) => write!(f, "Search error: {}", e),
            GraphAnalysisError::ConversionError(e) => write!(f, "Graph conversion error: {}", e),
            GraphAnalysisError::DominatingSetError(e) => {
                write!(f, "Dominating set computation error: {}", e)
            }
        }
    }
}

impl std::error::Error for GraphAnalysisError {}

/// Creates a graph from an adjacency matrix representation.
///
/// Constructs a graph where non-zero values in the matrix represent edges with weights.
/// Uses `WeightType` for edge weights and empty tuples for node and edge data.
///
/// # Arguments
/// - `matrix`: A 2D vector where `matrix[i][j]` is the weight of an edge from node `i` to node `j`. A value of 0 indicates no edge.
/// - `directed`: A boolean indicating whether the graph is directed (`true`) or undirected (`false`).
///
/// # Returns
/// - `Ok(Graph<WeightType, (), ()>)`: The constructed graph on success.
/// - `Err(GraphAnalysisError)`: If the matrix is invalid (e.g., non-square or malformed).
fn create_graph_from_matrix(
    matrix: Vec<Vec<WeightType>>,
    directed: bool,
) -> Result<Graph<WeightType, (), ()>, GraphAnalysisError> {
    Graph::from_adjacency_matrix(&matrix, directed, (), ()).map_err(|e| {
        GraphAnalysisError::IOError(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    })
}

/// Prints detailed information about the graph's nodes and edges.
///
/// Displays a summary of the graph structure, including node and edge counts and their details.
///
/// # Arguments
/// - `graph`: A reference to the graph to analyze.
fn print_graph_details(graph: &Graph<WeightType, (), ()>) {
    let nodes: Vec<(usize, &())> = graph.all_nodes().collect();
    let edges = graph.get_all_edges();

    println!("\n================== Graph Details ==================");
    println!(
        "Nodes ({}): {:?}",
        nodes.len(),
        nodes.iter().map(|(id, _)| id).collect::<Vec<_>>()
    );
    println!("Edges ({}): {:?}", edges.len(), edges);
}

/// Demonstrates input/output operations with the graph.
///
/// Saves the graph to CSV files and loads it back, displaying the results.
///
/// # Arguments
/// - `graph`: A mutable reference to the graph to process.
///
/// # Returns
/// - `Ok(())`: On successful completion of IO operations.
/// - `Err(GraphAnalysisError)`: If conversion, saving, or loading the graph fails.
fn demonstrate_io(graph: &mut Graph<WeightType, (), ()>) -> Result<(), GraphAnalysisError> {
    println!("\n================== IO and Format Demonstration ==================");
    let string_graph = graph
        .to_string_graph()
        .map_err(GraphAnalysisError::ConversionError)?;

    println!("\n[Saving to CSV]");
    string_graph
        .save_to_csv("nodes.csv", "edges.csv")
        .map_err(GraphAnalysisError::IOError)?;
    println!("Graph saved to nodes.csv and edges.csv");

    println!("\n[Loading from CSV]");
    let loaded_graph =
        Graph::<WeightType, String, String>::load_from_csv("nodes.csv", "edges.csv", true)
            .map_err(GraphAnalysisError::IOError)?;
    println!(
        "Loaded graph: {} nodes, {} edges",
        loaded_graph.nodes.len(),
        loaded_graph.edges.len()
    );
    Ok(())
}

/// Analyzes the graph and prints various structural properties.
///
/// Performs a comprehensive analysis including metrics, connectivity, centrality, attributes,
/// bridges, components, density, and dominating sets.
///
/// # Arguments
/// - `graph`: A mutable reference to the graph to analyze.
///
/// # Returns
/// - `Ok(())`: On successful analysis.
/// - `Err(GraphAnalysisError)`: If any analysis step fails.
fn analyze_graph(graph: &mut Graph<WeightType, (), ()>) -> Result<(), GraphAnalysisError> {
    println!("\n================== Graph Analysis ==================");

    let num_nodes = graph.nodes.len();
    let num_edges = graph.get_all_edges().len();
    print_metrics(num_nodes, num_edges, graph.directed);
    print_connectivity(graph, num_nodes)?;
    print_centrality(graph)?;
    print_attributes(graph);
    print_bridges(graph)?;
    print_connected_components(graph)?;
    print_density(num_nodes, num_edges, graph.directed);
    print_wiedemann_ford(graph)?;
    Ok(())
}

/// Prints basic metrics about the graph structure.
///
/// Displays the number of nodes, edges, and whether the graph is directed.
///
/// # Arguments
/// - `nodes`: The number of nodes in the graph.
/// - `edges`: The number of edges in the graph.
/// - `directed`: Whether the graph is directed (`true`) or undirected (`false`).
fn print_metrics(nodes: usize, edges: usize, directed: bool) {
    println!("\n[Basic Metrics]");
    println!("Number of nodes: {}", nodes);
    println!("Number of edges: {}", edges);
    println!(
        "Type of graph: {}",
        if directed { "directed" } else { "undirected" }
    );
}

/// Prints connectivity information and example paths.
///
/// Demonstrates path existence and shortest path finding if sufficient nodes exist.
///
/// # Arguments
/// - `graph`: A mutable reference to the graph.
/// - `node_count`: The total number of nodes in the graph.
///
/// # Returns
/// - `Ok(())`: On successful connectivity analysis.
/// - `Err(GraphAnalysisError)`: If connectivity checks fail (e.g., invalid nodes).
fn print_connectivity(
    graph: &mut Graph<WeightType, (), ()>,
    node_count: usize,
) -> Result<(), GraphAnalysisError> {
    println!("\n[Connectivity and Paths]");
    if node_count >= 6 {
        let has_path = graph
            .has_path(0, 5)
            .map_err(GraphAnalysisError::SearchError)?;
        println!("Path from 0 to 5 exists: {}", has_path);
        let bfs_path = graph
            .bfs_path(0, 5)
            .map_err(GraphAnalysisError::SearchError)?;
        println!("Shortest path 0->5: {:?}", bfs_path);
    } else {
        println!("Not enough nodes for path example (need at least 6)");
    }
    Ok(())
}

/// Prints centrality metrics for the graph's nodes.
///
/// Computes and displays degree centrality for each node.
///
/// # Arguments
/// - `graph`: A mutable reference to the graph.
///
/// # Returns
/// - `Ok(())`: On successful centrality computation.
/// - `Err(GraphAnalysisError)`: If centrality computation fails.
fn print_centrality(graph: &mut Graph<WeightType, (), ()>) -> Result<(), GraphAnalysisError> {
    println!("\n[Centrality]");
    let centrality = graph
        .degree_centrality()
        .map_err(GraphAnalysisError::CentralityError)?;
    println!("Degree centrality:");
    centrality
        .iter()
        .for_each(|(node, val)| println!("  Node {}: {}", node, val));
    Ok(())
}

/// Prints node and edge attributes present in the graph.
///
/// Aggregates and displays all attributes by key and value.
///
/// # Arguments
/// - `graph`: A mutable reference to the graph.
fn print_attributes(graph: &mut Graph<WeightType, (), ()>) {
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

/// Prints bridges in the graph.
///
/// Identifies and displays edges whose removal would disconnect the graph.
///
/// # Arguments
/// - `graph`: A mutable reference to the graph.
///
/// # Returns
/// - `Ok(())`: On successful bridge detection.
/// - `Err(GraphAnalysisError)`: If bridge detection fails.
fn print_bridges(graph: &mut Graph<WeightType, (), ()>) -> Result<(), GraphAnalysisError> {
    println!("\n[Bridges]");
    let bridges = graph
        .find_bridges()
        .map_err(GraphAnalysisError::BridgeError)?;
    println!("Bridges (critical edges): {:?}", bridges);
    Ok(())
}

/// Prints information about connected components.
///
/// Displays strongly/weakly connected components for directed graphs or regular components
/// for undirected graphs.
///
/// # Arguments
/// - `graph`: A mutable reference to the graph.
///
/// # Returns
/// - `Ok(())`: On successful component analysis.
/// - `Err(GraphAnalysisError)`: If component analysis fails.
fn print_connected_components(
    graph: &mut Graph<WeightType, (), ()>,
) -> Result<(), GraphAnalysisError> {
    println!("\n[Connected Components]");

    let components = if graph
        .is_directed()
        .map_err(GraphAnalysisError::ConnectivityError)?
    {
        println!("Strongly connected components:");
        let scc = graph
            .find_strongly_connected_components()
            .map_err(GraphAnalysisError::ConnectivityError)?;
        println!("  Count: {}", scc.len());

        println!("Weakly connected components:");
        let wcc = graph
            .find_weakly_connected_components()
            .map_err(GraphAnalysisError::ConnectivityError)?;
        println!("  Count: {}", wcc.len());
        wcc
    } else {
        graph
            .find_connected_components()
            .map_err(GraphAnalysisError::ConnectivityError)?
    };

    components
        .iter()
        .enumerate()
        .for_each(|(i, c)| println!("  Component {}: {} nodes", i, c.len()));

    let is_connected = graph
        .is_connected()
        .map_err(GraphAnalysisError::ConnectivityError)?;
    println!("Overall connectivity: {}", is_connected);

    println!("\nAdditional checks:");
    let is_weakly = graph
        .is_weakly_connected()
        .map_err(GraphAnalysisError::ConnectivityError)?;
    println!("Weakly connected: {}", is_weakly);

    if graph
        .is_directed()
        .map_err(GraphAnalysisError::ConnectivityError)?
    {
        let is_strongly = graph
            .is_strongly_connected()
            .map_err(GraphAnalysisError::ConnectivityError)?;
        println!("Strongly connected: {}", is_strongly);
    }
    Ok(())
}

/// Prints the density of the graph.
///
/// Displays the ratio of actual edges to the maximum possible edges.
///
/// # Arguments
/// - `nodes`: The number of nodes in the graph.
/// - `edges`: The number of edges in the graph.
/// - `directed`: Whether the graph is directed (`true`) or undirected (`false`).
fn print_density(nodes: usize, edges: usize, directed: bool) {
    println!("\n[Density]");
    let density = calculate_density(nodes, edges, directed);
    println!("Density: {:.4}\n", density);
}

/// Calculates the density of the graph.
///
/// Computes the ratio of existing edges to the maximum possible edges.
///
/// # Arguments
/// - `nodes`: The number of nodes in the graph.
/// - `edges`: The number of edges in the graph.
/// - `directed`: Whether the graph is directed (`true`) or undirected (`false`).
///
/// # Returns
/// - `f64`: The density value, ranging from 0.0 (no edges) to 1.0 (fully connected).
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

/// Prints the dominating set using the Wiedemann-Ford algorithm.
///
/// Identifies and displays nodes that dominate all others in the graph.
///
/// # Arguments
/// - `graph`: A mutable reference to the graph.
///
/// # Returns
/// - `Ok(())`: On successful computation and display of the dominating set.
/// - `Err(GraphAnalysisError)`: If dominating set computation fails.
fn print_wiedemann_ford(graph: &mut Graph<WeightType, (), ()>) -> Result<(), GraphAnalysisError> {
    let dominating_set = graph
        .find_dominating_set()
        .map_err(GraphAnalysisError::DominatingSetError)?;
    println!("\n[Wiedemann-Ford: Dominating Set]");
    println!("Dominating set: {:?}", dominating_set);
    Ok(())
}

/// Performs Leiden clustering and prints results.
///
/// Runs the Leiden clustering algorithm with varying parameters and displays the results.
///
/// # Arguments
/// - `graph`: A reference to the graph to cluster.
///
/// # Returns
/// - `Ok(())`: On successful clustering.
/// - `Err(GraphAnalysisError)`: If clustering fails (e.g., due to invalid parameters).
fn perform_clustering(graph: &Graph<WeightType, (), ()>) -> Result<(), GraphAnalysisError> {
    let resolutions = [0.2];
    let gammas = [0.2];
    const LEIDEN_ITERATIONS: usize = 50;

    println!("\n================ Leiden Clustering Experiments ================");

    for (i, &resolution) in resolutions.iter().enumerate() {
        for (j, &gamma) in gammas.iter().enumerate() {
            let fixed_seed = 42;

            let config_det = CommunityConfig {
                gamma,
                resolution,
                iterations: LEIDEN_ITERATIONS,
                deterministic: true,
                seed: None,
            };

            let communities_det = graph
                .detect_communities_with_config(config_det.clone())
                .map_err(GraphAnalysisError::CommunityDetectionError)?;
            println!("\nExperiment #{}-{} (Deterministic)", i + 1, j + 1);
            println!(
                "Parameters: γ = {:.1}, resolution = {:.1}",
                gamma, resolution
            );
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

            let config_non_det = CommunityConfig {
                deterministic: false,
                seed: None,
                ..config_det
            };

            let communities_non_det = graph
                .detect_communities_with_config(config_non_det)
                .map_err(GraphAnalysisError::CommunityDetectionError)?;
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
    Ok(())
}

/// Main entry point for the graph analysis demonstration.
///
/// Creates a sample graph, adds attributes, and runs a full suite of analyses.
///
/// # Returns
/// - `Ok(())`: On successful execution of all operations.
/// - `Err(GraphAnalysisError)`: If any operation fails.
fn main() -> Result<(), GraphAnalysisError> {
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

    let mut graph = create_graph_from_matrix(matrix, false)?;

    let _ = graph.set_node_attribute(1, "color".into(), "red".into());
    let _ = graph.set_node_attribute(2, "color".into(), "blue".into());
    let _ = graph.set_edge_attribute(1, 2, "type".into(), "road".into());
    let _ = graph.set_edge_attribute(2, 3, "type".into(), "rail".into());

    print_graph_details(&graph);
    analyze_graph(&mut graph)?;
    perform_clustering(&graph)?;
    demonstrate_io(&mut graph)?;
    Ok(())
}
