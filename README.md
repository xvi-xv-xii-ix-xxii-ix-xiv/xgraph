```
 ____   ____._____  .______  .______  ._______ .___.__  
 \   \_/   /:_ ___\ : __   \ :      \ : ____  |:   |  \ 
  \___ ___/ |   |___|  \____||   .   ||    :  ||   :   |
  /   _   \ |   /  ||   :  \ |   :   ||   |___||   .   |
 /___/ \___\|. __  ||   |___\|___|   ||___|    |___|   |
             :/ |. ||___|        |___|             |___|
             :   :/                                     
                 :                                      
                                                                                                               
```
[![Crates.io](https://img.shields.io/crates/v/xgraph.svg)](https://crates.io/crates/xgraph)
![Rust](https://img.shields.io/badge/Rust-1.70+-orange)
![License](https://img.shields.io/badge/License-MIT-blue)

# XGraph is a comprehensive Rust library providing efficient graph algorithms for solving real-world problems in social network analysis, transportation optimization, recommendation systems, and more.

## 🌟 Why XGraph?

- **High Performance**: Optimized algorithms for fast computation.
- **Flexible Data Model**: Store custom attributes for nodes and edges.
- **Practical Algorithms**: Covering connectivity, shortest paths, community detection, and more.
- **Easy Integration**: Simple API with serialization support, including CSV input/output (since v1.1.0).

## 💼 Applications

- **Social Network Analysis**: Detect key influencers and communities.
- **Logistics & Routing**: Find shortest and most reliable paths.
- **Telecommunication Networks**: Identify critical nodes and links.
- **Recommendation Systems**: Analyze user-item interaction graphs.

A comprehensive graph theory library implementing essential algorithms with full type flexibility and performance.

## Features 🌟

### Flexible Graph Structure
- Directed/Undirected graphs
- Custom node/edge data types
- Weighted edges
- Arbitrary attributes

### Core Algorithms

- **Bridge detection**: Identifies critical edges whose removal would disconnect the graph.
- **Centrality measures**: Includes Degree, Betweenness, and Closeness centrality to evaluate node importance.
- **Connectivity analysis**: Assesses graph connectivity, including strongly and weakly connected components.
- **Leiden community detection**: Detects communities in the graph with both deterministic and non-deterministic variants, allowing for reproducible results or randomized exploration based on configuration.
- **Shortest paths (Dijkstra)**: Computes the shortest paths between nodes using Dijkstra's algorithm.
- **Dominating set finding**: Identifies a minimal set of nodes that dominate all others in the graph.
- **Cycle detection**: Detects cycles within the graph structure.

### Advanced Operations
- Adjacency matrix conversion
- Graph validation
- Batch operations
- Attribute management
- Graph transposition
- CSV serialization (since v1.1.0)

## Quick Start 🚀

### Basic Usage
```rust
use xgraph::graph::Graph;

// Create undirected graph with String nodes and tuple edges
let mut graph = Graph::<f64, String, (f64, String)>::new(false);

// Add nodes with data
let london = graph.add_node("London".into());
let paris = graph.add_node("Paris".into());

// Add weighted edge with metadata
graph.add_edge(london, paris, 343.0, (343.0, "Eurostar".into())).unwrap();
```

### Shortest Paths
```rust
use xgraph::algorithms::ShortestPath;

let distances = graph.dijkstra(london);
println!("Paris distance: {}", distances[&paris]); // 343.0
```

### Community Detection
```rust
use xgraph::algorithms::leiden_clustering::Leiden;

let mut leiden = Leiden::new(adjacency_matrix, 0.5);
leiden.run();
println!("Communities: {:?}", leiden.get_communities());
```

## Core API Documentation 📚

### Graph Structure
```rust
pub struct Graph<W: Copy + PartialEq, N: Eq + Hash, E: Debug> {
    pub nodes: Slab<Node<W, N>>,  // Node storage
    pub edges: Slab<Edge<W, E>>,  // Edge storage 
    pub directed: bool,           // Graph directionality
}
```

### Node Structure
```rust
pub struct Node<W, N> {
    pub data: N,
    pub neighbors: Vec<(NodeId, W)>,
    pub attributes: HashMap<String, String>,
}
```

### Edge Structure
```rust
pub struct Edge<W, E> {
    pub from: NodeId,
    pub to: NodeId,
    pub weight: W,
    pub data: E,
    pub attributes: HashMap<String, String>,
}
```

### Algorithm Traits

**Bridges Detection**
```rust
impl Bridges for Graph {...}
let bridges = graph.find_bridges();
```

**Centrality Measures**
```rust
impl Centrality for Graph {...}
let betweenness = graph.betweenness_centrality();
```

**Connectivity Analysis**
```rust
impl Connectivity for Graph {...}
if graph.is_strongly_connected() { ... }
```

## Advanced Features 🔧

### Attribute Management
```rust
// Node attributes
graph.set_node_attribute(0, "population".into(), "8_982_000".into());

// Edge attributes
graph.set_edge_attribute(0, 1, "transport".into(), "rail".into());

// Retrieval
let pop = graph.get_node_attribute(0, "population");
```

### Matrix Conversions
```rust
// To adjacency matrix
let matrix = graph.to_adjacency_matrix();

// From matrix with default values
let graph = Graph::from_adjacency_matrix(
    &matrix, 
    true, 
    "Station".into(), 
    ("default".into(), 0.0)
).unwrap();
```

## Real-World Examples 🌍

### Social Network Analysis
```rust
let mut social_graph = Graph::new(false);

// Batch add users
let users = social_graph.add_nodes_batch(
    vec!["Alice", "Bob", "Charlie", "Diana"].into_iter()
);

// Create connections
social_graph.add_edges_batch(vec![
    (0, 1, 1, ("friends", 2)), 
    (1, 2, 1, ("colleagues", 5)),
    (2, 3, 1, ("family", 10))
]).unwrap();

// Analyze influence
let centrality = social_graph.degree_centrality();
let bridges = social_graph.find_bridges();
```

### Transportation Network Optimization
```rust
// Find critical connections
let transport_bridges = transport_graph.find_bridges();

// Calculate optimal depot routes
let depot_distances = transport_graph.dijkstra(main_depot);

// Cluster service regions
let mut leiden = Leiden::new(transport_weights, 0.75);
leiden.run();
let service_regions = leiden.get_communities();
```

## Full example of basic usage:
```rust
use std::collections::HashMap;
use xgraph::algorithms::connectivity::Connectivity;
use xgraph::algorithms::leiden_clustering::{CommunityConfig, CommunityDetection};
use xgraph::algorithms::wiedemann_ford::DominatingSetFinder;
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
fn create_graph_from_matrix(
    matrix: Vec<Vec<WeightType>>,
    directed: bool,
) -> Graph<WeightType, (), ()> {
    Graph::from_adjacency_matrix(&matrix, directed, (), ())
        .expect("Failed to create graph from matrix")
}

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

    println!("\n================== Graph Details ==================");
    println!(
        "Nodes ({}): {:?}",
        nodes.len(),
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
fn print_metrics(nodes: usize, edges: usize, directed: bool) {
    println!("\n[Basic Metrics]");
    println!("Number of nodes: {}", nodes);
    println!("Number of edges: {}", edges);
    println!(
        "Type of graph: {}",
        if directed { "directed" } else { "undirected" }
    );
}

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
    println!("Degree centrality:");
    centrality
        .iter()
        .for_each(|(node, val)| println!("  Node {}: {:.2}", node, val));
}

/// Prints node and edge attributes present in the graph.
///
/// Aggregates and displays all attributes, grouping them by key and value.
///
/// # Arguments
/// * `graph` - A mutable reference to the `Graph<WeightType, (), ()>` to analyze.
fn print_attributes(graph: &mut Graph<WeightType, (), ()>) {
    // Aggregate node attributes
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

    // Aggregate edge attributes
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
fn print_connected_components(graph: &mut Graph<WeightType, (), ()>) {
    println!("\n[Connected Components]");

    let components = if graph.is_directed() {
        // Directed graph: show both strong and weak components
        println!("Strongly connected components:");
        let scc = graph.find_strongly_connected_components();
        println!("  Count: {}", scc.len());

        println!("Weakly connected components:");
        let wcc = graph.find_weakly_connected_components();
        println!("  Count: {}", wcc.len());
        wcc
    } else {
        // Undirected graph: regular components
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

/// Prints the density of the graph.
///
/// Calculates and displays the graph's density as the ratio of actual to possible edges.
///
/// # Arguments
/// * `nodes` - The number of nodes in the graph.
/// * `edges` - The number of edges in the graph.
/// * `directed` - A boolean indicating if the graph is directed.
fn print_density(nodes: usize, edges: usize, directed: bool) {
    println!("\n[Density]");
    let density = calculate_density(nodes, edges, directed);
    println!("Density: {:.4}\n", density);
}

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
    };
    edges as f64 / possible_edges as f64
}

/// Prints the dominating set found using the Wiedemann-Ford algorithm.
///
/// Identifies a minimal set of nodes that dominate all others in the graph.
///
/// # Arguments
/// * `graph` - A mutable reference to the `Graph<WeightType, (), ()>` to analyze.
fn print_wiedemann_ford(graph: &mut Graph<WeightType, (), ()>) {
    let dominating_set = graph.find_dominating_set(); // Apply Wiedemann-Ford algorithm
    println!("\n[Wiedemann-Ford: Dominating Set]");
    println!("Dominating set: {:?}", dominating_set);
}

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

    println!("\n================ Leiden Clustering Experiments ================");

    for (i, &resolution) in resolutions.iter().enumerate() {
        for (j, &gamma) in gammas.iter().enumerate() {
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

/// Main entry point for the graph analysis demonstration.
///
/// Creates a sample graph from an adjacency matrix, adds attributes,
/// and runs various analyses and clustering experiments.
fn main() {
    // Define a sample adjacency matrix representing an undirected graph
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

    // Create and initialize the graph
    let mut graph = create_graph_from_matrix(matrix, false);

    // Add sample attributes to nodes and edges
    let _ = graph.set_node_attribute(1, "color".into(), "red".into());
    let _ = graph.set_node_attribute(2, "color".into(), "blue".into());
    let _ = graph.set_edge_attribute(1, 2, "type".into(), "road".into());
    let _ = graph.set_edge_attribute(2, 3, "type".into(), "rail".into());

    // Run the analysis and clustering
    print_graph_details(&graph);
    analyze_graph(&mut graph);
    perform_clustering(&graph);
    demonstrate_io(&mut graph);
}


```

## Testing & Validation ✅

Run comprehensive tests:
```bash
cargo test

test algorithms::bridges::tests::test_find_bridges_empty_graph ... ok
test algorithms::centrality::tests::test_closeness_centrality ... ok
test algorithms::centrality::tests::test_degree_centrality ... ok
test algorithms::bridges::tests::test_find_bridges_single_node ... ok
test algorithms::bridges::tests::test_find_bridges_simple ... ok
test algorithms::bridges::tests::test_find_bridges_no_bridges ... ok
test algorithms::centrality::tests::test_betweenness_centrality ... ok
test algorithms::bridges::tests::test_find_bridges_complex ... ok
test algorithms::centrality::tests::test_empty_graph ... ok
test algorithms::connectivity::tests::test_strongly_connected ... ok
test algorithms::connectivity::tests::test_transpose ... ok
test algorithms::connectivity::tests::test_weak_connectivity ... ok
test algorithms::leiden_clustering::tests::test_empty_graph ... ok
test algorithms::leiden_clustering::tests::test_single_node ... ok
test algorithms::leiden_clustering::tests::test_modularity_calculation ... ok
test algorithms::search::tests::test_bfs_path ... ok
test algorithms::leiden_clustering::tests::test_community_detection_basic ... ok
test algorithms::search::tests::test_cycle_detection_directed ... ok
test algorithms::search::tests::test_cycle_detection_undirected ... ok
test algorithms::leiden_clustering::tests::test_deterministic_behavior ... ok
test algorithms::search::tests::test_dfs ... ok
test algorithms::search::tests::test_has_path ... ok
test algorithms::leiden_clustering::tests::test_directed_graph ... ok
test algorithms::search::tests::test_invalid_nodes ... ok
test algorithms::search::tests::test_no_cycle_directed ... ok
test algorithms::search::tests::test_no_cycle_undirected ... ok
test algorithms::shortest_path::tests::test_dijkstra_basic ... ok
test algorithms::shortest_path::tests::test_unreachable_node ... ok
test algorithms::wiedemann_ford::tests::test_complex_dominating_set ... ok
test algorithms::wiedemann_ford::tests::test_simple_dominating_set ... ok
test graph::graph::tests::test_attributes ... ok
test graph::graph::tests::test_complete_graph ... ok
test graph::graph::tests::test_directed_graph ... ok
test graph::graph::tests::test_disconnected_graph ... ok
test graph::graph::tests::test_empty_graph ... ok
test graph::graph::tests::test_graph_validation ... ok
test graph::graph::tests::test_mixed_types ... ok
test graph::graph::tests::test_mixed_weight_types ... ok
test graph::graph::tests::test_varied_node_edge_types ... ok
test utils::reverse::tests::test_reverse_eq ... ok
test utils::reverse::tests::test_reverse_ord ... ok
test utils::reverse::tests::test_reverse_partial_ord ... ok
test io::csv_io::tests::test_csv_io ... ok

test result: ok. 43 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s

   Doc-tests xgraph

running 57 tests
test src/algorithms/bridges.rs - algorithms::bridges::Bridges::sort_bridges (line 139) ... ok
test src/algorithms/centrality.rs - algorithms::centrality::Centrality (line 37) ... ok
test src/algorithms/bridges.rs - algorithms::bridges::Bridges::find_bridges (line 114) ... ok
test src/algorithms/bridges.rs - algorithms::bridges::Bridges (line 94) ... ok
test src/algorithms/bridges.rs - algorithms::bridges (line 9) ... ok
test src/algorithms/bridges.rs - algorithms::bridges::Graph<W,N,E>::find_bridges (line 168) ... ok
test src/algorithms/centrality.rs - algorithms::centrality (line 11) ... ok
test src/algorithms/bridges.rs - algorithms::bridges::Bridges (line 75) ... ok
test src/algorithms/centrality.rs - algorithms::centrality::Centrality::betweenness_centrality (line 87) ... ok
test src/algorithms/centrality.rs - algorithms::centrality::Centrality::closeness_centrality (line 113) ... ok
test src/algorithms/centrality.rs - algorithms::centrality::Centrality::degree_centrality (line 62) ... ok
test src/algorithms/connectivity.rs - algorithms::connectivity (line 11) ... ok
test src/algorithms/connectivity.rs - algorithms::connectivity::Connectivity (line 40) ... ok
test src/algorithms/connectivity.rs - algorithms::connectivity::Connectivity::find_connected_components (line 111) ... ok
test src/algorithms/connectivity.rs - algorithms::connectivity::Connectivity::find_strongly_connected_components (line 86) ... ok
test src/algorithms/connectivity.rs - algorithms::connectivity::Connectivity::find_weakly_connected_components (line 65) ... ok
test src/algorithms/connectivity.rs - algorithms::connectivity::Connectivity::is_connected (line 186) ... ok
test src/algorithms/connectivity.rs - algorithms::connectivity::Connectivity::is_directed (line 206) ... ok
test src/algorithms/connectivity.rs - algorithms::connectivity::Connectivity::is_strongly_connected (line 164) ... ok
test src/algorithms/connectivity.rs - algorithms::connectivity::Connectivity::is_weakly_connected (line 144) ... ok
test src/algorithms/search.rs - algorithms::search (line 11) ... ok
test src/algorithms/search.rs - algorithms::search::Search::bfs_path (line 64) ... ok
test src/algorithms/search.rs - algorithms::search::Search::dfs (line 90) ... ok
test src/algorithms/search.rs - algorithms::search::Search::has_cycle (line 124) ... ok
test src/algorithms/search.rs - algorithms::search::Search::has_cycle_directed (line 154) ... ok
test src/algorithms/search.rs - algorithms::search::Search::has_cycle_undirected (line 184) ... ok
test src/algorithms/search.rs - algorithms::search::Search::has_node (line 108) ... ok
test src/algorithms/search.rs - algorithms::search::Search::has_path (line 40) ... ok
test src/algorithms/shortest_path.rs - algorithms::shortest_path (line 9) ... ok
test src/algorithms/shortest_path.rs - algorithms::shortest_path::ShortestPath (line 35) ... ok
test src/algorithms/shortest_path.rs - algorithms::shortest_path::ShortestPath::dijkstra (line 61) ... ok
test src/algorithms/wiedemann_ford.rs - algorithms::wiedemann_ford (line 10) ... ok
test src/algorithms/wiedemann_ford.rs - algorithms::wiedemann_ford (line 30) ... ok
test src/algorithms/wiedemann_ford.rs - algorithms::wiedemann_ford::DominatingSetFinder::find_dominating_set (line 62) ... ok
test src/algorithms/wiedemann_ford.rs - algorithms::wiedemann_ford::Graph<W,N,E>::find_dominating_set (line 96) ... ok
test src/graph/graph.rs - graph::graph (line 8) ... ok
test src/graph/graph.rs - graph::graph::Graph (line 37) ... ok
test src/graph/graph.rs - graph::graph::Graph<W,N,E>::add_edge (line 146) ... ok
test src/graph/graph.rs - graph::graph::Graph<W,N,E>::add_edges_batch (line 212) ... ok
test src/graph/graph.rs - graph::graph::Graph<W,N,E>::add_node (line 87) ... ok
test src/graph/graph.rs - graph::graph::Graph<W,N,E>::add_nodes_batch (line 193) ... ok
test src/graph/graph.rs - graph::graph::Graph<W,N,E>::all_nodes (line 580) ... ok
test src/graph/graph.rs - graph::graph::Graph<W,N,E>::from_adjacency_matrix (line 297) ... ok
test src/graph/graph.rs - graph::graph::Graph<W,N,E>::get_all_edge_attributes (line 483) ... ok
test src/graph/graph.rs - graph::graph::Graph<W,N,E>::get_all_edges (line 509) ... ok
test src/graph/graph.rs - graph::graph::Graph<W,N,E>::get_edge_attribute (line 462) ... ok
test src/graph/graph.rs - graph::graph::Graph<W,N,E>::get_neighbors (line 531) ... ok
test src/graph/graph.rs - graph::graph::Graph<W,N,E>::get_node_attribute (line 446) ... ok
test src/graph/graph.rs - graph::graph::Graph<W,N,E>::new (line 67) ... ok
test src/graph/graph.rs - graph::graph::Graph<W,N,E>::remove_edge (line 253) ... ok
test src/graph/graph.rs - graph::graph::Graph<W,N,E>::remove_node (line 109) ... ok
test src/graph/graph.rs - graph::graph::Graph<W,N,E>::set_edge_attribute (line 414) ... ok
test src/graph/graph.rs - graph::graph::Graph<W,N,E>::set_node_attribute (line 388) ... ok
test src/graph/graph.rs - graph::graph::Graph<W,N,E>::to_adjacency_matrix (line 352) ... ok
test src/graph/graph.rs - graph::graph::Graph<W,N,E>::validate_graph (line 553) ... ok
test src/io/csv_io.rs - io::csv_io::CsvIO::load_from_csv (line 73) ... ok
test src/io/csv_io.rs - io::csv_io::CsvIO::save_to_csv (line 37) ... ok

test result: ok. 57 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 12.99s


```

Test coverage includes:
- Graph manipulation invariants
- Algorithm correctness checks
- Edge case handling
- Memory safety verification

## License 📄

MIT License - See [LICENSE](LICENSE) for details.
```