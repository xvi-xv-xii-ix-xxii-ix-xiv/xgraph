//! Example usage of the heterogeneous graph library for Viking trade routes.
//!
//! This module demonstrates the creation, analysis, and clustering of a graph representing
//! Viking-era trade and travel routes between cities. It showcases various graph algorithms
//! and I/O operations provided by the `xgraph` library under the `hgraph` feature.
//!
//! To run this example, enable the `hgraph` feature in your `Cargo.toml`:
//! ```toml
//! [dependencies.xgraph]
//! version = "x.y.z"
//! features = ["hgraph"]
//! ```

#[cfg(feature = "hgraph")]
use std::collections::HashMap;
#[cfg(feature = "hgraph")]
use xgraph::hgraph::algorithms::h_bridges::HeteroBridges;
#[cfg(feature = "hgraph")]
use xgraph::hgraph::algorithms::h_centrality::HeteroCentrality;
#[cfg(feature = "hgraph")]
use xgraph::hgraph::algorithms::h_connectivity::HeteroConnectivity;
#[cfg(feature = "hgraph")]
use xgraph::hgraph::algorithms::h_leiden_clustering::{CommunityConfig, HeteroCommunityDetection};
#[cfg(feature = "hgraph")]
use xgraph::hgraph::algorithms::h_search::HeteroSearch;
#[cfg(feature = "hgraph")]
use xgraph::hgraph::algorithms::h_shortest_path::HeteroShortestPath;
#[cfg(feature = "hgraph")]
use xgraph::hgraph::algorithms::h_wiedemann_ford::HeteroDominatingSetFinder;
#[cfg(feature = "hgraph")]
use xgraph::hgraph::conversion::h_graph_conversion::GraphConversion;
#[cfg(feature = "hgraph")]
use xgraph::hgraph::h_edge::EdgeType;
#[cfg(feature = "hgraph")]
use xgraph::hgraph::h_graph::HeterogeneousGraph;
#[cfg(feature = "hgraph")]
use xgraph::hgraph::h_node::NodeType;
#[cfg(feature = "hgraph")]
use xgraph::hgraph::hcsv_io::CsvIO;

#[cfg(feature = "hgraph")]
type WeightType = f64;

#[cfg(feature = "hgraph")]
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
struct City(String);

#[cfg(feature = "hgraph")]
impl NodeType for City {
    fn as_string(&self) -> String {
        self.0.clone()
    }
}

#[cfg(feature = "hgraph")]
#[derive(Clone, Debug, Default, PartialEq)]
struct Route(String);

#[cfg(feature = "hgraph")]
impl EdgeType for Route {
    fn as_string(&self) -> String {
        self.0.clone()
    }
}

/// Creates a sample graph representing Viking trade and travel routes.
///
/// # Arguments
/// * `directed` - Whether the graph should be directed (`true`) or undirected (`false`).
///
/// # Returns
/// A `HeterogeneousGraph` with cities as nodes and routes as edges, including attributes.
#[cfg(feature = "hgraph")]
fn create_sample_graph(directed: bool) -> HeterogeneousGraph<WeightType, City, Route> {
    let mut graph = HeterogeneousGraph::new(directed);

    let cities = vec![
        "Hedeby",
        "Birka",
        "Kaupang",
        "Dublin",
        "York",
        "Ribe",
        "Trondheim",
        "Uppsala",
        "Novgorod",
        "Aarhus",
        "London",
        "Bristol",
        "Lindisfarne",
    ];
    let city_ids: Vec<usize> = cities
        .into_iter()
        .map(|name| graph.add_node(City(name.to_string())))
        .collect();

    let edge_0_1_river = graph
        .add_edge(city_ids[0], city_ids[1], 300.0, Route("river".to_string()))
        .unwrap();
    let edge_0_5_river = graph
        .add_edge(city_ids[0], city_ids[5], 150.0, Route("river".to_string()))
        .unwrap();
    let edge_0_9_trade = graph
        .add_edge(city_ids[0], city_ids[9], 200.0, Route("trade".to_string()))
        .unwrap();
    let edge_1_2_sea = graph
        .add_edge(city_ids[1], city_ids[2], 400.0, Route("sea".to_string()))
        .unwrap();
    let edge_1_7_trade = graph
        .add_edge(city_ids[1], city_ids[7], 350.0, Route("trade".to_string()))
        .unwrap();
    let edge_2_6_sea = graph
        .add_edge(city_ids[2], city_ids[6], 250.0, Route("sea".to_string()))
        .unwrap();
    let edge_3_4_sea = graph
        .add_edge(city_ids[3], city_ids[4], 500.0, Route("sea".to_string()))
        .unwrap();
    let edge_0_3_sea = graph
        .add_edge(city_ids[0], city_ids[3], 600.0, Route("sea".to_string()))
        .unwrap();
    let edge_4_5_river = graph
        .add_edge(city_ids[4], city_ids[5], 300.0, Route("river".to_string()))
        .unwrap();
    let edge_5_9_trade = graph
        .add_edge(city_ids[5], city_ids[9], 180.0, Route("trade".to_string()))
        .unwrap();
    let edge_6_7_mountain = graph
        .add_edge(
            city_ids[6],
            city_ids[7],
            450.0,
            Route("mountain".to_string()),
        )
        .unwrap();
    let edge_7_8_trade = graph
        .add_edge(city_ids[7], city_ids[8], 700.0, Route("trade".to_string()))
        .unwrap();
    let edge_1_8_river = graph
        .add_edge(city_ids[1], city_ids[8], 650.0, Route("river".to_string()))
        .unwrap();
    let edge_2_9_sea = graph
        .add_edge(city_ids[2], city_ids[9], 220.0, Route("sea".to_string()))
        .unwrap();
    let edge_4_10_sea = graph
        .add_edge(city_ids[4], city_ids[10], 400.0, Route("sea".to_string()))
        .unwrap();
    let edge_10_11_river = graph
        .add_edge(
            city_ids[10],
            city_ids[11],
            200.0,
            Route("river".to_string()),
        )
        .unwrap();
    let edge_11_12_sea = graph
        .add_edge(city_ids[11], city_ids[12], 150.0, Route("sea".to_string()))
        .unwrap();
    let edge_12_4_sea = graph
        .add_edge(city_ids[12], city_ids[4], 300.0, Route("sea".to_string()))
        .unwrap();

    let edge_0_1_trade = graph
        .add_edge(city_ids[0], city_ids[1], 320.0, Route("trade".to_string()))
        .unwrap();
    let edge_0_5_trade = graph
        .add_edge(city_ids[0], city_ids[5], 160.0, Route("trade".to_string()))
        .unwrap();
    let edge_1_8_trade = graph
        .add_edge(city_ids[1], city_ids[8], 680.0, Route("trade".to_string()))
        .unwrap();
    let edge_3_4_trade = graph
        .add_edge(city_ids[3], city_ids[4], 520.0, Route("trade".to_string()))
        .unwrap();
    let edge_4_10_river = graph
        .add_edge(city_ids[4], city_ids[10], 420.0, Route("river".to_string()))
        .unwrap();

    graph.set_node_attribute(0, "type".to_string(), "trade_hub".to_string());
    graph.set_node_attribute(0, "population".to_string(), "5000".to_string());
    graph.set_node_attribute(0, "region".to_string(), "Scandinavia".to_string());

    graph.set_node_attribute(1, "type".to_string(), "port".to_string());
    graph.set_node_attribute(1, "population".to_string(), "3000".to_string());
    graph.set_node_attribute(1, "region".to_string(), "Scandinavia".to_string());

    graph.set_node_attribute(2, "type".to_string(), "settlement".to_string());
    graph.set_node_attribute(2, "population".to_string(), "1000".to_string());
    graph.set_node_attribute(2, "region".to_string(), "Scandinavia".to_string());

    graph.set_node_attribute(3, "type".to_string(), "port".to_string());
    graph.set_node_attribute(3, "population".to_string(), "1000".to_string());
    graph.set_node_attribute(3, "region".to_string(), "Ireland".to_string());

    graph.set_node_attribute(4, "type".to_string(), "fortress".to_string());
    graph.set_node_attribute(4, "population".to_string(), "1000".to_string());
    graph.set_node_attribute(4, "region".to_string(), "England".to_string());

    graph.set_node_attribute(5, "type".to_string(), "trade_hub".to_string());
    graph.set_node_attribute(5, "population".to_string(), "1000".to_string());
    graph.set_node_attribute(5, "region".to_string(), "Scandinavia".to_string());

    graph.set_node_attribute(6, "type".to_string(), "settlement".to_string());
    graph.set_node_attribute(6, "population".to_string(), "1000".to_string());
    graph.set_node_attribute(6, "region".to_string(), "Scandinavia".to_string());

    graph.set_node_attribute(7, "type".to_string(), "religious_center".to_string());
    graph.set_node_attribute(7, "population".to_string(), "1000".to_string());
    graph.set_node_attribute(7, "region".to_string(), "Scandinavia".to_string());

    graph.set_node_attribute(8, "type".to_string(), "trade_hub".to_string());
    graph.set_node_attribute(8, "population".to_string(), "1000".to_string());
    graph.set_node_attribute(8, "region".to_string(), "Russia".to_string());

    graph.set_node_attribute(9, "type".to_string(), "port".to_string());
    graph.set_node_attribute(9, "population".to_string(), "1000".to_string());
    graph.set_node_attribute(9, "region".to_string(), "Scandinavia".to_string());

    graph.set_node_attribute(10, "type".to_string(), "port".to_string());
    graph.set_node_attribute(10, "population".to_string(), "7000".to_string());
    graph.set_node_attribute(10, "region".to_string(), "England".to_string());

    graph.set_node_attribute(11, "type".to_string(), "port".to_string());
    graph.set_node_attribute(11, "population".to_string(), "3000".to_string());
    graph.set_node_attribute(11, "region".to_string(), "England".to_string());

    graph.set_node_attribute(12, "type".to_string(), "religious_center".to_string());
    graph.set_node_attribute(12, "population".to_string(), "500".to_string());
    graph.set_node_attribute(12, "region".to_string(), "England".to_string());

    graph.set_edge_attribute(edge_0_1_river, "priority".to_string(), "low".to_string());
    graph.set_edge_attribute(edge_0_5_river, "priority".to_string(), "medium".to_string());
    graph.set_edge_attribute(edge_0_9_trade, "priority".to_string(), "high".to_string());
    graph.set_edge_attribute(edge_1_2_sea, "priority".to_string(), "medium".to_string());
    graph.set_edge_attribute(edge_1_7_trade, "priority".to_string(), "medium".to_string());
    graph.set_edge_attribute(edge_2_6_sea, "priority".to_string(), "high".to_string());
    graph.set_edge_attribute(edge_3_4_sea, "priority".to_string(), "medium".to_string());
    graph.set_edge_attribute(edge_0_3_sea, "priority".to_string(), "medium".to_string());
    graph.set_edge_attribute(edge_4_5_river, "priority".to_string(), "medium".to_string());
    graph.set_edge_attribute(edge_5_9_trade, "priority".to_string(), "high".to_string());
    graph.set_edge_attribute(
        edge_6_7_mountain,
        "priority".to_string(),
        "medium".to_string(),
    );
    graph.set_edge_attribute(edge_7_8_trade, "priority".to_string(), "medium".to_string());
    graph.set_edge_attribute(edge_1_8_river, "priority".to_string(), "medium".to_string());
    graph.set_edge_attribute(edge_2_9_sea, "priority".to_string(), "medium".to_string());
    graph.set_edge_attribute(edge_4_10_sea, "priority".to_string(), "high".to_string());
    graph.set_edge_attribute(
        edge_10_11_river,
        "priority".to_string(),
        "medium".to_string(),
    );
    graph.set_edge_attribute(edge_11_12_sea, "priority".to_string(), "low".to_string());
    graph.set_edge_attribute(edge_12_4_sea, "priority".to_string(), "medium".to_string());

    graph.set_edge_attribute(edge_0_1_trade, "priority".to_string(), "high".to_string());
    graph.set_edge_attribute(edge_0_5_trade, "priority".to_string(), "high".to_string());
    graph.set_edge_attribute(edge_1_8_trade, "priority".to_string(), "high".to_string());
    graph.set_edge_attribute(edge_3_4_trade, "priority".to_string(), "medium".to_string());
    graph.set_edge_attribute(
        edge_4_10_river,
        "priority".to_string(),
        "medium".to_string(),
    );

    graph
}

/// Prints an overview of the graph’s nodes and edges.
///
/// # Arguments
/// * `graph` - The graph to display.
#[cfg(feature = "hgraph")]
fn print_graph_details(graph: &HeterogeneousGraph<WeightType, City, Route>) {
    let nodes: Vec<(usize, &City)> = graph
        .nodes
        .iter()
        .map(|(id, node)| (id, &node.data))
        .collect();
    let edges = graph.get_all_edges();

    println!("\n=== Viking Graph Overview ===");
    println!(
        "Type: {}",
        if graph.directed {
            "Directed"
        } else {
            "Undirected"
        }
    );
    println!("Total nodes: {}", nodes.len());
    println!("Total edges: {}", edges.len());

    println!("\nNodes:");
    for (id, city) in nodes.iter() {
        println!("  {}: {}", id, city.as_string());
    }

    println!("\nEdges:");
    for (from, to, weight, edge) in edges.iter() {
        println!(
            "  {} -> {} (Weight: {}, Type: {})",
            from,
            to,
            weight,
            edge.as_string()
        );
    }
}

/// Demonstrates saving and loading the graph to/from CSV files.
///
/// # Arguments
/// * `graph` - The graph to demonstrate I/O operations on.
///
/// # Panics
/// Panics if CSV operations fail (for demonstration simplicity).
#[cfg(feature = "hgraph")]
fn demonstrate_io(graph: &mut HeterogeneousGraph<WeightType, City, Route>) {
    println!("\n=== IO and Format Demonstration ===");

    match graph.to_string_graph() {
        Ok(string_graph) => {
            println!("\nSaving to CSV:");
            match string_graph.save_to_csv("viking_nodes.csv", "viking_edges.csv") {
                Ok(()) => println!("  Graph saved to viking_nodes.csv and viking_edges.csv"),
                Err(e) => println!("  Failed to save to CSV: {}", e),
            }

            println!("\nLoading from CSV:");
            match HeterogeneousGraph::<WeightType, String, String>::load_from_csv(
                "viking_nodes.csv",
                "viking_edges.csv",
                true,
            ) {
                Ok(loaded_graph) => println!(
                    "  Loaded graph: {} nodes, {} edges",
                    loaded_graph.nodes.len(),
                    loaded_graph.edges.len()
                ),
                Err(e) => println!("  Failed to load from CSV: {}", e),
            }
        }
        Err(e) => println!("  Failed to convert graph to string representation: {}", e),
    }
}

/// Performs various graph analyses and prints results.
///
/// # Arguments
/// * `graph` - The graph to analyze.
#[cfg(feature = "hgraph")]
fn analyze_graph(graph: &mut HeterogeneousGraph<WeightType, City, Route>) {
    println!("\n=== Viking Graph Analysis ===");

    let num_nodes = graph.nodes.len();
    let num_edges = graph.get_all_edges().len();
    print_metrics(num_nodes, num_edges, graph.directed);
    print_connectivity(graph, num_nodes);
    print_centrality(graph);
    print_attributes(graph);
    print_bridges(graph);
    print_connected_components(graph);
    print_density(num_nodes, num_edges, graph.directed);
    print_wiedemann_ford(graph);
    print_shortest_paths(graph);
    analyze_by_edge_types(graph);
}

/// Prints basic graph metrics.
///
/// # Arguments
/// * `nodes` - Number of nodes in the graph.
/// * `edges` - Number of edges in the graph.
/// * `directed` - Whether the graph is directed.
#[cfg(feature = "hgraph")]
fn print_metrics(nodes: usize, edges: usize, directed: bool) {
    println!("\nBasic Metrics:");
    println!("  Number of nodes: {}", nodes);
    println!("  Number of edges: {}", edges);
    println!(
        "  Type of graph: {}",
        if directed { "directed" } else { "undirected" }
    );
}

/// Prints connectivity and path information.
///
/// # Arguments
/// * `graph` - The graph to check connectivity on.
/// * `node_count` - Number of nodes in the graph.
#[cfg(feature = "hgraph")]
fn print_connectivity(graph: &mut HeterogeneousGraph<WeightType, City, Route>, node_count: usize) {
    println!("\nConnectivity and Paths:");
    if node_count >= 6 {
        println!(
            "  Path from Hedeby (0) to Trondheim (6) exists: {}",
            graph.has_path(0, 6)
        );
        match graph.bfs_path(0, 6) {
            Ok(Some(path)) => println!("  BFS path Hedeby->Trondheim: {:?}", path),
            Ok(None) => println!("  BFS path Hedeby->Trondheim: No path found"),
            Err(e) => println!("  BFS path Hedeby->Trondheim: Error - {}", e),
        }
        match graph.dijkstra_path(0, 6) {
            Ok(Some(path)) => println!("  Dijkstra path Hedeby->Trondheim: {:?}", path),
            Ok(None) => println!("  Dijkstra path Hedeby->Trondheim: No path found"),
            Err(e) => println!("  Dijkstra path Hedeby->Trondheim: Error - {}", e),
        }
    } else {
        println!("  Not enough nodes for path example (need at least 6)");
    }
}

/// Prints degree centrality metrics.
///
/// # Arguments
/// * `graph` - The graph to compute centrality for.
#[cfg(feature = "hgraph")]
fn print_centrality(graph: &mut HeterogeneousGraph<WeightType, City, Route>) {
    println!("\nCentrality:");
    match graph.degree_centrality() {
        Ok(centrality) => {
            println!("  Degree centrality:");
            for (node, val) in centrality.iter() {
                println!("    Node {}: {}", node, val);
            }
        }
        Err(e) => println!("  Error computing degree centrality: {}", e),
    }
}

/// Prints node and edge attributes.
///
/// # Arguments
/// * `graph` - The graph to extract attributes from.
#[cfg(feature = "hgraph")]
fn print_attributes(graph: &mut HeterogeneousGraph<WeightType, City, Route>) {
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

    println!("\nNode Attributes:");
    if node_attrs.is_empty() {
        println!("  No node attributes");
    } else {
        for (attr, values) in node_attrs.iter() {
            println!("  Attribute '{}':", attr);
            for (val, ids) in values {
                println!("    {}: {} nodes ({:?})", val, ids.len(), ids);
            }
        }
    }

    let edge_attrs = graph
        .get_all_edges()
        .iter()
        .flat_map(|(from, to, weight, _edge_data)| {
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

    println!("\nEdge Attributes:");
    if edge_attrs.is_empty() {
        println!("  No edge attributes");
    } else {
        for (attr, values) in edge_attrs.iter() {
            println!("  Attribute '{}':", attr);
            for (val, edges) in values {
                println!("    {}: {} edges ({:?})", val, edges.len(), edges);
            }
        }
    }
}

/// Prints bridges in the graph.
///
/// # Arguments
/// * `graph` - The graph to find bridges in.
#[cfg(feature = "hgraph")]
fn print_bridges(graph: &mut HeterogeneousGraph<WeightType, City, Route>) {
    println!("\nBridges:");
    let bridges = graph.find_bridges();
    if bridges.is_empty() {
        println!("  No bridges found");
    } else {
        println!("  Critical edges ({}):", bridges.len());
        for (from, to, edge_type) in bridges {
            println!("    {} -> {} (Type: {})", from, to, edge_type);
        }
    }
}

/// Prints connected components and connectivity checks.
///
/// # Arguments
/// * `graph` - The graph to analyze connectivity for.
#[cfg(feature = "hgraph")]
fn print_connected_components(graph: &mut HeterogeneousGraph<WeightType, City, Route>) {
    println!("\nConnected Components:");

    let components = if graph.is_directed() {
        println!("  Strongly connected components:");
        match graph.find_strongly_connected_components() {
            Ok(scc) => {
                println!("    Count: {}", scc.len());
                scc
            }
            Err(e) => {
                println!("    Error: {}", e);
                Vec::new()
            }
        };

        println!("  Weakly connected components:");
        match graph.find_weakly_connected_components() {
            Ok(wcc) => {
                println!("    Count: {}", wcc.len());
                wcc
            }
            Err(e) => {
                println!("    Error: {}", e);
                Vec::new()
            }
        }
    } else {
        match graph.find_connected_components() {
            Ok(components) => components,
            Err(e) => {
                println!("    Error: {}", e);
                Vec::new()
            }
        }
    };

    println!("  Components found:");
    for (i, c) in components.iter().enumerate() {
        println!("    Component {}: {} nodes", i, c.len());
    }

    println!("\n  Additional connectivity checks:");
    println!(
        "    Overall connectivity: {}",
        graph.is_connected().unwrap_or(false)
    );
    println!(
        "    Weakly connected: {}",
        graph.is_weakly_connected().unwrap_or(false)
    );
    if graph.is_directed() {
        println!(
            "    Strongly connected: {}",
            graph.is_strongly_connected().unwrap_or(false)
        );
    }
}

/// Prints the graph’s density.
///
/// # Arguments
/// * `nodes` - Number of nodes in the graph.
/// * `edges` - Number of edges in the graph.
/// * `directed` - Whether the graph is directed.
#[cfg(feature = "hgraph")]
fn print_density(nodes: usize, edges: usize, directed: bool) {
    println!("\nDensity:");
    let density = calculate_density(nodes, edges, directed);
    println!("  Graph density: {:.4}", density);
}

/// Calculates the graph’s density.
///
/// # Arguments
/// * `nodes` - Number of nodes in the graph.
/// * `edges` - Number of edges in the graph.
/// * `directed` - Whether the graph is directed.
///
/// # Returns
/// The density as a `f64` value.
#[cfg(feature = "hgraph")]
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
/// # Arguments
/// * `graph` - The graph to find the dominating set for.
#[cfg(feature = "hgraph")]
fn print_wiedemann_ford(graph: &mut HeterogeneousGraph<WeightType, City, Route>) {
    println!("\nWiedemann-Ford Dominating Set:");
    match graph.find_dominating_set() {
        Ok(dominating_set) => println!("  Dominating set (all edges): {:?}", dominating_set),
        Err(e) => println!("  Error computing dominating set: {}", e),
    }
}

/// Prints shortest path distances from Hedeby using Dijkstra’s algorithm.
///
/// # Arguments
/// * `graph` - The graph to compute shortest paths for.
#[cfg(feature = "hgraph")]
fn print_shortest_paths(graph: &mut HeterogeneousGraph<WeightType, City, Route>) {
    println!("\nShortest Paths (Dijkstra):");
    match graph.dijkstra(0) {
        Ok((distances, _)) => println!("  Distances from Hedeby (node 0): {:?}", distances),
        Err(e) => println!("  Error computing distances from Hedeby: {}", e),
    }
}

/// Analyzes the graph by edge types and prints results.
///
/// # Arguments
/// * `graph` - The graph to analyze by edge types.
#[cfg(feature = "hgraph")]
fn analyze_by_edge_types(graph: &mut HeterogeneousGraph<WeightType, City, Route>) {
    println!("\n=== Analysis by Edge Types ===");

    let edge_types = vec![
        vec![Route("river".to_string())],
        vec![Route("sea".to_string())],
        vec![Route("trade".to_string())],
    ];

    for edge_type in edge_types {
        let type_name = edge_type[0].as_string();
        let edge_type_owned: Vec<String> =
            edge_type.iter().map(|route| route.as_string()).collect();
        let edge_type_str: Vec<&str> = edge_type_owned.iter().map(|s| s.as_str()).collect();

        println!("\nEdge Type: {}", type_name);

        let config = CommunityConfig {
            gamma: 0.2,
            resolution: 0.5,
            iterations: 10,
            deterministic: true,
            seed: Some(42),
            min_community_size: 2,
        };
        match graph.detect_communities_by_types(&edge_type, config) {
            Ok(communities) => {
                println!("  Leiden Clustering:");
                println!("    Found {} communities:", communities.len());
                for (idx, (comm_id, nodes)) in communities.iter().enumerate() {
                    println!(
                        "      Cluster {} (ID: {}): {} nodes: {:?}",
                        idx + 1,
                        comm_id,
                        nodes.len(),
                        nodes
                    );
                }
            }
            Err(e) => println!("    Error detecting communities: {}", e),
        }

        match graph.find_dominating_set_by_types(&edge_type) {
            Ok(dominating_set) => {
                println!("  Dominating Set:");
                println!("    Nodes: {:?}", dominating_set);
            }
            Err(e) => println!("  Error computing dominating set by types: {}", e),
        }

        match graph.dijkstra_by_types(0, &edge_type_str) {
            Ok((distances, _)) => println!(
                "  Shortest Paths from Hedeby (0): Distances: {:?}",
                distances
            ),
            Err(e) => println!("  Shortest Paths from Hedeby (0): Error - {}", e),
        }
    }
}

/// Performs clustering experiments using the Leiden algorithm.
///
/// # Arguments
/// * `graph` - The graph to cluster.
#[cfg(feature = "hgraph")]
fn perform_clustering(graph: &HeterogeneousGraph<WeightType, City, Route>) {
    let resolutions = [0.5, 1.0];
    let gammas = [0.5, 1.0];
    const LEIDEN_ITERATIONS: usize = 10;

    println!("\n=== Viking Leiden Clustering Experiments ===");

    for (i, &resolution) in resolutions.iter().enumerate() {
        for (j, &gamma) in gammas.iter().enumerate() {
            let fixed_seed = 42;

            let config_det = CommunityConfig {
                gamma,
                resolution,
                iterations: LEIDEN_ITERATIONS,
                deterministic: true,
                seed: Some(fixed_seed),
                min_community_size: 2,
            };

            match graph.detect_communities_with_config(config_det.clone()) {
                Ok(communities_det) => {
                    println!("\nExperiment #{}-{} (Deterministic)", i + 1, j + 1);
                    println!(
                        "  Parameters: γ = {:.1}, resolution = {:.1}",
                        gamma, resolution
                    );
                    println!("  Found {} communities:", communities_det.len());
                    for (idx, (comm_id, nodes)) in communities_det.iter().enumerate() {
                        println!(
                            "    Cluster {} (ID: {}): {} nodes: {:?}",
                            idx + 1,
                            comm_id,
                            nodes.len(),
                            nodes
                        );
                    }
                }
                Err(e) => println!("  Error in deterministic clustering: {}", e),
            }

            let config_non_det = CommunityConfig {
                deterministic: false,
                seed: None,
                ..config_det
            };

            match graph.detect_communities_with_config(config_non_det) {
                Ok(communities_non_det) => {
                    println!("\nExperiment #{}-{} (Non-Deterministic)", i + 1, j + 1);
                    println!(
                        "  Parameters: γ = {:.1}, resolution = {:.1}",
                        gamma, resolution
                    );
                    println!("  Found {} communities:", communities_non_det.len());
                    for (idx, (comm_id, nodes)) in communities_non_det.iter().enumerate() {
                        println!(
                            "    Cluster {} (ID: {}): {} nodes: {:?}",
                            idx + 1,
                            comm_id,
                            nodes.len(),
                            nodes
                        );
                    }
                }
                Err(e) => println!("  Error in non-deterministic clustering: {}", e),
            }
        }
    }
}

/// Main entry point when the `hgraph` feature is enabled.
#[cfg(feature = "hgraph")]
fn main() {
    let mut graph = create_sample_graph(false);
    print_graph_details(&graph);
    analyze_graph(&mut graph);
    perform_clustering(&graph);
    demonstrate_io(&mut graph);
}

/// Main entry point when the `hgraph` feature is disabled.
#[cfg(not(feature = "hgraph"))]
fn main() {
    println!("This example requires the 'hgraph' feature. Enable it in Cargo.toml with:");
    println!("```toml");
    println!("[dependencies.xgraph]");
    println!("version = \"x.y.z\"");
    println!("features = [\"hgraph\"]");
    println!("```");
}
