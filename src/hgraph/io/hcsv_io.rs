//! Module for CSV input/output operations on heterogeneous multigraphs.
//!
//! This module provides functionality to serialize and deserialize `HeterogeneousGraph` instances
//! to and from CSV files. It supports saving and loading node and edge data along with their
//! attributes in a structured format. The implementation is feature-gated under `hgraph`.
//!
//! ## CSV Format
//! - **Nodes CSV**: Columns include `node_id`, `data`, and dynamically determined attribute keys.
//! - **Edges CSV**: Columns include `edge_id`, `from`, `to`, `weight`, `data`, and dynamic attributes.
//!
//! ## Features
//! - Available only when the `hgraph` feature is enabled in `Cargo.toml`:
//!   ```toml
//!   [dependencies.xgraph]
//!   version = "x.y.z"
//!   features = ["hgraph"]
//!   ```
//!
//! ## Error Handling
//! Operations return custom `Result` types with detailed errors instead of panicking, allowing
//! users to handle issues gracefully.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};

#[cfg(feature = "hgraph")]
use crate::hgraph::h_graph::HeterogeneousGraph;

// --- Error Handling Additions ---

/// Error type for CSV input/output operations.
///
/// Represents various failure modes that can occur during graph serialization or deserialization.
#[cfg(feature = "hgraph")]
#[derive(Debug)]
pub enum CsvIoError {
    /// An I/O error occurred while reading or writing files.
    IoError(io::Error),
    /// The CSV file is malformed or missing required headers.
    InvalidFormat(String),
    /// Failed to parse a field (e.g., node data, weight, or edge data).
    ParseError {
        /// The field that failed to parse.
        field: String,
        /// The value that could not be parsed.
        value: String,
        /// The underlying parse error details.
        details: String,
    },
}

#[cfg(feature = "hgraph")]
impl std::fmt::Display for CsvIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CsvIoError::IoError(e) => write!(f, "I/O error: {}", e),
            CsvIoError::InvalidFormat(msg) => write!(f, "Invalid CSV format: {}", msg),
            CsvIoError::ParseError {
                field,
                value,
                details,
            } => write!(f, "Failed to parse {} '{}': {}", field, value, details),
        }
    }
}

#[cfg(feature = "hgraph")]
impl std::error::Error for CsvIoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CsvIoError::IoError(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "hgraph")]
impl From<io::Error> for CsvIoError {
    fn from(err: io::Error) -> Self {
        CsvIoError::IoError(err)
    }
}

/// Result type alias for CSV I/O operations.
///
/// Wraps operation results to provide detailed error information via `CsvIoError`.
#[cfg(feature = "hgraph")]
pub type Result<T> = std::result::Result<T, CsvIoError>;

// --- Trait Definition ---

/// A trait for CSV input/output operations on heterogeneous multigraphs.
///
/// This trait defines methods to save a graph to CSV files and load a graph from CSV files.
/// It supports saving and loading node and edge data along with their attributes in a structured format.
///
/// # Type Parameters
/// - `W`: The weight type of the graph edges (e.g., `u32`, `f64`).
/// - `N`: The node data type, must implement `NodeType`.
/// - `E`: The edge data type, must implement `EdgeType`.
///
/// # Requirements
/// - For `save_to_csv`: `W`, `N`, and `E` must implement `Display`.
/// - For `load_from_csv`: `W`, `N`, and `E` must implement `FromStr` and `Default`, with debuggable parse errors.
///
/// # Features
/// This trait is only available when the `hgraph` feature is enabled.
#[cfg(feature = "hgraph")]
pub trait CsvIO<W, N, E> {
    /// Saves the graph to CSV files.
    ///
    /// Writes the graph's nodes and edges to two separate CSV files:
    /// - `nodes_file`: Contains node IDs, their data (via `as_string`), and attributes as columns.
    /// - `edges_file`: Contains edge IDs, source and target node IDs, weights, edge data (via `as_string`), and attributes as columns.
    ///
    /// Attributes are dynamically determined from the graph and written as additional columns.
    /// Missing attributes for a node or edge are represented as empty strings.
    ///
    /// # Arguments
    /// - `nodes_file`: The path to the file where nodes will be saved.
    /// - `edges_file`: The path to the file where edges will be saved.
    ///
    /// # Returns
    /// A `Result<()>` indicating success or failure with a `CsvIoError` on error.
    ///
    /// # Examples
    /// ```rust
    /// #[cfg(feature = "hgraph")]
    /// {
    ///     use crate::xgraph::hgraph::h_graph::HeterogeneousGraph;
    ///     use crate::xgraph::hgraph::hcsv_io::CsvIO;
    ///     let mut graph = HeterogeneousGraph::<f64, String, String>::new(false);
    ///     let n1 = graph.add_node("Alice".to_string());
    ///     let n2 = graph.add_node("Bob".to_string());
    ///     graph.add_edge(n1, n2, 1.5, "friend".to_string()).unwrap();
    ///     match graph.save_to_csv("nodes.csv", "edges.csv") {
    ///         Ok(()) => println!("Graph saved successfully"),
    ///         Err(e) => println!("Failed to save graph: {}", e),
    ///     }
    /// }
    /// ```
    fn save_to_csv(&self, nodes_file: &str, edges_file: &str) -> Result<()>
    where
        W: Copy + PartialEq + std::fmt::Display,
        N: Clone + std::fmt::Debug + std::fmt::Display,
        E: Clone + std::fmt::Debug + std::fmt::Display;

    /// Loads a graph from CSV files.
    ///
    /// Reads a graph from two CSV files:
    /// - `nodes_file`: Contains node IDs, data, and attributes.
    /// - `edges_file`: Contains edge IDs, source and target node IDs, weights, edge data, and attributes.
    ///
    /// The node and edge data are parsed from the CSV using `FromStr`. Attributes are loaded dynamically based on the CSV headers.
    ///
    /// # Arguments
    /// - `nodes_file`: The path to the file containing nodes.
    /// - `edges_file`: The path to the file containing edges.
    /// - `directed`: A boolean indicating whether the loaded graph should be directed.
    ///
    /// # Returns
    /// A `Result<Self>` containing the loaded graph or a `CsvIoError` on failure.
    ///
    /// # Examples
    /// ```rust
    /// #[cfg(feature = "hgraph")]
    /// {
    ///     use crate::xgraph::hgraph::h_graph::HeterogeneousGraph;
    ///     use crate::xgraph::hgraph::hcsv_io::CsvIO;
    ///     match HeterogeneousGraph::<f64, String, String>::load_from_csv("nodes.csv", "edges.csv", false) {
    ///         Ok(graph) => println!("Loaded graph with {} nodes", graph.nodes.len()),
    ///         Err(e) => println!("Failed to load graph: {}", e),
    ///     }
    /// }
    /// ```
    fn load_from_csv(nodes_file: &str, edges_file: &str, directed: bool) -> Result<Self>
    where
        Self: Sized,
        W: Copy + PartialEq + Default + std::str::FromStr,
        N: Clone + std::fmt::Debug + std::str::FromStr,
        E: Clone + std::fmt::Debug + Default + std::str::FromStr,
        <W as std::str::FromStr>::Err: std::fmt::Debug,
        <N as std::str::FromStr>::Err: std::fmt::Debug,
        <E as std::str::FromStr>::Err: std::fmt::Debug;
}

// --- Implementation ---

#[cfg(feature = "hgraph")]
impl<W, N, E> CsvIO<W, N, E> for HeterogeneousGraph<W, N, E>
where
    W: Copy + PartialEq + std::fmt::Debug,
    N: crate::hgraph::h_node::NodeType,
    E: crate::hgraph::h_edge::EdgeType + Default,
{
    fn save_to_csv(&self, nodes_file: &str, edges_file: &str) -> Result<()>
    where
        W: std::fmt::Display,
        N: std::fmt::Display,
        E: std::fmt::Display,
    {
        // --- Save Nodes ---
        let mut nodes_writer = File::create(nodes_file).map_err(CsvIoError::IoError)?;

        // Collect and sort unique node attribute keys
        let mut node_attrs: Vec<String> = self
            .nodes
            .iter()
            .flat_map(|(_, node)| node.attributes.keys())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .cloned()
            .collect();
        node_attrs.sort();

        // Write header
        writeln!(nodes_writer, "node_id,data,{}", node_attrs.join(","))
            .map_err(CsvIoError::IoError)?;

        // Write node data and attributes
        for (id, node) in self.nodes.iter() {
            let attrs: Vec<String> = node_attrs
                .iter()
                .map(|key| {
                    node.attributes
                        .get(key)
                        .map_or("".to_string(), |v| v.to_string())
                })
                .collect();
            writeln!(
                nodes_writer,
                "{},{},{}",
                id,
                node.data.as_string(),
                attrs.join(",")
            )
            .map_err(CsvIoError::IoError)?;
        }

        // --- Save Edges ---
        let mut edges_writer = File::create(edges_file).map_err(CsvIoError::IoError)?;

        // Collect and sort unique edge attribute keys
        let mut edge_attrs: Vec<String> = self
            .edges
            .iter()
            .flat_map(|(_, edge)| edge.attributes.keys())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .cloned()
            .collect();
        edge_attrs.sort();

        // Write header
        writeln!(
            edges_writer,
            "edge_id,from,to,weight,data,{}",
            edge_attrs.join(",")
        )
        .map_err(CsvIoError::IoError)?;

        // Write edge data and attributes
        for (id, edge) in self.edges.iter() {
            let attrs: Vec<String> = edge_attrs
                .iter()
                .map(|key| {
                    edge.attributes
                        .get(key)
                        .map_or("".to_string(), |v| v.to_string())
                })
                .collect();
            writeln!(
                edges_writer,
                "{},{},{},{},{},{}",
                id,
                edge.from,
                edge.to,
                edge.weight,
                edge.data.as_string(),
                attrs.join(",")
            )
            .map_err(CsvIoError::IoError)?;
        }

        Ok(())
    }

    fn load_from_csv(nodes_file: &str, edges_file: &str, directed: bool) -> Result<Self>
    where
        Self: Sized,
        W: Default + std::str::FromStr,
        N: std::str::FromStr,
        E: std::str::FromStr,
        <W as std::str::FromStr>::Err: std::fmt::Debug,
        <N as std::str::FromStr>::Err: std::fmt::Debug,
        <E as std::str::FromStr>::Err: std::fmt::Debug,
    {
        let mut graph = HeterogeneousGraph::new(directed);

        // --- Load Nodes ---
        let nodes_reader = BufReader::new(File::open(nodes_file).map_err(CsvIoError::IoError)?);
        let mut lines = nodes_reader.lines();

        // Parse header
        let header = lines
            .next()
            .ok_or(CsvIoError::InvalidFormat("Empty nodes file".to_string()))?
            .map_err(CsvIoError::IoError)?;
        let attr_keys: Vec<&str> = header.split(',').skip(2).collect();
        if header.is_empty() || !header.starts_with("node_id,data") {
            return Err(CsvIoError::InvalidFormat(
                "Nodes CSV must start with 'node_id,data'".to_string(),
            ));
        }

        // Parse nodes
        for (line_num, line) in lines.enumerate() {
            let line = line.map_err(CsvIoError::IoError)?;
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 2 {
                return Err(CsvIoError::InvalidFormat(format!(
                    "Line {} in nodes CSV has too few fields",
                    line_num + 2
                )));
            }
            let _node_id: usize = parts[0].parse().unwrap_or_default(); // Ignored, new ID assigned
            let data = parts[1].parse().map_err(|e| CsvIoError::ParseError {
                field: "node data".to_string(),
                value: parts[1].to_string(),
                details: format!("{:?}", e),
            })?;
            let node = graph.add_node(data);

            // Load attributes
            for (key, value) in attr_keys.iter().zip(parts.iter().skip(2)) {
                if !value.is_empty() {
                    graph.set_node_attribute(node, key.to_string(), value.to_string());
                }
            }
        }

        // --- Load Edges ---
        let edges_reader = BufReader::new(File::open(edges_file).map_err(CsvIoError::IoError)?);
        let mut lines = edges_reader.lines();

        // Parse header
        let header = lines
            .next()
            .ok_or(CsvIoError::InvalidFormat("Empty edges file".to_string()))?
            .map_err(CsvIoError::IoError)?;
        let attr_keys: Vec<&str> = header.split(',').skip(5).collect();
        if header.is_empty() || !header.starts_with("edge_id,from,to,weight,data") {
            return Err(CsvIoError::InvalidFormat(
                "Edges CSV must start with 'edge_id,from,to,weight,data'".to_string(),
            ));
        }

        // Parse edges
        for (line_num, line) in lines.enumerate() {
            let line = line.map_err(CsvIoError::IoError)?;
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 5 {
                return Err(CsvIoError::InvalidFormat(format!(
                    "Line {} in edges CSV has too few fields",
                    line_num + 2
                )));
            }
            let _edge_id: usize = parts[0].parse().unwrap_or_default(); // Ignored, new ID assigned
            let from: usize = parts[1].parse().unwrap_or_default();
            let to: usize = parts[2].parse().unwrap_or_default();
            let weight = parts[3].parse().map_err(|e| CsvIoError::ParseError {
                field: "weight".to_string(),
                value: parts[3].to_string(),
                details: format!("{:?}", e),
            })?;
            let data = parts[4].parse().map_err(|e| CsvIoError::ParseError {
                field: "edge data".to_string(),
                value: parts[4].to_string(),
                details: format!("{:?}", e),
            })?;

            let edge_id = graph.add_edge(from, to, weight, data).unwrap_or_default(); // Assuming add_edge returns Result<usize, _>

            // Load attributes
            for (key, value) in attr_keys.iter().zip(parts.iter().skip(5)) {
                if !value.is_empty() {
                    graph.set_edge_attribute(edge_id, key.to_string(), value.to_string());
                }
            }
        }

        Ok(graph)
    }
}

// --- Tests ---

#[cfg(test)]
#[cfg(feature = "hgraph")]
mod tests {
    use super::*;
    use crate::hgraph::h_edge::EdgeType;
    use crate::hgraph::h_node::NodeType;
    use std::fmt;
    use std::str::FromStr;

    /// A simple node type for testing, representing a labeled entity.
    #[derive(Clone, Eq, PartialEq, Hash, Debug)]
    struct TestNode(String);

    impl NodeType for TestNode {
        fn as_string(&self) -> String {
            self.0.clone()
        }
    }

    impl fmt::Display for TestNode {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl FromStr for TestNode {
        type Err = String;
        fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
            Ok(TestNode(s.to_string()))
        }
    }

    /// A simple edge type for testing, representing a connection type.
    #[derive(Clone, Debug, Default)]
    struct TestEdge(String);

    impl EdgeType for TestEdge {
        fn as_string(&self) -> String {
            self.0.clone()
        }
    }

    impl fmt::Display for TestEdge {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl FromStr for TestEdge {
        type Err = String;
        fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
            Ok(TestEdge(s.to_string()))
        }
    }

    /// Tests basic CSV save and load functionality for a simple graph.
    #[test]
    fn test_csv_io_basic() {
        let mut graph = HeterogeneousGraph::<u32, TestNode, TestEdge>::new(false);
        let n1 = graph.add_node(TestNode("A".to_string()));
        let n2 = graph.add_node(TestNode("B".to_string()));
        let edge_id = graph
            .add_edge(n1, n2, 1, TestEdge("edge".to_string()))
            .unwrap();
        graph.set_node_attribute(n1, "color".to_string(), "red".to_string());
        graph.set_edge_attribute(edge_id, "type".to_string(), "road".to_string());

        assert!(graph
            .save_to_csv("test_nodes_basic.csv", "test_edges_basic.csv")
            .is_ok());
        let loaded_graph = HeterogeneousGraph::<u32, TestNode, TestEdge>::load_from_csv(
            "test_nodes_basic.csv",
            "test_edges_basic.csv",
            false,
        )
        .unwrap();

        assert_eq!(graph.nodes.len(), loaded_graph.nodes.len());
        assert_eq!(graph.edges.len(), loaded_graph.edges.len());
        assert_eq!(
            graph.get_node_data(n1).unwrap().as_string(),
            loaded_graph.get_node_data(0).unwrap().as_string()
        );
        assert_eq!(
            graph.get_edge_data_by_id(edge_id).unwrap().as_string(),
            loaded_graph.get_edge_data_by_id(0).unwrap().as_string()
        );
    }

    /// Tests handling of multiple edges between the same nodes (multigraph behavior).
    #[test]
    fn test_csv_io_multigraph() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(true);
        let n1 = graph.add_node(TestNode("X".to_string()));
        let n2 = graph.add_node(TestNode("Y".to_string()));
        let edge1 = graph
            .add_edge(n1, n2, 1.5, TestEdge("friend".to_string()))
            .unwrap();
        let edge2 = graph
            .add_edge(n1, n2, 2.0, TestEdge("colleague".to_string()))
            .unwrap();
        graph.set_node_attribute(n1, "type".to_string(), "person".to_string());
        graph.set_edge_attribute(edge1, "strength".to_string(), "strong".to_string());
        graph.set_edge_attribute(edge2, "strength".to_string(), "weak".to_string());

        assert!(graph
            .save_to_csv("test_nodes_multi.csv", "test_edges_multi.csv")
            .is_ok());
        let loaded_graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::load_from_csv(
            "test_nodes_multi.csv",
            "test_edges_multi.csv",
            true,
        )
        .unwrap();

        assert_eq!(graph.nodes.len(), loaded_graph.nodes.len());
        assert_eq!(graph.edges.len(), loaded_graph.edges.len());
        assert_eq!(loaded_graph.get_edges_between(0, 1).len(), 2);
    }

    /// Tests error handling for invalid CSV files.
    #[test]
    fn test_csv_io_invalid_file() {
        let result = HeterogeneousGraph::<u32, TestNode, TestEdge>::load_from_csv(
            "nonexistent_nodes.csv",
            "nonexistent_edges.csv",
            false,
        );
        assert!(matches!(result, Err(CsvIoError::IoError(_))));

        std::fs::write("empty_nodes.csv", "").unwrap();
        let result = HeterogeneousGraph::<u32, TestNode, TestEdge>::load_from_csv(
            "empty_nodes.csv",
            "nonexistent_edges.csv",
            false,
        );
        assert!(matches!(result, Err(CsvIoError::InvalidFormat(_))));
    }
}
