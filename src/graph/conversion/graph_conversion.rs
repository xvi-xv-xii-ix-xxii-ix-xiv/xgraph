//! Module for graph conversion between different node and edge data types
//!
//! This module provides functionality to transform a graph's node and edge data types while preserving
//! its structure, weights, and attributes. It is useful for adapting graphs to different contexts,
//! such as converting to string-based representations for serialization or analysis with different
//! data types.
//!
//! # Features
//! - Generic conversion to any node and edge data types with default values
//! - Specialized conversion to string-based node and edge data
//! - Preservation of graph structure, weights, and attributes
//!
//! # Examples
//!
//! Converting a graph to use different data types:
//! ```rust
//! use xgraph::graph::graph::Graph;
//! use xgraph::graph::conversion::GraphConversion;
//!
//! let mut original = Graph::<u32, (), ()>::new(false);
//! let n1 = original.add_node(());
//! let n2 = original.add_node(());
//! original.add_edge(n1, n2, 5, ()).unwrap();
//!
//! let converted: Graph<u32, String, i32> = original.convert_to().unwrap();
//! assert_eq!(converted.nodes.len(), 2);
//! assert_eq!(converted.get_all_edges()[0].2, 5); // Weight preserved
//! ```
//!
//! Converting to a string-based graph:
//! ```rust
//! use xgraph::graph::graph::Graph;
//! use xgraph::graph::conversion::GraphConversion;
//!
//! let mut original = Graph::<u32, i32, bool>::new(true);
//! let n1 = original.add_node(1);
//! let n2 = original.add_node(2);
//! original.add_edge(n1, n2, 10, true).unwrap();
//!
//! let string_graph = original.to_string_graph().unwrap();
//! assert_eq!(string_graph.nodes.get(n1).unwrap().data, "0"); // Node data is node ID as string
//! assert_eq!(string_graph.get_all_edges()[0].3, ""); // Edge data is empty string
//! ```

use crate::graph::graph::Graph;
use std::hash::Hash;

/// Error type for graph conversion operations.
///
/// Represents errors that may occur during the conversion of a graph's node and edge data types.
#[derive(Debug)]
pub enum GraphConversionError {
    /// Indicates that an edge could not be added during conversion due to invalid node indices.
    EdgeAdditionFailed(usize, usize),
    /// Indicates that setting a node attribute failed during conversion.
    NodeAttributeError(usize, String),
    /// Indicates that setting an edge attribute failed during conversion.
    EdgeAttributeError(usize, usize, String),
}

impl std::fmt::Display for GraphConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphConversionError::EdgeAdditionFailed(from, to) => {
                write!(
                    f,
                    "Failed to add edge from node {} to node {} during conversion",
                    from, to
                )
            }
            GraphConversionError::NodeAttributeError(id, key) => {
                write!(
                    f,
                    "Failed to set attribute '{}' for node {} during conversion",
                    key, id
                )
            }
            GraphConversionError::EdgeAttributeError(from, to, key) => {
                write!(
                    f,
                    "Failed to set attribute '{}' for edge from {} to {} during conversion",
                    key, from, to
                )
            }
        }
    }
}

impl std::error::Error for GraphConversionError {}

/// Trait for converting a graph between different node and edge data types.
///
/// Provides methods to transform a `Graph<W, N, E>` into a new graph with different node (`N2`) and
/// edge (`E2`) types, or specifically into a graph with `String` as the data type for both nodes
/// and edges.
///
/// # Type Parameters
/// - `W`: The weight type of the graph edges (e.g., `u32`, `f64`).
/// - `N`: The original node data type.
/// - `E`: The original edge data type.
///
/// # Requirements
/// - The weight type `W` must implement `Copy`, `PartialEq`, and `Default`.
/// - The original node type `N` must implement `Clone`, `Eq`, `Hash`, and `Debug`.
/// - The original edge type `E` must implement `Clone`, `Debug`, and `Default`.
///
/// # Examples
/// ```rust
/// use xgraph::graph::graph::Graph;
/// use xgraph::graph::conversion::GraphConversion;
///
/// let mut graph = Graph::<u32, (), ()>::new(false);
/// let n1 = graph.add_node(());
/// let n2 = graph.add_node(());
/// graph.add_edge(n1, n2, 1, ()).unwrap();
/// let converted: Graph<u32, String, String> = graph.convert_to().unwrap();
/// assert_eq!(converted.nodes.len(), 2);
/// ```
pub trait GraphConversion<W, N, E> {
    /// Converts the graph into a new graph with specified node and edge types.
    ///
    /// Creates a new graph with the same structure (nodes, edges, weights, and directionality),
    /// but with node data of type `N2` and edge data of type `E2`. Node and edge data are initialized
    /// using their default values, while attributes are preserved.
    ///
    /// # Type Parameters
    /// - `N2`: The new node data type, must implement `Clone`, `Eq`, `Hash`, `Debug`, and `Default`.
    /// - `E2`: The new edge data type, must implement `Clone`, `Debug`, and `Default`.
    ///
    /// # Returns
    /// - `Ok(Graph<W, N2, E2>)`: The converted graph on success.
    /// - `Err(GraphConversionError)`: If an edge cannot be added or an attribute cannot be set.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::graph::conversion::GraphConversion;
    ///
    /// let mut graph = Graph::<u32, i32, bool>::new(true);
    /// let n1 = graph.add_node(42);
    /// let n2 = graph.add_node(84);
    /// graph.add_edge(n1, n2, 7, true).unwrap();
    /// let converted: Graph<u32, String, i32> = graph.convert_to().unwrap();
    /// assert_eq!(converted.get_all_edges()[0].2, 7); // Weight preserved
    /// ```
    fn convert_to<N2, E2>(&self) -> Result<Graph<W, N2, E2>, GraphConversionError>
    where
        W: Copy + PartialEq + Default,
        N2: Clone + Eq + Hash + std::fmt::Debug + Default,
        E2: Clone + std::fmt::Debug + Default;

    /// Converts the graph into a graph with `String` node and edge data types.
    ///
    /// Transforms the graph into one where node data is the string representation of the node ID,
    /// and edge data is an empty string. Attributes are preserved from the original graph.
    ///
    /// # Returns
    /// - `Ok(Graph<W, String, String>)`: The converted string-based graph on success.
    /// - `Err(GraphConversionError)`: If an edge cannot be added or an attribute cannot be set.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::graph::conversion::GraphConversion;
    ///
    /// let mut graph = Graph::<u32, (), ()>::new(false);
    /// let n1 = graph.add_node(());
    /// let n2 = graph.add_node(());
    /// graph.add_edge(n1, n2, 3, ()).unwrap();
    /// let string_graph = graph.to_string_graph().unwrap();
    /// assert_eq!(string_graph.nodes.get(n1).unwrap().data, "0"); // Node data is node ID as string
    /// assert_eq!(string_graph.get_all_edges()[0].3, ""); // Edge data is empty string
    /// ```
    fn to_string_graph(&self) -> Result<Graph<W, String, String>, GraphConversionError>
    where
        W: Copy + PartialEq + Default;
}

impl<W, N, E> GraphConversion<W, N, E> for Graph<W, N, E>
where
    W: Copy + PartialEq + Default,
    N: Clone + Eq + Hash + std::fmt::Debug,
    E: Clone + std::fmt::Debug + Default,
{
    fn convert_to<N2, E2>(&self) -> Result<Graph<W, N2, E2>, GraphConversionError>
    where
        W: Copy + PartialEq + Default,
        N2: Clone + Eq + Hash + std::fmt::Debug + Default,
        E2: Clone + std::fmt::Debug + Default,
    {
        // Create a new graph with the same directionality
        let mut new_graph = Graph::<W, N2, E2>::new(self.directed);

        // Convert nodes: use default values for new node data, copy attributes
        for (_id, node) in self.nodes.iter() {
            let new_id = new_graph.add_node(N2::default());
            for (key, value) in &node.attributes {
                new_graph
                    .set_node_attribute(new_id, key.clone(), value.clone())
                    .map_err(|_| GraphConversionError::NodeAttributeError(new_id, key.clone()))?;
            }
        }

        // Convert edges: use default values for new edge data, copy attributes
        for (from, to, weight, _) in self.get_all_edges() {
            if new_graph.add_edge(from, to, weight, E2::default()).is_err() {
                return Err(GraphConversionError::EdgeAdditionFailed(from, to));
            }
            if let Some(attrs) = self.get_all_edge_attributes(from, to) {
                for (key, value) in attrs {
                    new_graph
                        .set_edge_attribute(from, to, key.clone(), value.clone())
                        .map_err(|_| {
                            GraphConversionError::EdgeAttributeError(from, to, key.clone())
                        })?;
                }
            }
        }

        Ok(new_graph)
    }

    fn to_string_graph(&self) -> Result<Graph<W, String, String>, GraphConversionError>
    where
        W: Copy + PartialEq + Default,
    {
        // Create a new graph with the same directionality
        let mut new_graph = Graph::<W, String, String>::new(self.directed);

        // Convert nodes: use node ID as string data, copy attributes
        for (id, node) in self.nodes.iter() {
            let new_id = new_graph.add_node(id.to_string());
            for (key, value) in &node.attributes {
                new_graph
                    .set_node_attribute(new_id, key.clone(), value.clone())
                    .map_err(|_| GraphConversionError::NodeAttributeError(new_id, key.clone()))?;
            }
        }

        // Convert edges: use empty string as edge data, copy attributes
        for (from, to, weight, _) in self.get_all_edges() {
            if new_graph
                .add_edge(from, to, weight, "".to_string())
                .is_err()
            {
                return Err(GraphConversionError::EdgeAdditionFailed(from, to));
            }
            if let Some(attrs) = self.get_all_edge_attributes(from, to) {
                for (key, value) in attrs {
                    new_graph
                        .set_edge_attribute(from, to, key.clone(), value.clone())
                        .map_err(|_| {
                            GraphConversionError::EdgeAttributeError(from, to, key.clone())
                        })?;
                }
            }
        }

        Ok(new_graph)
    }
}
