//! Module for converting heterogeneous multigraphs between different node and edge types.
//!
//! This module provides functionality to transform a `HeterogeneousGraph` into another graph
//! with different node and edge data types. It supports:
//! - Generic conversion to a graph with new node (`N2`) and edge (`E2`) types using defaults.
//! - Conversion to a string-based graph where node and edge data are represented as strings.
//!
//! The module is available only when the `hgraph` feature is enabled in your `Cargo.toml`:
//! ```toml
//! [dependencies.xgraph]
//! version = "x.y.z"
//! features = ["hgraph"]
//! ```

#[cfg(feature = "hgraph")]
use crate::hgraph::h_graph::HeterogeneousGraph;
#[cfg(feature = "hgraph")]
use std::hash::Hash;

// Error handling additions

/// Error type for graph conversion failures.
///
/// Represents errors that may occur during graph conversion operations, such as edge addition failures.
#[cfg(feature = "hgraph")]
#[derive(Debug)]
pub enum GraphConversionError {
    /// Indicates an error occurred while adding an edge to the new graph.
    EdgeAdditionError {
        /// The source node ID.
        from: usize,
        /// The target node ID.
        to: usize,
        /// The error message.
        message: String,
    },
}

#[cfg(feature = "hgraph")]
impl std::fmt::Display for GraphConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphConversionError::EdgeAdditionError { from, to, message } => {
                write!(f, "Failed to add edge from {} to {}: {}", from, to, message)
            }
        }
    }
}

#[cfg(feature = "hgraph")]
impl std::error::Error for GraphConversionError {}

/// Result type alias for graph conversion operations.
///
/// Wraps the result of conversion methods to enable error handling without panicking.
#[cfg(feature = "hgraph")]
pub type Result<T> = std::result::Result<T, GraphConversionError>;

#[cfg(feature = "hgraph")]
/// Trait for converting heterogeneous multigraphs to different node and edge types.
///
/// Provides methods to transform the graph’s node and edge data into new types while preserving
/// structure, weights, and attributes. Methods return a `Result` to handle potential errors.
pub trait GraphConversion<W, N, E>
where
    W: Copy + PartialEq + Default + std::fmt::Debug,
    N: Clone + Eq + Hash + std::fmt::Debug + crate::hgraph::h_node::NodeType,
    E: Clone + std::fmt::Debug + Default + crate::hgraph::h_edge::EdgeType,
{
    /// Converts the graph to a new graph with specified node and edge types.
    ///
    /// Creates a new graph where nodes are initialized with `N2::default()` and edges with
    /// `E2::default()`, preserving the original structure, weights, and attributes.
    ///
    /// # Type Parameters
    /// * `N2` - The new node type, must implement `Default`.
    /// * `E2` - The new edge type, must implement `Default`.
    ///
    /// # Returns
    /// A `Result` containing the new `HeterogeneousGraph` with types `W`, `N2`, and `E2`.
    ///
    /// # Errors
    /// Returns `GraphConversionError::EdgeAdditionError` if adding an edge to the new graph fails.
    fn convert_to<N2, E2>(&self) -> Result<HeterogeneousGraph<W, N2, E2>>
    where
        W: Copy + PartialEq + Default + std::fmt::Debug,
        N2: Clone + Eq + Hash + std::fmt::Debug + Default + crate::hgraph::h_node::NodeType,
        E2: Clone + std::fmt::Debug + Default + crate::hgraph::h_edge::EdgeType;

    /// Converts the graph to a string-based representation.
    ///
    /// Creates a new graph where node and edge data are converted to strings using their
    /// `as_string()` methods, preserving weights and attributes.
    ///
    /// # Returns
    /// A `Result` containing the new `HeterogeneousGraph` with `String` node and edge types.
    ///
    /// # Errors
    /// Returns `GraphConversionError::EdgeAdditionError` if adding an edge to the new graph fails.
    fn to_string_graph(&self) -> Result<HeterogeneousGraph<W, String, String>>
    where
        W: Copy + PartialEq + Default + std::fmt::Debug,
        N: crate::hgraph::h_node::NodeType,
        E: crate::hgraph::h_edge::EdgeType;
}

#[cfg(feature = "hgraph")]
impl<W, N, E> GraphConversion<W, N, E> for HeterogeneousGraph<W, N, E>
where
    W: Copy + PartialEq + Default + std::fmt::Debug,
    N: Clone + Eq + Hash + std::fmt::Debug + crate::hgraph::h_node::NodeType,
    E: Clone + std::fmt::Debug + Default + crate::hgraph::h_edge::EdgeType,
{
    fn convert_to<N2, E2>(&self) -> Result<HeterogeneousGraph<W, N2, E2>>
    where
        W: Copy + PartialEq + Default + std::fmt::Debug,
        N2: Clone + Eq + Hash + std::fmt::Debug + Default + crate::hgraph::h_node::NodeType,
        E2: Clone + std::fmt::Debug + Default + crate::hgraph::h_edge::EdgeType,
    {
        let mut new_graph = HeterogeneousGraph::<W, N2, E2>::new(self.directed);

        // Copy nodes with default data and preserve attributes
        for (_id, node) in self.nodes.iter() {
            let new_id = new_graph.add_node(N2::default());
            for (key, value) in &node.attributes {
                new_graph.set_node_attribute(new_id, key.clone(), value.clone());
            }
        }

        // Copy edges with default data and preserve attributes
        for (from, to, weight, _edge_data) in self.get_all_edges() {
            new_graph
                .add_edge(from, to, weight, E2::default())
                .map_err(|e| GraphConversionError::EdgeAdditionError {
                    from,
                    to,
                    message: e.to_string(),
                })?;
            if let Some(attrs) = self.get_all_edge_attributes(from, to) {
                for (key, value) in attrs {
                    new_graph.set_edge_attribute(from, key.clone(), value.clone());
                }
            }
        }

        Ok(new_graph)
    }

    fn to_string_graph(&self) -> Result<HeterogeneousGraph<W, String, String>>
    where
        W: Copy + PartialEq + Default + std::fmt::Debug,
        N: crate::hgraph::h_node::NodeType,
        E: crate::hgraph::h_edge::EdgeType,
    {
        let mut new_graph = HeterogeneousGraph::<W, String, String>::new(self.directed);

        // Copy nodes as strings and preserve attributes
        for (_id, node) in self.nodes.iter() {
            let new_id = new_graph.add_node(node.data.as_string());
            for (key, value) in &node.attributes {
                new_graph.set_node_attribute(new_id, key.clone(), value.clone());
            }
        }

        // Copy edges as strings and preserve attributes
        for (from, to, weight, edge_data) in self.get_all_edges() {
            new_graph
                .add_edge(from, to, weight, edge_data.as_string())
                .map_err(|e| GraphConversionError::EdgeAdditionError {
                    from,
                    to,
                    message: e.to_string(),
                })?;
            if let Some(attrs) = self.get_all_edge_attributes(from, to) {
                for (key, value) in attrs {
                    new_graph.set_edge_attribute(from, key.clone(), value.clone());
                }
            }
        }

        Ok(new_graph)
    }
}
