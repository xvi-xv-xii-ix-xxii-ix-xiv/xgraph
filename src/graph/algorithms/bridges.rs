//! Bridge detection algorithms for graphs.
//!
//! This module provides functionality to identify bridge edges in a graph using Tarjan's algorithm.
//! A bridge is an edge whose removal increases the number of connected components in the graph.
//! This implementation is independent of the `hgraph` feature and works with the basic `Graph` type.
//!
//! For heterogeneous multigraph support, enable the `hgraph` feature in your `Cargo.toml`:
//! ```toml
//! [dependencies.xgraph]
//! version = "x.y.z"
//! features = ["hgraph"]
//! ```
//! Then use `hgraph::h_bridges::HeteroBridges` instead.
//!
//! # Examples
//!
//! Finding bridges in a simple graph:
//! ```rust
//! use xgraph::graph::graph::Graph;
//! use xgraph::graph::algorithms::bridges::Bridges;
//!
//! let mut graph = Graph::<f64, String, String>::new(false);
//! let n0 = graph.add_node("A".to_string());
//! let n1 = graph.add_node("B".to_string());
//! let n2 = graph.add_node("C".to_string());
//!
//! graph.add_edge(n0, n1, 1.0, "road".to_string()).unwrap();
//! graph.add_edge(n1, n2, 2.0, "bridge".to_string()).unwrap();
//!
//! let bridges = graph.find_bridges().unwrap();
//! assert_eq!(bridges, vec![(n0, n1), (n1, n2)]);
//! ```
//!
//! No bridges in a cycle:
//! ```rust
//! use xgraph::graph::graph::Graph;
//! use xgraph::graph::algorithms::bridges::Bridges;
//!
//! let mut graph = Graph::<f64, String, String>::new(false);
//! let n0 = graph.add_node("A".to_string());
//! let n1 = graph.add_node("B".to_string());
//! let n2 = graph.add_node("C".to_string());
//!
//! graph.add_edge(n0, n1, 1.0, "road".to_string()).unwrap();
//! graph.add_edge(n1, n2, 2.0, "bridge".to_string()).unwrap();
//! graph.add_edge(n2, n0, 3.0, "path".to_string()).unwrap();
//!
//! let bridges = graph.find_bridges().unwrap();
//! assert!(bridges.is_empty());
//! ```

use crate::graph::graph::Graph;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

/// Error type for bridge detection failures.
#[derive(Debug)]
pub enum BridgeError {
    /// Indicates an invalid node was encountered during bridge detection.
    InvalidNode(usize),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeError::InvalidNode(id) => {
                write!(f, "Invalid node reference: node ID {} not found", id)
            }
        }
    }
}

impl std::error::Error for BridgeError {}

/// Context for Tarjan's bridge-finding algorithm.
///
/// This structure maintains the state during the depth-first search (DFS) process
/// used to identify bridges in a graph.
///
/// # Fields
/// - `visited`: Set of nodes that have been visited during DFS
/// - `disc`: Discovery times for each node
/// - `low`: Lowest reachable vertex times
/// - `parent`: Parent nodes in the DFS tree
/// - `bridges`: Collection of identified bridge edges
/// - `time`: Current time step in the DFS process
#[derive(Debug)]
pub struct BridgeContext {
    visited: HashSet<usize>,
    disc: HashMap<usize, u32>,
    low: HashMap<usize, u32>,
    parent: HashMap<usize, usize>,
    bridges: Vec<(usize, usize)>,
    time: u32,
}

impl BridgeContext {
    /// Creates a new empty context for bridge detection.
    ///
    /// # Returns
    /// A new `BridgeContext` instance with all fields initialized to their default empty states.
    pub fn new() -> Self {
        Self {
            visited: HashSet::new(),
            disc: HashMap::new(),
            low: HashMap::new(),
            parent: HashMap::new(),
            bridges: Vec::new(),
            time: 0,
        }
    }
}

impl Default for BridgeContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait defining bridge detection functionality for graphs.
///
/// This trait provides methods to identify bridge edges in undirected graphs
/// using Tarjan's algorithm with O(V + E) time complexity.
///
/// # Requirements
/// Implementing types must support graph operations and maintain node/edge relationships.
pub trait Bridges {
    /// Finds all bridge edges in the graph.
    ///
    /// # Returns
    /// A `Result` containing either:
    /// - `Ok(Vec<(usize, usize)>)`: A sorted vector of tuples representing bridge edges
    /// - `Err(BridgeError)`: An error if the computation fails
    fn find_bridges(&self) -> Result<Vec<(usize, usize)>, BridgeError>;

    /// Sorts bridge edges for consistent output.
    ///
    /// Normalizes node order (smaller ID first) and sorts lexicographically.
    ///
    /// # Arguments
    /// - `bridges`: Mutable reference to the vector of bridge edges to be sorted
    fn sort_bridges(bridges: &mut Vec<(usize, usize)>);

    /// Performs depth-first search to detect bridges.
    ///
    /// # Arguments
    /// - `node`: The current node being processed
    /// - `context`: Mutable reference to the bridge detection context
    ///
    /// # Returns
    /// A `Result` indicating success or failure of the DFS operation
    fn bridge_dfs(&self, node: usize, context: &mut BridgeContext) -> Result<(), BridgeError>;
}

impl<W, N, E> Bridges for Graph<W, N, E>
where
    W: Copy + Default + PartialEq,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    fn find_bridges(&self) -> Result<Vec<(usize, usize)>, BridgeError> {
        let mut context = BridgeContext::new();

        // Iterate over all nodes to handle disconnected components
        for node in 0..self.nodes.len() {
            if !context.visited.contains(&node) && self.nodes.contains(node) {
                self.bridge_dfs(node, &mut context)?;
            }
        }

        Self::sort_bridges(&mut context.bridges);
        Ok(context.bridges)
    }

    fn sort_bridges(bridges: &mut Vec<(usize, usize)>) {
        // Normalize node order (smaller ID first) and sort
        bridges.iter_mut().for_each(|(u, v)| {
            if u > v {
                std::mem::swap(u, v);
            }
        });
        bridges.sort_unstable();
    }

    fn bridge_dfs(&self, node: usize, context: &mut BridgeContext) -> Result<(), BridgeError> {
        context.time += 1;
        context.disc.insert(node, context.time);
        context.low.insert(node, context.time);
        context.visited.insert(node);

        // Check if node exists and get its neighbors
        let neighbors = self.nodes.get(node).ok_or(BridgeError::InvalidNode(node))?;

        for &(neighbor, _) in &neighbors.neighbors {
            if !context.visited.contains(&neighbor) {
                context.parent.insert(neighbor, node);
                self.bridge_dfs(neighbor, context)?;

                let node_low = *context.low.get(&node).unwrap();
                let neighbor_low = *context.low.get(&neighbor).unwrap();
                context.low.insert(node, node_low.min(neighbor_low));

                // Check if this edge is a bridge
                if *context.low.get(&neighbor).unwrap() > *context.disc.get(&node).unwrap() {
                    context.bridges.push((node, neighbor));
                }
            } else if context.parent.get(&node) != Some(&neighbor) {
                let node_low = *context.low.get(&node).unwrap();
                let neighbor_disc = *context.disc.get(&neighbor).unwrap();
                context.low.insert(node, node_low.min(neighbor_disc));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::graph::Graph;

    #[test]
    fn test_find_bridges_simple() {
        let mut graph = Graph::<f64, String, String>::new(false);
        let n0 = graph.add_node("A".to_string());
        let n1 = graph.add_node("B".to_string());
        let n2 = graph.add_node("C".to_string());

        graph.add_edge(n0, n1, 1.0, "road".to_string()).unwrap();
        graph.add_edge(n1, n2, 2.0, "bridge".to_string()).unwrap();

        let bridges = graph.find_bridges().unwrap();
        assert_eq!(bridges, vec![(n0, n1), (n1, n2)]);
    }

    #[test]
    fn test_find_bridges_cycle() {
        let mut graph = Graph::<f64, String, String>::new(false);
        let n0 = graph.add_node("A".to_string());
        let n1 = graph.add_node("B".to_string());
        let n2 = graph.add_node("C".to_string());

        graph.add_edge(n0, n1, 1.0, "road".to_string()).unwrap();
        graph.add_edge(n1, n2, 2.0, "bridge".to_string()).unwrap();
        graph.add_edge(n2, n0, 3.0, "path".to_string()).unwrap();

        let bridges = graph.find_bridges().unwrap();
        assert!(bridges.is_empty());
    }

    #[test]
    fn test_find_bridges_complex() {
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

        let graph =
            Graph::from_adjacency_matrix(&matrix, false, "node".to_string(), "edge".to_string())
                .unwrap();
        let bridges = graph.find_bridges().unwrap();
        assert_eq!(bridges, vec![(2, 3), (8, 9)]);
    }

    #[test]
    fn test_find_bridges_empty() {
        let graph = Graph::<f64, String, String>::new(false);
        let bridges = graph.find_bridges().unwrap();
        assert!(bridges.is_empty());
    }

    #[test]
    fn test_find_bridges_single_node() {
        let mut graph = Graph::<f64, String, String>::new(false);
        graph.add_node("A".to_string());
        let bridges = graph.find_bridges().unwrap();
        assert!(bridges.is_empty());
    }
}
