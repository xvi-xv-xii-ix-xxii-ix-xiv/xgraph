//! Module for finding dominating sets in graphs using greedy heuristics
//!
//! This module provides functionality to compute minimum dominating sets in graphs—subsets of vertices
//! where every vertex in the graph is either in the set or adjacent to at least one vertex in the set.
//! It uses a greedy heuristic approach for efficiency, suitable for both directed and undirected graphs.
//!
//! # Features
//! - Greedy algorithm for finding dominating sets
//! - Support for graphs with varying node and edge types
//! - Comprehensive error handling for invalid graph states
//!
//! # Examples
//!
//! Basic usage with a simple triangle graph:
//! ```rust
//! use xgraph::graph::graph::Graph;
//! use xgraph::graph::algorithms::wiedemann_ford::DominatingSetFinder;
//!
//! let adjacency_matrix = vec![
//!     vec![0, 1, 1],
//!     vec![1, 0, 1],
//!     vec![1, 1, 0],
//! ];
//! let graph = Graph::from_adjacency_matrix(&adjacency_matrix, false, (), ()).unwrap();
//! let dominating_set = graph.find_dominating_set().unwrap();
//! assert!(dominating_set.len() == 1); // Any single node dominates a complete graph of 3
//! ```
//!
//! Real-world scenario with disconnected components:
//! ```rust
//! use xgraph::graph::graph::Graph;
//! use xgraph::graph::algorithms::wiedemann_ford::DominatingSetFinder;
//!
//! let adjacency_matrix = vec![
//!     vec![0, 1, 1, 0, 0, 0],
//!     vec![1, 0, 1, 0, 0, 0],
//!     vec![1, 1, 0, 0, 0, 0],
//!     vec![0, 0, 0, 0, 1, 1],
//!     vec![0, 0, 0, 1, 0, 1],
//!     vec![0, 0, 0, 1, 1, 0],
//! ];
//! let graph = Graph::from_adjacency_matrix(&adjacency_matrix, false, (), ()).unwrap();
//! let ds = graph.find_dominating_set().unwrap();
//! assert_eq!(ds.len(), 2); // One node from each triangle is needed
//! ```

use crate::graph::graph::Graph;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

/// Error type for dominating set computation failures.
///
/// Represents errors that may occur during the computation of a dominating set.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DominatingSetError {
    /// Indicates that a neighbor node referenced in the graph does not exist.
    InvalidNeighborNode(usize),
}

impl std::fmt::Display for DominatingSetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DominatingSetError::InvalidNeighborNode(id) => {
                write!(f, "Invalid neighbor node: node ID {} not found", id)
            }
        }
    }
}

impl std::error::Error for DominatingSetError {}

/// Trait for finding dominating sets in graphs.
///
/// Defines a method to compute a dominating set, ensuring every vertex is either in the set
/// or adjacent to a member of the set.
///
/// # Examples
/// Finding a dominating set in a star graph:
/// ```rust
/// use xgraph::graph::graph::Graph;
/// use xgraph::graph::algorithms::wiedemann_ford::DominatingSetFinder;
///
/// let adjacency_matrix = vec![
///     vec![0, 1, 1, 1],
///     vec![1, 0, 0, 0],
///     vec![1, 0, 0, 0],
///     vec![1, 0, 0, 0],
/// ];
/// let graph = Graph::from_adjacency_matrix(&adjacency_matrix, false, (), ()).unwrap();
/// let dominating_set = graph.find_dominating_set().unwrap();
/// assert!(dominating_set.len() == 1 && dominating_set.contains(&0)); // Center node dominates all
/// ```
pub trait DominatingSetFinder {
    /// Finds a dominating set in the graph.
    ///
    /// Returns a set of node indices such that every node in the graph is either in the set
    /// or adjacent to at least one node in the set.
    ///
    /// # Returns
    /// - `Ok(HashSet<usize>)`: A set of node indices forming a dominating set.
    /// - `Err(DominatingSetError)`: If an invalid neighbor node is encountered during computation.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::graph::algorithms::wiedemann_ford::DominatingSetFinder;
    ///
    /// let adjacency_matrix = vec![
    ///     vec![0, 0, 1, 1],
    ///     vec![0, 0, 1, 1],
    ///     vec![1, 1, 0, 0],
    ///     vec![1, 1, 0, 0],
    /// ];
    /// let graph = Graph::from_adjacency_matrix(&adjacency_matrix, false, (), ()).unwrap();
    /// let dominating_set = graph.find_dominating_set().unwrap();
    /// assert!(dominating_set.len() == 2); // Needs nodes from both partitions
    /// ```
    fn find_dominating_set(&self) -> Result<HashSet<usize>, DominatingSetError>;
}

impl<W, N, E> DominatingSetFinder for Graph<W, N, E>
where
    W: Copy + Default + PartialEq,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    fn find_dominating_set(&self) -> Result<HashSet<usize>, DominatingSetError> {
        let mut dominating_set = HashSet::new();
        let mut covered = HashSet::new();

        // Check neighbors for validity before sorting
        for node in self.all_nodes().map(|(id, _)| id) {
            for (neighbor, _) in self.get_neighbors(node) {
                if !self.nodes.contains(neighbor) {
                    return Err(DominatingSetError::InvalidNeighborNode(neighbor));
                }
            }
        }

        // Sort nodes by degree (descending) using a key that subtracts from MAX to reverse order
        let mut nodes: Vec<_> = self.all_nodes().map(|(id, _)| id).collect();
        nodes.sort_by_cached_key(|&node| usize::MAX - self.get_neighbors(node).len());

        for node in nodes {
            if !covered.contains(&node) {
                dominating_set.insert(node);
                covered.insert(node);
                for (neighbor, _) in self.get_neighbors(node) {
                    covered.insert(neighbor); // Already validated above
                }
            }
        }
        Ok(dominating_set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to create a graph from an adjacency matrix for testing.
    fn create_test_graph(matrix: Vec<Vec<u32>>) -> Graph<u32, (), ()> {
        Graph::from_adjacency_matrix(&matrix, false, (), ()).expect("Failed to create test graph")
    }

    /// Test the dominating set computation on a simple graph.
    #[test]
    fn test_simple_dominating_set() {
        let matrix = vec![
            vec![0, 1, 1, 0],
            vec![1, 0, 1, 1],
            vec![1, 1, 0, 0],
            vec![0, 1, 0, 0],
        ];
        let graph = create_test_graph(matrix);
        let ds = graph.find_dominating_set().unwrap();
        assert!(!ds.is_empty(), "Dominating set should not be empty");
        assert!(ds.len() <= 2, "Dominating set size should be at most 2");
    }

    /// Test the dominating set computation on a more complex graph.
    #[test]
    fn test_complex_dominating_set() {
        let matrix = vec![
            vec![0, 1, 0, 1, 0],
            vec![1, 0, 1, 0, 0],
            vec![0, 1, 0, 1, 1],
            vec![1, 0, 1, 0, 0],
            vec![0, 0, 1, 0, 0],
        ];
        let graph = create_test_graph(matrix);
        let ds = graph.find_dominating_set().unwrap();
        assert!(!ds.is_empty(), "Dominating set should not be empty");
        assert!(ds.len() <= 3, "Dominating set size should be at most 3");
    }
}
