//! Module for finding dominating sets in graphs using greedy heuristics
//!
//! Provides a trait and implementation for calculating minimum dominating sets -
//! subsets of vertices where every vertex is either in the set or adjacent to a set member.
//!
//! # Examples
//!
//! Basic usage:
//!
//! ```
//! use xgraph::graph::graph::Graph;
//! use xgraph::algorithms::wiedemann_ford::DominatingSetFinder;
//!
//! // Create a simple triangle graph
//! let adjacency_matrix = vec![
//!     vec![0, 1, 1],
//!     vec![1, 0, 1],
//!     vec![1, 1, 0],
//! ];
//!
//! let graph = Graph::from_adjacency_matrix(&adjacency_matrix, false, (), ()).unwrap();
//! let dominating_set = graph.find_dominating_set();
//!
//! // In a complete graph of 3 nodes, any single node forms a dominating set
//! assert!(dominating_set.len() == 1);
//! ```
//!
//! Real-world scenario with disconnected components:
//!
//! ```
//! use xgraph::graph::graph::Graph;
//! use xgraph::algorithms::wiedemann_ford::DominatingSetFinder;
//! // Create graph with two disconnected triangles
//! let adjacency_matrix = vec![
//!     vec![0,1,1,0,0,0],
//!     vec![1,0,1,0,0,0],
//!     vec![1,1,0,0,0,0],
//!     vec![0,0,0,0,1,1],
//!     vec![0,0,0,1,0,1],
//!     vec![0,0,0,1,1,0],
//! ];
//!
//! let graph = Graph::from_adjacency_matrix(&adjacency_matrix, false, (), ()).unwrap();
//! let ds = graph.find_dominating_set();
//!
//! // Should need 2 nodes - one from each triangle
//! assert_eq!(ds.len(), 2);
//! ```

use crate::graph::graph::Graph;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

/// A trait that defines a method for finding a dominating set in a graph.
/// The dominating set is a subset of vertices such that every vertex in the graph
/// is either in the set or adjacent to at least one member of the set.
pub trait DominatingSetFinder {
    /// Finds a dominating set in the graph and returns it as a `HashSet` containing node indices.
    ///
    /// # Example
    /// ```
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::wiedemann_ford::DominatingSetFinder;
    ///
    /// // Create a star graph with 4 nodes
    /// let adjacency_matrix = vec![
    ///     vec![0, 1, 1, 1],
    ///     vec![1, 0, 0, 0],
    ///     vec![1, 0, 0, 0],
    ///     vec![1, 0, 0, 0],
    /// ];
    /// let graph = Graph::from_adjacency_matrix(&adjacency_matrix, false, (), ()).unwrap();
    /// let dominating_set = graph.find_dominating_set();
    ///
    /// // The center node (0) should cover all other nodes
    /// assert!(dominating_set.len() == 1 && dominating_set.contains(&0));
    /// ```
    fn find_dominating_set(&self) -> HashSet<usize>;
}

impl<W, N, E> DominatingSetFinder for Graph<W, N, E>
where
    W: Copy + Default + PartialEq,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    /// Finds a dominating set using a greedy heuristic algorithm that selects nodes with the highest degree first.
    ///
    /// This implementation:
    /// 1. Sorts nodes by degree in descending order
    /// 2. Iteratively selects nodes that cover the maximum number of uncovered vertices
    /// 3. Stops when all vertices are covered
    ///
    /// # Example
    /// ```
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::wiedemann_ford::DominatingSetFinder;
    ///
    /// // Create a bipartite graph
    /// let adjacency_matrix = vec![
    ///     vec![0, 0, 1, 1],
    ///     vec![0, 0, 1, 1],
    ///     vec![1, 1, 0, 0],
    ///     vec![1, 1, 0, 0],
    /// ];
    /// let graph = Graph::from_adjacency_matrix(&adjacency_matrix, false, (), ()).unwrap();
    /// let dominating_set = graph.find_dominating_set();
    ///
    /// // The dominating set should contain nodes from both partitions
    /// assert!(dominating_set.len() == 2);
    /// ```
    fn find_dominating_set(&self) -> HashSet<usize> {
        let mut dominating_set = HashSet::new();
        let mut covered = HashSet::new();

        // Sort nodes by degree (descending) using a key that subtracts from MAX to reverse order
        let mut nodes: Vec<_> = self.all_nodes().map(|(id, _)| id).collect();
        nodes.sort_by_cached_key(|&node| usize::MAX - self.get_neighbors(node).len());

        for node in nodes {
            if !covered.contains(&node) {
                dominating_set.insert(node);
                covered.insert(node);
                for (neighbor, _) in self.get_neighbors(node) {
                    covered.insert(neighbor);
                }
            }
        }
        dominating_set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to create a graph from an adjacency matrix.
    fn create_test_graph(matrix: Vec<Vec<u32>>) -> Graph<u32, (), ()> {
        Graph::from_adjacency_matrix(&matrix, false, (), ()).expect("Failed to create test graph")
    }

    /// Test the dominating set finding on a simple graph.
    #[test]
    fn test_simple_dominating_set() {
        let matrix = vec![
            vec![0, 1, 1, 0],
            vec![1, 0, 1, 1],
            vec![1, 1, 0, 0],
            vec![0, 1, 0, 0],
        ];
        let graph = create_test_graph(matrix);
        let ds = graph.find_dominating_set();
        assert!(!ds.is_empty(), "Dominating set should not be empty");
        assert!(ds.len() <= 2, "Dominating set size should be at most 2");
    }

    /// Test the dominating set finding on a more complex graph.
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
        let ds = graph.find_dominating_set();
        assert!(!ds.is_empty(), "Dominating set should not be empty");
        assert!(ds.len() <= 3, "Dominating set size should be at most 3");
    }
}
