//! Graph connectivity analysis algorithms
//!
//! This module provides functionality to analyze graph connectivity, including:
//! - Weakly connected components (for undirected graphs or when ignoring direction)
//! - Strongly connected components (for directed graphs using Kosaraju's algorithm)
//! - Overall graph connectivity checks
//!
//! All methods return `Result` types to handle potential errors gracefully instead of panicking.
//! This implementation works with the basic `Graph` type and is independent of specific features.
//!
//! # Features
//! - Weak connectivity analysis via BFS
//! - Strong connectivity analysis via Kosaraju's algorithm
//! - Flexible connectivity checks based on graph directionality
//!
//! # Examples
//!
//! Basic usage with error handling:
//! ```rust
//! use xgraph::graph::graph::Graph;
//! use xgraph::graph::algorithms::connectivity::Connectivity;
//!
//! let mut graph = Graph::<u32, &str, ()>::new(true);
//! let a = graph.add_node("A");
//! let b = graph.add_node("B");
//! graph.add_edge(a, b, 1, ()).unwrap();
//!
//! let scc = graph.find_strongly_connected_components().unwrap();
//! let is_strong = graph.is_strongly_connected().unwrap();
//! ```

use crate::graph::graph::Graph;
use crate::graph::node::Node;
use std::collections::{HashSet, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;

/// Error type for connectivity analysis failures.
#[derive(Debug)]
pub enum ConnectivityError {
    /// Indicates an invalid node was encountered during computation.
    InvalidNode(usize),
    /// Indicates an error during graph transposition.
    TransposeError(String),
}

impl std::fmt::Display for ConnectivityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectivityError::InvalidNode(id) => {
                write!(f, "Invalid node reference: node ID {} not found", id)
            }
            ConnectivityError::TransposeError(msg) => {
                write!(f, "Graph transposition failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for ConnectivityError {}

/// Trait for graph connectivity analysis.
///
/// Provides methods to analyze both weak and strong connectivity in directed and undirected graphs
/// with proper error handling.
///
/// # Type Parameters
/// - `W`: The weight type of the graph edges.
/// - `N`: The node data type.
/// - `E`: The edge data type.
///
/// # Examples
///
/// Finding connected components in a social network:
/// ```rust
/// use xgraph::graph::graph::Graph;
/// use xgraph::graph::algorithms::connectivity::Connectivity;
///
/// let mut social_graph = Graph::new(false);
/// let alice = social_graph.add_node("Alice");
/// let bob = social_graph.add_node("Bob");
/// let charlie = social_graph.add_node("Charlie");
///
/// social_graph.add_edge(alice, bob, 1, ()).unwrap();
/// social_graph.add_edge(charlie, bob, 1, ()).unwrap();
///
/// let components = social_graph.find_connected_components().unwrap();
/// assert_eq!(components.len(), 1);
/// ```
pub trait Connectivity<W, N, E> {
    /// Finds weakly connected components using BFS.
    ///
    /// Treats the graph as undirected by considering all edge connections regardless of direction.
    ///
    /// # Returns
    /// - `Ok(Vec<Vec<usize>>)`: Vector of vectors containing node IDs for each component.
    /// - `Err(ConnectivityError)`: If computation fails due to invalid nodes.
    fn find_weakly_connected_components(&self) -> Result<Vec<Vec<usize>>, ConnectivityError>;

    /// Finds strongly connected components using Kosaraju's algorithm.
    ///
    /// Implements a two-phase algorithm with O(|V| + |E|) complexity:
    /// 1. DFS to determine finish order
    /// 2. DFS on transposed graph in reverse finish order
    ///
    /// # Returns
    /// - `Ok(Vec<Vec<usize>>)`: Vector of vectors containing node IDs for each component.
    /// - `Err(ConnectivityError)`: If computation fails due to invalid nodes or transposition errors.
    fn find_strongly_connected_components(&self) -> Result<Vec<Vec<usize>>, ConnectivityError>;

    /// Automatically selects appropriate connectivity type based on graph directionality.
    ///
    /// Calls `find_strongly_connected_components` for directed graphs and
    /// `find_weakly_connected_components` for undirected graphs.
    ///
    /// # Returns
    /// - `Ok(Vec<Vec<usize>>)`: Vector of vectors containing node IDs for each component.
    /// - `Err(ConnectivityError)`: If computation fails.
    fn find_connected_components(&self) -> Result<Vec<Vec<usize>>, ConnectivityError> {
        if self.is_directed().unwrap_or(false) {
            self.find_strongly_connected_components()
        } else {
            self.find_weakly_connected_components()
        }
    }

    /// Checks if the graph is weakly connected using BFS.
    ///
    /// A graph is weakly connected if there’s a path between every pair of nodes when considering
    /// all edges as undirected.
    ///
    /// # Returns
    /// - `Ok(bool)`: True if the graph is weakly connected, false otherwise.
    /// - `Err(ConnectivityError)`: If computation fails.
    fn is_weakly_connected(&self) -> Result<bool, ConnectivityError>;

    /// Checks if the graph is strongly connected using Kosaraju's algorithm.
    ///
    /// A graph is strongly connected if there’s a directed path between every pair of nodes.
    ///
    /// # Returns
    /// - `Ok(bool)`: True if the graph is strongly connected, false otherwise.
    /// - `Err(ConnectivityError)`: If computation fails.
    fn is_strongly_connected(&self) -> Result<bool, ConnectivityError>;

    /// Checks overall connectivity based on graph type.
    ///
    /// Returns strong connectivity for directed graphs and weak connectivity for undirected graphs.
    ///
    /// # Returns
    /// - `Ok(bool)`: True if the graph is connected according to its type, false otherwise.
    /// - `Err(ConnectivityError)`: If computation fails.
    fn is_connected(&self) -> Result<bool, ConnectivityError> {
        if self.is_directed().unwrap_or(false) {
            self.is_strongly_connected()
        } else {
            self.is_weakly_connected()
        }
    }

    /// Determines if the graph is directed.
    ///
    /// # Returns
    /// - `Ok(bool)`: True if the graph is directed, false if undirected.
    /// - `Err(ConnectivityError)`: If the graph state is invalid (currently never returned).
    fn is_directed(&self) -> Result<bool, ConnectivityError>;
}

impl<W, N, E> Connectivity<W, N, E> for Graph<W, N, E>
where
    W: Copy + Default + PartialEq + Debug,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    fn find_weakly_connected_components(&self) -> Result<Vec<Vec<usize>>, ConnectivityError> {
        let mut visited = HashSet::new();
        let mut components = Vec::new();

        for node in self.nodes.iter().map(|(id, _)| id) {
            if !visited.contains(&node) {
                let mut component = Vec::new();
                let mut queue = VecDeque::new();
                queue.push_back(node);
                visited.insert(node);

                while let Some(current) = queue.pop_front() {
                    component.push(current);

                    let node_data = self
                        .nodes
                        .get(current)
                        .ok_or(ConnectivityError::InvalidNode(current))?;
                    let mut neighbors = node_data
                        .neighbors
                        .iter()
                        .map(|(n, _)| *n)
                        .collect::<Vec<_>>();

                    if self.is_directed().unwrap_or(false) {
                        neighbors.extend(self.get_predecessors(current));
                    }

                    for neighbor in neighbors {
                        if !visited.contains(&neighbor) {
                            if !self.nodes.contains(neighbor) {
                                return Err(ConnectivityError::InvalidNode(neighbor));
                            }
                            visited.insert(neighbor);
                            queue.push_back(neighbor);
                        }
                    }
                }
                components.push(component);
            }
        }
        Ok(components)
    }

    fn find_strongly_connected_components(&self) -> Result<Vec<Vec<usize>>, ConnectivityError> {
        let mut visited = HashSet::new();
        let mut order = Vec::with_capacity(self.nodes.len());

        for node in self.nodes.iter().map(|(id, _)| id) {
            if !visited.contains(&node) {
                self.dfs_order(node, &mut visited, &mut order)?;
            }
        }

        let transposed = self.transpose()?;
        let mut visited = HashSet::new();
        let mut components = Vec::new();

        for &node in order.iter().rev() {
            if !visited.contains(&node) {
                let mut component = Vec::new();
                transposed.dfs_collect(node, &mut visited, &mut component)?;
                component.sort();
                components.push(component);
            }
        }

        Ok(components)
    }

    fn is_weakly_connected(&self) -> Result<bool, ConnectivityError> {
        let components = self.find_weakly_connected_components()?;
        Ok(components.len() == 1)
    }

    fn is_strongly_connected(&self) -> Result<bool, ConnectivityError> {
        if self.nodes.is_empty() {
            return Ok(true);
        }

        let start_node = self.nodes.iter().next().unwrap().0;
        self.strong_connectivity_check(start_node)
    }

    fn is_directed(&self) -> Result<bool, ConnectivityError> {
        Ok(self.directed)
    }
}

impl<W, N, E> Graph<W, N, E>
where
    W: Copy + Default + PartialEq + Debug,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    /// Performs DFS to determine the finish order of nodes for Kosaraju's algorithm.
    fn dfs_order(
        &self,
        node: usize,
        visited: &mut HashSet<usize>,
        order: &mut Vec<usize>,
    ) -> Result<(), ConnectivityError> {
        visited.insert(node);
        let neighbors = self
            .nodes
            .get(node)
            .ok_or(ConnectivityError::InvalidNode(node))?
            .neighbors
            .iter();
        for (neighbor, _) in neighbors {
            if !visited.contains(neighbor) {
                self.dfs_order(*neighbor, visited, order)?;
            }
        }
        order.push(node);
        Ok(())
    }

    /// Performs DFS to collect nodes of a strongly connected component.
    fn dfs_collect(
        &self,
        node: usize,
        visited: &mut HashSet<usize>,
        component: &mut Vec<usize>,
    ) -> Result<(), ConnectivityError> {
        visited.insert(node);
        component.push(node);
        let neighbors = self
            .nodes
            .get(node)
            .ok_or(ConnectivityError::InvalidNode(node))?
            .neighbors
            .iter();
        for (neighbor, _) in neighbors {
            if !visited.contains(neighbor) {
                self.dfs_collect(*neighbor, visited, component)?;
            }
        }
        Ok(())
    }

    /// Checks strong connectivity starting from a given node.
    fn strong_connectivity_check(&self, start: usize) -> Result<bool, ConnectivityError> {
        let mut forward_visited = HashSet::new();
        self.dfs_collect(start, &mut forward_visited, &mut vec![])?;

        if forward_visited.len() != self.nodes.len() {
            return Ok(false);
        }

        let transposed = self.transpose()?;
        let mut backward_visited = HashSet::new();
        transposed.dfs_collect(start, &mut backward_visited, &mut vec![])?;

        Ok(backward_visited.len() == self.nodes.len())
    }

    /// Retrieves the predecessors of a node (used for directed graphs).
    fn get_predecessors(&self, node: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|(_, e)| e.to == node)
            .map(|(_, e)| e.from)
            .collect()
    }

    /// Transposes the graph by reversing all edge directions.
    fn transpose(&self) -> Result<Self, ConnectivityError> {
        let mut transposed = Graph::new(true);
        let mut nodes: Vec<_> = self.nodes.iter().map(|(id, _)| id).collect();
        nodes.sort();

        for &id in &nodes {
            let node = self
                .nodes
                .get(id)
                .ok_or(ConnectivityError::InvalidNode(id))?;
            transposed.nodes.insert(Node {
                data: node.data.clone(),
                neighbors: Vec::new(),
                attributes: node.attributes.clone(),
            });
        }

        for (_, edge) in self.edges.iter() {
            transposed
                .add_edge(edge.to, edge.from, edge.weight, edge.data.clone())
                .map_err(|e| ConnectivityError::TransposeError(e.to_string()))?;

            if let Some(attrs) = self.get_all_edge_attributes(edge.from, edge.to) {
                for (k, v) in attrs {
                    transposed
                        .set_edge_attribute(edge.to, edge.from, k.clone(), v.clone())
                        .map_err(|e| ConnectivityError::TransposeError(e.to_string()))?;
                }
            }
        }

        Ok(transposed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strongly_connected() {
        let mut graph = Graph::<u32, (), ()>::new(true);

        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        graph.add_edge(n0, n1, 1, ()).unwrap();
        graph.add_edge(n1, n2, 1, ()).unwrap();
        graph.add_edge(n2, n0, 1, ()).unwrap();

        assert!(graph.is_strongly_connected().unwrap());
        let scc = graph.find_strongly_connected_components().unwrap();
        assert_eq!(scc.len(), 1);
    }

    #[test]
    fn test_weak_connectivity() {
        let mut graph = Graph::<u32, (), ()>::new(true);

        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        graph.add_edge(n0, n1, 1, ()).unwrap();

        assert!(!graph.is_strongly_connected().unwrap());
        assert!(graph.is_weakly_connected().unwrap());
    }

    #[test]
    fn test_transpose() {
        let mut graph = Graph::<u32, (), ()>::new(true);
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        graph.add_edge(n0, n1, 1, ()).unwrap();

        let transposed = graph.transpose().unwrap();
        assert_eq!(transposed.get_neighbors(n1), vec![(n0, 1)]);
    }
}
