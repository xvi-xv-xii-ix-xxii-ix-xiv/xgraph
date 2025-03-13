//! Graph search algorithms and operations
//!
//! This module provides functionality for performing various search operations on graphs,
//! including path finding, cycle detection, and node existence checks. It is designed to work
//! with both directed and undirected graphs, offering robust implementations of Depth-First
//! Search (DFS) and Breadth-First Search (BFS) algorithms.
//!
//! # Features
//! - Path existence checking using DFS
//! - Shortest path finding using BFS
//! - Cycle detection tailored for directed and undirected graphs
//! - Node existence verification
//!
//! # Examples
//!
//! Basic usage with a directed graph:
//! ```rust
//! use xgraph::graph::graph::Graph;
//! use xgraph::graph::algorithms::search::Search;
//!
//! let mut graph = Graph::<u32, &str, &str>::new(true);
//! let a = graph.add_node("A");
//! let b = graph.add_node("B");
//! graph.add_edge(a, b, 1, "link").unwrap();
//!
//! assert!(graph.has_path(a, b).unwrap());
//! assert_eq!(graph.bfs_path(a, b).unwrap(), Some(vec![a, b]));
//! ```

use crate::graph::graph::Graph;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;

/// Error type for search operation failures.
///
/// Represents errors that may occur during graph search operations.
#[derive(Debug)]
pub enum SearchError {
    /// Indicates that a node referenced in the search does not exist in the graph.
    InvalidNode(usize),
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchError::InvalidNode(id) => {
                write!(f, "Invalid node reference: node ID {} not found", id)
            }
        }
    }
}

impl std::error::Error for SearchError {}

/// Trait for graph search operations.
///
/// Provides fundamental algorithms for graph traversal and analysis with error handling.
///
/// # Type Parameters
/// - `W`: Edge weight type, must be copyable, have a default value, and support partial equality.
/// - `N`: Node data type, must be clonable, equatable, hashable, and debuggable.
/// - `E`: Edge data type, must be clonable and debuggable.
///
/// # Examples
/// ```rust
/// use xgraph::graph::graph::Graph;
/// use xgraph::graph::algorithms::search::Search;
///
/// let mut graph = Graph::<u32, u32, ()>::new(true);
/// let a = graph.add_node(0);
/// let b = graph.add_node(1);
/// graph.add_edge(a, b, 1, ()).unwrap();
///
/// assert!(graph.has_path(a, b).unwrap());
/// assert_eq!(graph.bfs_path(a, b).unwrap(), Some(vec![a, b]));
/// ```
pub trait Search<W, N, E>
where
    W: Copy + Default + PartialEq,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    /// Checks if a path exists between two nodes using DFS.
    ///
    /// Traverses the graph depth-first to determine if there is a valid path from `start` to `target`.
    ///
    /// # Arguments
    /// - `start`: The ID of the starting node.
    /// - `target`: The ID of the target node.
    ///
    /// # Returns
    /// - `Ok(bool)`: `true` if a path exists, `false` otherwise.
    /// - `Err(SearchError)`: If either `start` or `target` is not a valid node.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::graph::algorithms::search::Search;
    ///
    /// let mut graph = Graph::<u32, u32, ()>::new(true);
    /// let a = graph.add_node(0);
    /// let b = graph.add_node(1);
    /// let c = graph.add_node(2);
    /// graph.add_edge(a, b, 1, ()).unwrap();
    /// graph.add_edge(b, c, 1, ()).unwrap();
    ///
    /// assert!(graph.has_path(a, c).unwrap());
    /// assert!(!graph.has_path(c, a).unwrap()); // Directed graph
    /// ```
    fn has_path(&self, start: usize, target: usize) -> Result<bool, SearchError>;

    /// Finds the shortest path between nodes using BFS.
    ///
    /// Uses breadth-first search to find the shortest unweighted path from `start` to `target`.
    ///
    /// # Arguments
    /// - `start`: The ID of the starting node.
    /// - `target`: The ID of the target node.
    ///
    /// # Returns
    /// - `Ok(Option<Vec<usize>>)`: A vector of node IDs representing the shortest path, or `None` if no path exists.
    /// - `Err(SearchError)`: If either `start` or `target` is not a valid node.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::graph::algorithms::search::Search;
    ///
    /// let mut graph = Graph::<u32, (), ()>::new(true);
    /// let nodes = (0..4).map(|i| graph.add_node(())).collect::<Vec<_>>();
    /// graph.add_edge(nodes[0], nodes[1], 1, ()).unwrap();
    /// graph.add_edge(nodes[1], nodes[2], 1, ()).unwrap();
    /// graph.add_edge(nodes[0], nodes[3], 1, ()).unwrap();
    ///
    /// assert_eq!(
    ///     graph.bfs_path(nodes[0], nodes[2]).unwrap(),
    ///     Some(vec![nodes[0], nodes[1], nodes[2]])
    /// );
    /// ```
    fn bfs_path(&self, start: usize, target: usize) -> Result<Option<Vec<usize>>, SearchError>;

    /// Performs a recursive DFS to check for a path between nodes.
    ///
    /// Internal helper method for path checking, not typically called directly by users.
    ///
    /// # Arguments
    /// - `current`: The current node ID in the recursion.
    /// - `target`: The target node ID to find.
    /// - `visited`: A mutable reference to a set tracking visited nodes.
    ///
    /// # Returns
    /// - `Ok(bool)`: `true` if a path to `target` is found, `false` otherwise.
    /// - `Err(SearchError)`: If `current` or `target` is not a valid node.
    fn dfs(
        &self,
        current: usize,
        target: usize,
        visited: &mut HashSet<usize>,
    ) -> Result<bool, SearchError>;

    /// Checks if a node exists in the graph.
    ///
    /// Verifies whether a node with the given ID is present in the graph.
    ///
    /// # Arguments
    /// - `node`: The ID of the node to check.
    ///
    /// # Returns
    /// - `Ok(bool)`: `true` if the node exists, `false` otherwise.
    /// - `Err(SearchError)`: Never returned in this implementation, but included for consistency.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::graph::algorithms::search::Search;
    ///
    /// let mut graph = Graph::<i32, (), ()>::new(false);
    /// let n0 = graph.add_node(());
    /// assert!(graph.has_node(n0).unwrap());
    /// assert!(!graph.has_node(999).unwrap());
    /// ```
    fn has_node(&self, node: usize) -> Result<bool, SearchError>;

    /// Detects whether the graph contains any cycles.
    ///
    /// Uses DFS-based algorithms tailored to the graph's directionality:
    /// - Directed graphs: Uses a recursion stack to detect back edges.
    /// - Undirected graphs: Uses parent tracking to avoid false positives from bidirectional edges.
    ///
    /// # Returns
    /// - `Ok(bool)`: `true` if a cycle is detected, `false` otherwise.
    /// - `Err(SearchError)`: If an invalid node is encountered during traversal.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::graph::algorithms::search::Search;
    ///
    /// let mut graph = Graph::<u32, (), ()>::new(true);
    /// let nodes = (0..3).map(|i| graph.add_node(())).collect::<Vec<_>>();
    /// graph.add_edge(nodes[0], nodes[1], 1, ()).unwrap();
    /// graph.add_edge(nodes[1], nodes[2], 1, ()).unwrap();
    /// graph.add_edge(nodes[2], nodes[0], 1, ()).unwrap();
    /// assert!(graph.has_cycle().unwrap());
    /// ```
    fn has_cycle(&self) -> Result<bool, SearchError>;

    /// Helper method for cycle detection in directed graphs.
    ///
    /// Uses DFS with a recursion stack to detect cycles in directed graphs.
    ///
    /// # Arguments
    /// - `node`: The current node ID.
    /// - `visited`: A mutable reference to a set of visited nodes.
    /// - `recursion_stack`: A mutable reference to a set tracking the current recursion path.
    ///
    /// # Returns
    /// - `Ok(bool)`: `true` if a cycle is detected, `false` otherwise.
    /// - `Err(SearchError)`: If `node` is not a valid node.
    fn has_cycle_directed(
        &self,
        node: usize,
        visited: &mut HashSet<usize>,
        recursion_stack: &mut HashSet<usize>,
    ) -> Result<bool, SearchError>;

    /// Helper method for cycle detection in undirected graphs.
    ///
    /// Uses DFS with parent tracking to detect cycles in undirected graphs.
    ///
    /// # Arguments
    /// - `node`: The current node ID.
    /// - `parent`: The ID of the parent node in the DFS tree, or `None` if root.
    /// - `visited`: A mutable reference to a set of visited nodes.
    ///
    /// # Returns
    /// - `Ok(bool)`: `true` if a cycle is detected, `false` otherwise.
    /// - `Err(SearchError)`: If `node` is not a valid node.
    fn has_cycle_undirected(
        &self,
        node: usize,
        parent: Option<usize>,
        visited: &mut HashSet<usize>,
    ) -> Result<bool, SearchError>;
}

impl<W, N, E> Search<W, N, E> for Graph<W, N, E>
where
    W: Copy + Default + PartialEq,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    fn has_path(&self, start: usize, target: usize) -> Result<bool, SearchError> {
        if !self.nodes.contains(start) {
            return Err(SearchError::InvalidNode(start));
        }
        if !self.nodes.contains(target) {
            return Err(SearchError::InvalidNode(target));
        }
        let mut visited = HashSet::new();
        self.dfs(start, target, &mut visited)
    }

    fn bfs_path(&self, start: usize, target: usize) -> Result<Option<Vec<usize>>, SearchError> {
        if !self.nodes.contains(start) {
            return Err(SearchError::InvalidNode(start));
        }
        if !self.nodes.contains(target) {
            return Err(SearchError::InvalidNode(target));
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent = HashMap::new();

        queue.push_back(start);
        visited.insert(start);

        while let Some(current) = queue.pop_front() {
            if current == target {
                let mut path = vec![current];
                let mut node = current;
                while let Some(&p) = parent.get(&node) {
                    path.push(p);
                    node = p;
                }
                path.reverse();
                return Ok(Some(path));
            }

            for &(neighbor, _) in &self.nodes[current].neighbors {
                if !visited.contains(&neighbor) {
                    if !self.nodes.contains(neighbor) {
                        return Err(SearchError::InvalidNode(neighbor));
                    }
                    visited.insert(neighbor);
                    parent.insert(neighbor, current);
                    queue.push_back(neighbor);
                }
            }
        }
        Ok(None)
    }

    fn dfs(
        &self,
        current: usize,
        target: usize,
        visited: &mut HashSet<usize>,
    ) -> Result<bool, SearchError> {
        if !self.nodes.contains(current) {
            return Err(SearchError::InvalidNode(current));
        }
        if !self.nodes.contains(target) {
            return Err(SearchError::InvalidNode(target));
        }

        if current == target {
            return Ok(true);
        }

        visited.insert(current);

        for &(neighbor, _) in &self.nodes[current].neighbors {
            if !visited.contains(&neighbor) {
                if !self.nodes.contains(neighbor) {
                    return Err(SearchError::InvalidNode(neighbor));
                }
                if self.dfs(neighbor, target, visited)? {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn has_node(&self, node: usize) -> Result<bool, SearchError> {
        Ok(self.nodes.contains(node))
    }

    fn has_cycle(&self) -> Result<bool, SearchError> {
        if self.directed {
            let mut visited = HashSet::new();
            let mut recursion_stack = HashSet::new();

            for (node_id, _) in self.nodes.iter() {
                if !visited.contains(&node_id)
                    && self.has_cycle_directed(node_id, &mut visited, &mut recursion_stack)?
                {
                    return Ok(true);
                }
            }
            Ok(false)
        } else {
            let mut visited = HashSet::new();

            for (node_id, _) in self.nodes.iter() {
                if !visited.contains(&node_id)
                    && self.has_cycle_undirected(node_id, None, &mut visited)?
                {
                    return Ok(true);
                }
            }
            Ok(false)
        }
    }

    fn has_cycle_directed(
        &self,
        node: usize,
        visited: &mut HashSet<usize>,
        recursion_stack: &mut HashSet<usize>,
    ) -> Result<bool, SearchError> {
        if !self.nodes.contains(node) {
            return Err(SearchError::InvalidNode(node));
        }

        if recursion_stack.contains(&node) {
            return Ok(true);
        }

        if visited.contains(&node) {
            return Ok(false);
        }

        visited.insert(node);
        recursion_stack.insert(node);

        if let Some(neighbors) = self.nodes.get(node).map(|n| &n.neighbors) {
            for &(neighbor, _) in neighbors {
                if !self.nodes.contains(neighbor) {
                    return Err(SearchError::InvalidNode(neighbor));
                }
                if self.has_cycle_directed(neighbor, visited, recursion_stack)? {
                    return Ok(true);
                }
            }
        }

        recursion_stack.remove(&node);
        Ok(false)
    }

    fn has_cycle_undirected(
        &self,
        node: usize,
        parent: Option<usize>,
        visited: &mut HashSet<usize>,
    ) -> Result<bool, SearchError> {
        if !self.nodes.contains(node) {
            return Err(SearchError::InvalidNode(node));
        }

        if visited.contains(&node) {
            return Ok(true);
        }

        visited.insert(node);

        if let Some(neighbors) = self.nodes.get(node).map(|n| &n.neighbors) {
            for &(neighbor, _) in neighbors {
                if Some(neighbor) == parent {
                    continue;
                }
                if !self.nodes.contains(neighbor) {
                    return Err(SearchError::InvalidNode(neighbor));
                }
                if self.has_cycle_undirected(neighbor, Some(node), visited)? {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_path() {
        let mut graph = Graph::<u32, (), ()>::new(false);
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        graph.add_edge(n0, n1, 1, ()).unwrap();

        assert!(graph.has_path(n0, n1).unwrap());
        assert!(graph.has_path(n1, n0).unwrap());
        assert!(matches!(
            graph.has_path(999, n0),
            Err(SearchError::InvalidNode(999))
        ));
    }

    #[test]
    fn test_bfs_path() {
        let mut graph = Graph::<u32, (), ()>::new(true);
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        graph.add_edge(n0, n1, 1, ()).unwrap();
        graph.add_edge(n1, n2, 1, ()).unwrap();

        assert_eq!(graph.bfs_path(n0, n2).unwrap(), Some(vec![n0, n1, n2]));
        assert!(matches!(
            graph.bfs_path(999, n0),
            Err(SearchError::InvalidNode(999))
        ));
    }

    #[test]
    fn test_dfs() {
        let mut graph = Graph::<u32, (), ()>::new(false);
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        graph.add_edge(n0, n1, 1, ()).unwrap();
        graph.add_edge(n1, n2, 1, ()).unwrap();

        let mut visited = HashSet::new();
        assert!(graph.dfs(n0, n2, &mut visited).unwrap());
        assert!(matches!(
            graph.dfs(999, n0, &mut visited),
            Err(SearchError::InvalidNode(999))
        ));
    }

    #[test]
    fn test_invalid_nodes() {
        let graph = Graph::<u32, (), ()>::new(false);
        assert!(!graph.has_node(0).unwrap());
        assert!(matches!(
            graph.has_path(0, 1),
            Err(SearchError::InvalidNode(0))
        ));
        assert!(matches!(
            graph.bfs_path(0, 1),
            Err(SearchError::InvalidNode(0))
        ));
    }

    #[test]
    fn test_cycle_detection_directed() {
        let mut graph = Graph::<u32, (), ()>::new(true);
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        graph.add_edge(n0, n1, 1, ()).unwrap();
        graph.add_edge(n1, n2, 1, ()).unwrap();
        graph.add_edge(n2, n0, 1, ()).unwrap();

        assert!(graph.has_cycle().unwrap());
    }

    #[test]
    fn test_cycle_detection_undirected() {
        let mut graph = Graph::<u32, (), ()>::new(false);
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        graph.add_edge(n0, n1, 1, ()).unwrap();
        graph.add_edge(n1, n2, 1, ()).unwrap();
        graph.add_edge(n2, n0, 1, ()).unwrap();

        assert!(graph.has_cycle().unwrap());
    }

    #[test]
    fn test_no_cycle_directed() {
        let mut graph = Graph::<u32, (), ()>::new(true);
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        graph.add_edge(n0, n1, 1, ()).unwrap();
        graph.add_edge(n1, n2, 1, ()).unwrap();

        assert!(!graph.has_cycle().unwrap());
    }

    #[test]
    fn test_no_cycle_undirected() {
        let mut graph = Graph::<u32, (), ()>::new(false);
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        graph.add_edge(n0, n1, 1, ()).unwrap();
        graph.add_edge(n1, n2, 1, ()).unwrap();

        assert!(!graph.has_cycle().unwrap());
    }
}
