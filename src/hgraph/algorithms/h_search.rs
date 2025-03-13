//! Graph search algorithms and operations for heterogeneous multigraphs.
//!
//! This module provides a suite of graph traversal and analysis algorithms tailored for
//! heterogeneous multigraphs, supporting both directed and undirected graphs with multiple
//! edge types between nodes. Key functionalities include:
//! - Path existence checking using Depth-First Search (DFS).
//! - Shortest path finding using Breadth-First Search (BFS).
//! - Cycle detection for directed and undirected graphs.
//! - Node existence verification.
//!
//! All methods support filtering by edge types, enabling analysis of subgraphs defined by
//! specific edge categories. The module is available only when the `hgraph` feature is enabled
//! in your `Cargo.toml`:
//! ```toml
//! [dependencies.xgraph]
//! version = "x.y.z"
//! features = ["hgraph"]
//! ```

#[cfg(feature = "hgraph")]
use crate::hgraph::h_graph::HeterogeneousGraph;
#[cfg(feature = "hgraph")]
use std::collections::{HashMap, HashSet, VecDeque};
#[cfg(feature = "hgraph")]
use std::fmt::Debug;
#[cfg(feature = "hgraph")]
use std::hash::Hash;

// Error handling additions

/// Error type for graph search operation failures.
///
/// Represents errors that may occur during graph traversal or analysis, such as referencing
/// invalid nodes.
#[cfg(feature = "hgraph")]
#[derive(Debug)]
pub enum SearchError {
    /// Indicates a node referenced in the graph does not exist.
    InvalidNodeReference(usize),
}

#[cfg(feature = "hgraph")]
impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchError::InvalidNodeReference(id) => {
                write!(f, "Invalid node reference: node ID {} not found", id)
            }
        }
    }
}

#[cfg(feature = "hgraph")]
impl std::error::Error for SearchError {}

/// Result type alias for search operations.
///
/// Wraps the result of search methods, enabling error handling without panicking.
#[cfg(feature = "hgraph")]
pub type Result<T> = std::result::Result<T, SearchError>;

#[cfg(feature = "hgraph")]
/// Trait for graph search operations in heterogeneous multigraphs.
///
/// Defines methods for path finding, cycle detection, and node existence checks, with support
/// for edge type filtering. All methods that may fail due to invalid input return a `Result`
/// to handle errors gracefully.
pub trait HeteroSearch<W, N, E>
where
    W: Copy + Default + PartialEq + Debug,
    N: Clone + Eq + Hash + Debug + crate::hgraph::h_node::NodeType,
    E: Clone + Default + Debug + crate::hgraph::h_edge::EdgeType,
{
    /// Checks if a path exists between two nodes using DFS.
    ///
    /// # Arguments
    /// * `start` - The starting node ID.
    /// * `target` - The target node ID.
    ///
    /// # Returns
    /// `true` if a path exists from `start` to `target`, `false` otherwise.
    fn has_path(&self, start: usize, target: usize) -> bool;

    /// Checks if a path exists between two nodes using DFS, considering only specific edge types.
    ///
    /// # Arguments
    /// * `start` - The starting node ID.
    /// * `target` - The target node ID.
    /// * `edge_types` - A slice of edge type strings to filter the search.
    ///
    /// # Returns
    /// `true` if a path exists from `start` to `target` using only the specified edge types,
    /// `false` otherwise.
    fn has_path_by_types(&self, start: usize, target: usize, edge_types: &[&str]) -> bool;

    /// Finds the shortest path between nodes using BFS.
    ///
    /// Returns the path as a vector of node IDs from `start` to `target`, or an error if either
    /// node does not exist.
    ///
    /// # Arguments
    /// * `start` - The starting node ID.
    /// * `target` - The target node ID.
    ///
    /// # Returns
    /// A `Result` containing `Some(Vec<usize>)` with the path if found, `None` if no path exists,
    /// or an error if nodes are invalid.
    fn bfs_path(&self, start: usize, target: usize) -> Result<Option<Vec<usize>>>;

    /// Finds the shortest path between nodes using BFS, considering only specific edge types.
    ///
    /// Returns the path as a vector of node IDs from `start` to `target`, or an error if either
    /// node does not exist.
    ///
    /// # Arguments
    /// * `start` - The starting node ID.
    /// * `target` - The target node ID.
    /// * `edge_types` - A slice of edge type strings to filter the search.
    ///
    /// # Returns
    /// A `Result` containing `Some(Vec<usize>)` with the path if found, `None` if no path exists,
    /// or an error if nodes are invalid.
    fn bfs_path_by_types(
        &self,
        start: usize,
        target: usize,
        edge_types: &[&str],
    ) -> Result<Option<Vec<usize>>>;

    /// Recursive DFS implementation for path checking.
    ///
    /// Internal helper method for `has_path`.
    ///
    /// # Arguments
    /// * `current` - The current node ID being explored.
    /// * `target` - The target node ID.
    /// * `visited` - A set of already visited node IDs.
    ///
    /// # Returns
    /// `true` if a path to `target` is found, `false` otherwise.
    fn dfs(&self, current: usize, target: usize, visited: &mut HashSet<usize>) -> bool;

    /// Recursive DFS implementation for path checking with edge type filtering.
    ///
    /// Internal helper method for `has_path_by_types`.
    ///
    /// # Arguments
    /// * `current` - The current node ID being explored.
    /// * `target` - The target node ID.
    /// * `visited` - A set of already visited node IDs.
    /// * `edge_types` - A slice of edge type strings to filter the search.
    ///
    /// # Returns
    /// `true` if a path to `target` is found using specified edge types, `false` otherwise.
    fn dfs_by_types(
        &self,
        current: usize,
        target: usize,
        visited: &mut HashSet<usize>,
        edge_types: &[&str],
    ) -> bool;

    /// Checks if a node exists in the graph.
    ///
    /// # Arguments
    /// * `node` - The node ID to check.
    ///
    /// # Returns
    /// `true` if the node exists, `false` otherwise.
    fn has_node(&self, node: usize) -> bool;

    /// Detects cycles in the graph.
    ///
    /// Uses DFS-based cycle detection, differing in approach based on whether the graph is
    /// directed or undirected.
    ///
    /// # Returns
    /// `true` if the graph contains at least one cycle, `false` otherwise.
    fn has_cycle(&self) -> bool;

    /// Detects cycles in the graph, considering only specific edge types.
    ///
    /// Uses DFS-based cycle detection with edge type filtering.
    ///
    /// # Arguments
    /// * `edge_types` - A slice of edge type strings to filter the search.
    ///
    /// # Returns
    /// `true` if a cycle exists using only the specified edge types, `false` otherwise.
    fn has_cycle_by_types(&self, edge_types: &[&str]) -> bool;

    /// Helper for directed cycle detection.
    ///
    /// Internal method using DFS with a recursion stack to detect cycles in directed graphs.
    ///
    /// # Arguments
    /// * `node` - The current node ID being explored.
    /// * `visited` - A set of already visited node IDs.
    /// * `recursion_stack` - A set tracking nodes in the current recursion path.
    ///
    /// # Returns
    /// `true` if a cycle is detected, `false` otherwise.
    fn has_cycle_directed(
        &self,
        node: usize,
        visited: &mut HashSet<usize>,
        recursion_stack: &mut HashSet<usize>,
    ) -> bool;

    /// Helper for directed cycle detection with edge type filtering.
    ///
    /// Internal method using DFS with a recursion stack and edge type filtering.
    ///
    /// # Arguments
    /// * `node` - The current node ID being explored.
    /// * `visited` - A set of already visited node IDs.
    /// * `recursion_stack` - A set tracking nodes in the current recursion path.
    /// * `edge_types` - A slice of edge type strings to filter the search.
    ///
    /// # Returns
    /// `true` if a cycle is detected using specified edge types, `false` otherwise.
    fn has_cycle_directed_by_types(
        &self,
        node: usize,
        visited: &mut HashSet<usize>,
        recursion_stack: &mut HashSet<usize>,
        edge_types: &[&str],
    ) -> bool;

    /// Helper for undirected cycle detection.
    ///
    /// Internal method using DFS with parent tracking to detect cycles in undirected graphs.
    ///
    /// # Arguments
    /// * `node` - The current node ID being explored.
    /// * `parent` - The parent node ID in the DFS tree, if any.
    /// * `visited` - A set of already visited node IDs.
    ///
    /// # Returns
    /// `true` if a cycle is detected, `false` otherwise.
    fn has_cycle_undirected(
        &self,
        node: usize,
        parent: Option<usize>,
        visited: &mut HashSet<usize>,
    ) -> bool;

    /// Helper for undirected cycle detection with edge type filtering.
    ///
    /// Internal method using DFS with parent tracking and edge type filtering.
    ///
    /// # Arguments
    /// * `node` - The current node ID being explored.
    /// * `parent` - The parent node ID in the DFS tree, if any.
    /// * `visited` - A set of already visited node IDs.
    /// * `edge_types` - A slice of edge type strings to filter the search.
    ///
    /// # Returns
    /// `true` if a cycle is detected using specified edge types, `false` otherwise.
    fn has_cycle_undirected_by_types(
        &self,
        node: usize,
        parent: Option<usize>,
        visited: &mut HashSet<usize>,
        edge_types: &[&str],
    ) -> bool;
}

#[cfg(feature = "hgraph")]
impl<W, N, E> HeteroSearch<W, N, E> for HeterogeneousGraph<W, N, E>
where
    W: Copy + Default + PartialEq + Debug,
    N: Clone + Eq + Hash + Debug + crate::hgraph::h_node::NodeType,
    E: Clone + Default + Debug + crate::hgraph::h_edge::EdgeType,
{
    fn has_path(&self, start: usize, target: usize) -> bool {
        let mut visited = HashSet::new();
        self.dfs(start, target, &mut visited)
    }

    fn has_path_by_types(&self, start: usize, target: usize, edge_types: &[&str]) -> bool {
        let mut visited = HashSet::new();
        self.dfs_by_types(start, target, &mut visited, edge_types)
    }

    fn bfs_path(&self, start: usize, target: usize) -> Result<Option<Vec<usize>>> {
        if !self.has_node(start) {
            return Err(SearchError::InvalidNodeReference(start));
        }
        if !self.has_node(target) {
            return Err(SearchError::InvalidNodeReference(target));
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

            for &(neighbor, ref _edges) in &self.nodes[current].neighbors {
                if !visited.contains(&neighbor) && self.has_node(neighbor) {
                    visited.insert(neighbor);
                    parent.insert(neighbor, current);
                    queue.push_back(neighbor);
                }
            }
        }
        Ok(None)
    }

    fn bfs_path_by_types(
        &self,
        start: usize,
        target: usize,
        edge_types: &[&str],
    ) -> Result<Option<Vec<usize>>> {
        if !self.has_node(start) {
            return Err(SearchError::InvalidNodeReference(start));
        }
        if !self.has_node(target) {
            return Err(SearchError::InvalidNodeReference(target));
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

            for &(neighbor, ref edges) in &self.nodes[current].neighbors {
                if edges.iter().any(|&(edge_id, _)| {
                    let edge = self.edges.get(edge_id).expect("Edge should exist");
                    edge_types.contains(&edge.data.as_string().as_str())
                }) && !visited.contains(&neighbor)
                    && self.has_node(neighbor)
                {
                    visited.insert(neighbor);
                    parent.insert(neighbor, current);
                    queue.push_back(neighbor);
                }
            }
        }
        Ok(None)
    }

    fn dfs(&self, current: usize, target: usize, visited: &mut HashSet<usize>) -> bool {
        if current == target {
            return true;
        }

        if !self.has_node(current) || !self.has_node(target) {
            return false;
        }

        visited.insert(current);

        for &(neighbor, ref _edges) in &self.nodes[current].neighbors {
            if !visited.contains(&neighbor)
                && self.has_node(neighbor)
                && self.dfs(neighbor, target, visited)
            {
                return true;
            }
        }
        false
    }

    fn dfs_by_types(
        &self,
        current: usize,
        target: usize,
        visited: &mut HashSet<usize>,
        edge_types: &[&str],
    ) -> bool {
        if current == target {
            return true;
        }

        if !self.has_node(current) || !self.has_node(target) {
            return false;
        }

        visited.insert(current);

        for &(neighbor, ref edges) in &self.nodes[current].neighbors {
            if edges.iter().any(|&(edge_id, _)| {
                let edge = self.edges.get(edge_id).expect("Edge should exist");
                edge_types.contains(&edge.data.as_string().as_str())
            }) && !visited.contains(&neighbor)
                && self.has_node(neighbor)
                && self.dfs_by_types(neighbor, target, visited, edge_types)
            {
                return true;
            }
        }
        false
    }

    fn has_node(&self, node: usize) -> bool {
        self.nodes.get(node).is_some()
    }

    fn has_cycle(&self) -> bool {
        if self.directed {
            let mut visited = HashSet::new();
            let mut recursion_stack = HashSet::new();

            for (node_id, _) in self.nodes.iter() {
                if !visited.contains(&node_id)
                    && self.has_cycle_directed(node_id, &mut visited, &mut recursion_stack)
                {
                    return true;
                }
            }
            false
        } else {
            let mut visited = HashSet::new();

            for (node_id, _) in self.nodes.iter() {
                if !visited.contains(&node_id)
                    && self.has_cycle_undirected(node_id, None, &mut visited)
                {
                    return true;
                }
            }
            false
        }
    }

    fn has_cycle_by_types(&self, edge_types: &[&str]) -> bool {
        if self.directed {
            let mut visited = HashSet::new();
            let mut recursion_stack = HashSet::new();

            for (node_id, _) in self.nodes.iter() {
                if !visited.contains(&node_id)
                    && self.has_cycle_directed_by_types(
                        node_id,
                        &mut visited,
                        &mut recursion_stack,
                        edge_types,
                    )
                {
                    return true;
                }
            }
            false
        } else {
            let mut visited = HashSet::new();

            for (node_id, _) in self.nodes.iter() {
                if !visited.contains(&node_id)
                    && self.has_cycle_undirected_by_types(node_id, None, &mut visited, edge_types)
                {
                    return true;
                }
            }
            false
        }
    }

    fn has_cycle_directed(
        &self,
        node: usize,
        visited: &mut HashSet<usize>,
        recursion_stack: &mut HashSet<usize>,
    ) -> bool {
        if recursion_stack.contains(&node) {
            return true;
        }

        if visited.contains(&node) {
            return false;
        }

        visited.insert(node);
        recursion_stack.insert(node);

        if let Some(neighbors) = self.nodes.get(node).map(|n| &n.neighbors) {
            for &(neighbor, ref _edges) in neighbors {
                if self.has_cycle_directed(neighbor, visited, recursion_stack) {
                    return true;
                }
            }
        }

        recursion_stack.remove(&node);
        false
    }

    fn has_cycle_directed_by_types(
        &self,
        node: usize,
        visited: &mut HashSet<usize>,
        recursion_stack: &mut HashSet<usize>,
        edge_types: &[&str],
    ) -> bool {
        if recursion_stack.contains(&node) {
            return true;
        }

        if visited.contains(&node) {
            return false;
        }

        visited.insert(node);
        recursion_stack.insert(node);

        if let Some(neighbors) = self.nodes.get(node).map(|n| &n.neighbors) {
            for &(neighbor, ref edges) in neighbors {
                if edges.iter().any(|&(edge_id, _)| {
                    let edge = self.edges.get(edge_id).expect("Edge should exist");
                    edge_types.contains(&edge.data.as_string().as_str())
                }) && self.has_cycle_directed_by_types(
                    neighbor,
                    visited,
                    recursion_stack,
                    edge_types,
                ) {
                    return true;
                }
            }
        }

        recursion_stack.remove(&node);
        false
    }

    fn has_cycle_undirected(
        &self,
        node: usize,
        parent: Option<usize>,
        visited: &mut HashSet<usize>,
    ) -> bool {
        if visited.contains(&node) {
            return true;
        }

        visited.insert(node);

        if let Some(neighbors) = self.nodes.get(node).map(|n| &n.neighbors) {
            for &(neighbor, ref _edges) in neighbors {
                if Some(neighbor) == parent {
                    continue;
                }

                if self.has_cycle_undirected(neighbor, Some(node), visited) {
                    return true;
                }
            }
        }

        false
    }

    fn has_cycle_undirected_by_types(
        &self,
        node: usize,
        parent: Option<usize>,
        visited: &mut HashSet<usize>,
        edge_types: &[&str],
    ) -> bool {
        if visited.contains(&node) {
            return true;
        }

        visited.insert(node);

        if let Some(neighbors) = self.nodes.get(node).map(|n| &n.neighbors) {
            for &(neighbor, ref edges) in neighbors {
                if Some(neighbor) == parent {
                    continue;
                }

                if edges.iter().any(|&(edge_id, _)| {
                    let edge = self.edges.get(edge_id).expect("Edge should exist");
                    edge_types.contains(&edge.data.as_string().as_str())
                }) && self.has_cycle_undirected_by_types(
                    neighbor,
                    Some(node),
                    visited,
                    edge_types,
                ) {
                    return true;
                }
            }
        }

        false
    }
}

#[cfg(test)]
#[cfg(feature = "hgraph")]
mod tests {
    use super::*;
    use crate::hgraph::h_edge::EdgeType;
    use crate::hgraph::h_node::NodeType;

    #[derive(Clone, Eq, PartialEq, Hash, Debug)]
    struct TestNode(String);
    impl NodeType for TestNode {
        fn as_string(&self) -> String {
            self.0.clone()
        }
    }

    #[derive(Clone, Debug, Default)]
    struct TestEdge(String);
    impl EdgeType for TestEdge {
        fn as_string(&self) -> String {
            self.0.clone()
        }
    }

    #[test]
    fn test_has_path() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();

        assert!(graph.has_path(n0, n1));
        assert!(graph.has_path(n1, n0));
    }

    #[test]
    fn test_has_path_by_types() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));
        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1.0, TestEdge("bridge".to_string()))
            .unwrap();

        assert!(graph.has_path_by_types(n0, n1, &["road"]));
        assert!(!graph.has_path_by_types(n0, n2, &["road"]));
        assert!(graph.has_path_by_types(n0, n2, &["road", "bridge"]));
    }

    #[test]
    fn test_bfs_path() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(true);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));
        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1.0, TestEdge("bridge".to_string()))
            .unwrap();

        assert_eq!(graph.bfs_path(n0, n2).unwrap(), Some(vec![n0, n1, n2]));
    }

    #[test]
    fn test_bfs_path_by_types() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(true);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));
        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1.0, TestEdge("bridge".to_string()))
            .unwrap();

        assert_eq!(
            graph.bfs_path_by_types(n0, n1, &["road"]).unwrap(),
            Some(vec![n0, n1])
        );
        assert_eq!(graph.bfs_path_by_types(n0, n2, &["road"]).unwrap(), None);
        assert_eq!(
            graph
                .bfs_path_by_types(n0, n2, &["road", "bridge"])
                .unwrap(),
            Some(vec![n0, n1, n2])
        );
    }

    #[test]
    fn test_dfs() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));
        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1.0, TestEdge("bridge".to_string()))
            .unwrap();

        let mut visited = HashSet::new();
        assert!(graph.dfs(n0, n2, &mut visited));
    }

    #[test]
    fn test_invalid_nodes() {
        let graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        assert!(!graph.has_node(0));
        assert!(!graph.has_path(0, 1));
        assert!(!graph.has_path_by_types(0, 1, &["road"]));
        assert!(matches!(
            graph.bfs_path(0, 1),
            Err(SearchError::InvalidNodeReference(0))
        ));
        assert!(matches!(
            graph.bfs_path_by_types(0, 1, &["road"]),
            Err(SearchError::InvalidNodeReference(0))
        ));
    }

    #[test]
    fn test_cycle_detection_directed() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(true);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1.0, TestEdge("bridge".to_string()))
            .unwrap();
        graph
            .add_edge(n2, n0, 1.0, TestEdge("path".to_string()))
            .unwrap();

        assert!(graph.has_cycle());
    }

    #[test]
    fn test_cycle_detection_directed_by_types() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(true);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1.0, TestEdge("bridge".to_string()))
            .unwrap();
        graph
            .add_edge(n2, n0, 1.0, TestEdge("path".to_string()))
            .unwrap();

        assert!(graph.has_cycle_by_types(&["road", "bridge", "path"]));
        assert!(!graph.has_cycle_by_types(&["road", "bridge"]));
    }

    #[test]
    fn test_cycle_detection_undirected() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1.0, TestEdge("bridge".to_string()))
            .unwrap();
        graph
            .add_edge(n2, n0, 1.0, TestEdge("path".to_string()))
            .unwrap();

        assert!(graph.has_cycle());
    }

    #[test]
    fn test_cycle_detection_undirected_by_types() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1.0, TestEdge("bridge".to_string()))
            .unwrap();
        graph
            .add_edge(n2, n0, 1.0, TestEdge("path".to_string()))
            .unwrap();

        assert!(graph.has_cycle_by_types(&["road", "bridge", "path"]));
        assert!(!graph.has_cycle_by_types(&["road", "bridge"]));
    }

    #[test]
    fn test_no_cycle_directed() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(true);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1.0, TestEdge("bridge".to_string()))
            .unwrap();

        assert!(!graph.has_cycle());
    }

    #[test]
    fn test_no_cycle_directed_by_types() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(true);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1.0, TestEdge("bridge".to_string()))
            .unwrap();

        assert!(!graph.has_cycle_by_types(&["road", "bridge"]));
    }

    #[test]
    fn test_no_cycle_undirected() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1.0, TestEdge("bridge".to_string()))
            .unwrap();

        assert!(!graph.has_cycle());
    }

    #[test]
    fn test_no_cycle_undirected_by_types() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1.0, TestEdge("bridge".to_string()))
            .unwrap();

        assert!(!graph.has_cycle_by_types(&["road", "bridge"]));
    }

    #[test]
    fn test_multiple_edges() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(true);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n0, n1, 2.0, TestEdge("path".to_string()))
            .unwrap();

        assert!(graph.has_path(n0, n1));
        assert!(graph.has_path_by_types(n0, n1, &["road"]));
        assert_eq!(graph.bfs_path(n0, n1).unwrap(), Some(vec![n0, n1]));
        assert_eq!(
            graph.bfs_path_by_types(n0, n1, &["path"]).unwrap(),
            Some(vec![n0, n1])
        );
        assert!(!graph.has_cycle());
        assert!(!graph.has_cycle_by_types(&["road", "path"]));
    }
}
