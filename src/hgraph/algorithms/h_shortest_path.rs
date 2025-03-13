//! Shortest path algorithms for heterogeneous multigraphs.
//!
//! This module provides implementations of Dijkstra's algorithm for computing shortest paths
//! in heterogeneous multigraphs, supporting both directed and undirected graphs with multiple
//! edge types. Key functionalities include:
//! - Computing shortest path distances and predecessors from a single source.
//! - Finding the shortest path between two nodes.
//! - Edge type filtering for type-specific shortest path analysis.
//!
//! The module is available only when the `hgraph` feature is enabled in your `Cargo.toml`:
//! ```toml
//! [dependencies.xgraph]
//! version = "x.y.z"
//! features = ["hgraph"]
//! ```

use crate::h_search::HeteroSearch;
#[cfg(feature = "hgraph")]
use crate::hgraph::h_graph::HeterogeneousGraph;
#[cfg(feature = "hgraph")]
use std::cmp::Ordering;
#[cfg(feature = "hgraph")]
use std::collections::{BinaryHeap, HashMap};
#[cfg(feature = "hgraph")]
use std::fmt::Debug;
#[cfg(feature = "hgraph")]
use std::hash::Hash;
#[cfg(feature = "hgraph")]
use std::ops::Add;

// Error handling additions

/// Error type for shortest path computation failures.
///
/// Represents errors that may occur during shortest path calculations, such as invalid node references.
#[cfg(feature = "hgraph")]
#[derive(Debug)]
pub enum ShortestPathError {
    /// Indicates a node referenced in the graph does not exist.
    InvalidNodeReference(usize),
}

#[cfg(feature = "hgraph")]
impl std::fmt::Display for ShortestPathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShortestPathError::InvalidNodeReference(id) => {
                write!(f, "Invalid node reference: node ID {} not found", id)
            }
        }
    }
}

#[cfg(feature = "hgraph")]
impl std::error::Error for ShortestPathError {}

/// Result type alias for shortest path operations.
///
/// Wraps the result of shortest path methods to enable error handling without panicking.
#[cfg(feature = "hgraph")]
pub type Result<T> = std::result::Result<T, ShortestPathError>;

#[cfg(feature = "hgraph")]
/// Trait for shortest path operations in heterogeneous multigraphs.
///
/// Provides methods for computing shortest path distances and reconstructing paths using
/// Dijkstra's algorithm, with support for edge type filtering. Methods that may fail due to
/// invalid inputs return a `Result` to handle errors gracefully.
pub trait HeteroShortestPath<W, N, E>
where
    W: Add<Output = W> + PartialOrd + Copy + Default + From<u8> + Debug + PartialEq,
    N: Clone + Eq + Hash + Debug + crate::hgraph::h_node::NodeType,
    E: Clone + Default + Debug + crate::hgraph::h_edge::EdgeType,
{
    /// Computes shortest path distances and predecessors from a start node using Dijkstra's algorithm.
    ///
    /// Uses all edges regardless of type to find the shortest paths from the start node to all reachable nodes.
    ///
    /// # Arguments
    /// * `start` - The starting node ID.
    ///
    /// # Returns
    /// A `Result` containing a tuple of:
    /// - `HashMap<usize, W>`: Shortest distances from `start` to each reachable node.
    /// - `HashMap<usize, usize>`: Predecessor nodes for path reconstruction.
    ///
    /// # Errors
    /// Returns `ShortestPathError::InvalidNodeReference` if `start` does not exist in the graph.
    fn dijkstra(&self, start: usize) -> Result<(HashMap<usize, W>, HashMap<usize, usize>)>;

    /// Computes shortest path distances and predecessors from a start node, considering only specific edge types.
    ///
    /// Filters edges by type to compute shortest paths in a subgraph defined by `edge_types`.
    ///
    /// # Arguments
    /// * `start` - The starting node ID.
    /// * `edge_types` - A slice of edge type strings to filter the computation.
    ///
    /// # Returns
    /// A `Result` containing a tuple of:
    /// - `HashMap<usize, W>`: Shortest distances from `start` to each reachable node.
    /// - `HashMap<usize, usize>`: Predecessor nodes for path reconstruction.
    ///
    /// # Errors
    /// Returns `ShortestPathError::InvalidNodeReference` if `start` does not exist in the graph.
    fn dijkstra_by_types(
        &self,
        start: usize,
        edge_types: &[&str],
    ) -> Result<(HashMap<usize, W>, HashMap<usize, usize>)>;

    /// Finds the shortest path between two nodes using Dijkstra's algorithm.
    ///
    /// Returns the path as a vector of node IDs from `start` to `end`, using all edges.
    ///
    /// # Arguments
    /// * `start` - The starting node ID.
    /// * `end` - The target node ID.
    ///
    /// # Returns
    /// A `Result` containing `Some(Vec<usize>)` with the shortest path if found, `None` if no path exists,
    /// or an error if either node is invalid.
    fn dijkstra_path(&self, start: usize, end: usize) -> Result<Option<Vec<usize>>>;

    /// Finds the shortest path between two nodes using Dijkstra's algorithm, considering only specific edge types.
    ///
    /// Returns the path as a vector of node IDs from `start` to `end`, filtered by `edge_types`.
    ///
    /// # Arguments
    /// * `start` - The starting node ID.
    /// * `end` - The target node ID.
    /// * `edge_types` - A slice of edge type strings to filter the computation.
    ///
    /// # Returns
    /// A `Result` containing `Some(Vec<usize>)` with the shortest path if found, `None` if no path exists,
    /// or an error if either node is invalid.
    fn dijkstra_path_by_types(
        &self,
        start: usize,
        end: usize,
        edge_types: &[&str],
    ) -> Result<Option<Vec<usize>>>;
}

#[cfg(feature = "hgraph")]
#[derive(Copy, Clone, PartialEq)]
/// Represents a state in Dijkstra's algorithm with a cost and node ID.
///
/// Used in the priority queue to prioritize nodes with lower costs.
struct State<W> {
    cost: W,
    node: usize,
}

#[cfg(feature = "hgraph")]
impl<W: PartialOrd> PartialOrd for State<W> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(feature = "hgraph")]
impl<W: PartialEq> Eq for State<W> {}

#[cfg(feature = "hgraph")]
impl<W: PartialOrd> Ord for State<W> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap behavior (lower cost has higher priority)
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

#[cfg(feature = "hgraph")]
impl<W, N, E> HeteroShortestPath<W, N, E> for HeterogeneousGraph<W, N, E>
where
    W: Add<Output = W> + PartialOrd + Copy + Default + From<u8> + Debug + PartialEq,
    N: Clone + Eq + Hash + Debug + crate::hgraph::h_node::NodeType,
    E: Clone + Default + Debug + crate::hgraph::h_edge::EdgeType,
{
    fn dijkstra(&self, start: usize) -> Result<(HashMap<usize, W>, HashMap<usize, usize>)> {
        if !self.has_node(start) {
            return Err(ShortestPathError::InvalidNodeReference(start));
        }

        let mut distances = HashMap::new();
        let mut previous = HashMap::new();
        let mut heap = BinaryHeap::new();

        distances.insert(start, W::default());
        heap.push(State {
            cost: W::default(),
            node: start,
        });

        while let Some(State { cost, node }) = heap.pop() {
            if let Some(current_cost) = distances.get(&node) {
                if cost.partial_cmp(current_cost) == Some(Ordering::Greater) {
                    continue;
                }
            } else {
                continue;
            }

            for &(neighbor, ref edges) in &self.nodes[node].neighbors {
                let min_weight = edges
                    .iter()
                    .map(|&(_, weight)| weight)
                    .filter(|w| w.partial_cmp(&W::default()).is_some())
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

                if let Some(min_weight) = min_weight {
                    let next_cost = cost + min_weight;
                    if self.has_node(neighbor) {
                        match distances.entry(neighbor) {
                            std::collections::hash_map::Entry::Occupied(mut entry) => {
                                if next_cost.partial_cmp(entry.get()) == Some(Ordering::Less) {
                                    entry.insert(next_cost);
                                    previous.insert(neighbor, node);
                                    heap.push(State {
                                        cost: next_cost,
                                        node: neighbor,
                                    });
                                }
                            }
                            std::collections::hash_map::Entry::Vacant(entry) => {
                                entry.insert(next_cost);
                                previous.insert(neighbor, node);
                                heap.push(State {
                                    cost: next_cost,
                                    node: neighbor,
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok((distances, previous))
    }

    fn dijkstra_by_types(
        &self,
        start: usize,
        edge_types: &[&str],
    ) -> Result<(HashMap<usize, W>, HashMap<usize, usize>)> {
        if !self.has_node(start) {
            return Err(ShortestPathError::InvalidNodeReference(start));
        }

        let mut distances = HashMap::new();
        let mut previous = HashMap::new();
        let mut heap = BinaryHeap::new();

        distances.insert(start, W::default());
        heap.push(State {
            cost: W::default(),
            node: start,
        });

        while let Some(State { cost, node }) = heap.pop() {
            if let Some(current_cost) = distances.get(&node) {
                if cost.partial_cmp(current_cost) == Some(Ordering::Greater) {
                    continue;
                }
            } else {
                continue;
            }

            for &(neighbor, ref edges) in &self.nodes[node].neighbors {
                let min_weight = edges
                    .iter()
                    .filter(|&&(edge_id, _)| {
                        let edge = self.edges.get(edge_id).expect("Edge should exist");
                        edge_types.contains(&edge.data.as_string().as_str())
                    })
                    .map(|&(_, weight)| weight)
                    .filter(|w| w.partial_cmp(&W::default()).is_some())
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

                if let Some(min_weight) = min_weight {
                    let next_cost = cost + min_weight;
                    if self.has_node(neighbor) {
                        match distances.entry(neighbor) {
                            std::collections::hash_map::Entry::Occupied(mut entry) => {
                                if next_cost.partial_cmp(entry.get()) == Some(Ordering::Less) {
                                    entry.insert(next_cost);
                                    previous.insert(neighbor, node);
                                    heap.push(State {
                                        cost: next_cost,
                                        node: neighbor,
                                    });
                                }
                            }
                            std::collections::hash_map::Entry::Vacant(entry) => {
                                entry.insert(next_cost);
                                previous.insert(neighbor, node);
                                heap.push(State {
                                    cost: next_cost,
                                    node: neighbor,
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok((distances, previous))
    }

    fn dijkstra_path(&self, start: usize, end: usize) -> Result<Option<Vec<usize>>> {
        if !self.has_node(start) {
            return Err(ShortestPathError::InvalidNodeReference(start));
        }
        if !self.has_node(end) {
            return Err(ShortestPathError::InvalidNodeReference(end));
        }

        let (_, previous) = self.dijkstra(start)?;

        if !previous.contains_key(&end) && start != end {
            return Ok(None);
        }

        let mut path = Vec::new();
        let mut current = end;

        while current != start {
            path.push(current);
            if let Some(&prev) = previous.get(&current) {
                current = prev;
            } else {
                break;
            }
        }
        path.push(start);
        path.reverse();

        Ok(Some(path))
    }

    fn dijkstra_path_by_types(
        &self,
        start: usize,
        end: usize,
        edge_types: &[&str],
    ) -> Result<Option<Vec<usize>>> {
        if !self.has_node(start) {
            return Err(ShortestPathError::InvalidNodeReference(start));
        }
        if !self.has_node(end) {
            return Err(ShortestPathError::InvalidNodeReference(end));
        }

        let (_, previous) = self.dijkstra_by_types(start, edge_types)?;

        if !previous.contains_key(&end) && start != end {
            return Ok(None);
        }

        let mut path = Vec::new();
        let mut current = end;

        while current != start {
            path.push(current);
            if let Some(&prev) = previous.get(&current) {
                current = prev;
            } else {
                break;
            }
        }
        path.push(start);
        path.reverse();

        Ok(Some(path))
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
    fn test_dijkstra() {
        let mut graph = HeterogeneousGraph::<u32, TestNode, TestEdge>::new(true);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 4, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 3, TestEdge("bridge".to_string()))
            .unwrap();
        graph
            .add_edge(n0, n2, 8, TestEdge("path".to_string()))
            .unwrap();

        let (distances, _) = graph.dijkstra(n0).unwrap();
        assert_eq!(distances[&n0], 0);
        assert_eq!(distances[&n1], 4);
        assert_eq!(distances[&n2], 7); // 0 -> 1 -> 2
    }

    #[test]
    fn test_dijkstra_by_types() {
        let mut graph = HeterogeneousGraph::<u32, TestNode, TestEdge>::new(true);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 4, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 3, TestEdge("bridge".to_string()))
            .unwrap();
        graph
            .add_edge(n0, n2, 8, TestEdge("path".to_string()))
            .unwrap();

        let (distances, _) = graph.dijkstra_by_types(n0, &["road", "bridge"]).unwrap();
        assert_eq!(distances[&n0], 0);
        assert_eq!(distances[&n1], 4);
        assert_eq!(distances[&n2], 7); // 0 -> 1 -> 2

        let (distances, _) = graph.dijkstra_by_types(n0, &["road"]).unwrap();
        assert_eq!(distances[&n0], 0);
        assert_eq!(distances[&n1], 4);
        assert_eq!(distances.get(&n2), None); // n2 unreachable with only "road"
    }

    #[test]
    fn test_dijkstra_path() {
        let mut graph = HeterogeneousGraph::<u32, TestNode, TestEdge>::new(true);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 4, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 3, TestEdge("bridge".to_string()))
            .unwrap();
        graph
            .add_edge(n0, n2, 8, TestEdge("path".to_string()))
            .unwrap();

        let path = graph.dijkstra_path(n0, n2).unwrap();
        assert_eq!(path, Some(vec![n0, n1, n2]));
    }

    #[test]
    fn test_dijkstra_path_by_types() {
        let mut graph = HeterogeneousGraph::<u32, TestNode, TestEdge>::new(true);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 4, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 3, TestEdge("bridge".to_string()))
            .unwrap();
        graph
            .add_edge(n0, n2, 8, TestEdge("path".to_string()))
            .unwrap();

        let path = graph
            .dijkstra_path_by_types(n0, n2, &["road", "bridge"])
            .unwrap();
        assert_eq!(path, Some(vec![n0, n1, n2]));

        let path = graph.dijkstra_path_by_types(n0, n2, &["path"]).unwrap();
        assert_eq!(path, Some(vec![n0, n2]));

        let path = graph.dijkstra_path_by_types(n0, n2, &["road"]).unwrap();
        assert_eq!(path, None); // n2 unreachable with only "road"
    }

    #[test]
    fn test_unreachable_node() {
        let mut graph = HeterogeneousGraph::<u32, TestNode, TestEdge>::new(true);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));

        let (distances, _) = graph.dijkstra(n0).unwrap();
        assert_eq!(distances[&n0], 0);
        assert_eq!(distances.get(&n1), None);

        let path = graph.dijkstra_path(n0, n1).unwrap();
        assert_eq!(path, None);

        let path = graph.dijkstra_path_by_types(n0, n1, &["road"]).unwrap();
        assert_eq!(path, None);
    }

    #[test]
    fn test_same_start_end() {
        let mut graph = HeterogeneousGraph::<u32, TestNode, TestEdge>::new(true);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        graph
            .add_edge(n0, n1, 4, TestEdge("road".to_string()))
            .unwrap();

        let path = graph.dijkstra_path(n0, n0).unwrap();
        assert_eq!(path, Some(vec![n0]));

        let path = graph.dijkstra_path_by_types(n0, n0, &["road"]).unwrap();
        assert_eq!(path, Some(vec![n0]));
    }

    #[test]
    fn test_invalid_node() {
        let graph = HeterogeneousGraph::<u32, TestNode, TestEdge>::new(true);
        assert!(matches!(
            graph.dijkstra(0),
            Err(ShortestPathError::InvalidNodeReference(0))
        ));
        assert!(matches!(
            graph.dijkstra_path(0, 1),
            Err(ShortestPathError::InvalidNodeReference(0))
        ));
        assert!(matches!(
            graph.dijkstra_path_by_types(0, 1, &["road"]),
            Err(ShortestPathError::InvalidNodeReference(0))
        ));
    }
}
