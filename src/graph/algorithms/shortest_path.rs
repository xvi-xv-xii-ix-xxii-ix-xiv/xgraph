//! Shortest path algorithms for weighted graphs
//!
//! This module provides implementations of algorithms for finding shortest paths in weighted graphs,
//! specifically Dijkstra's algorithm for graphs with non-negative edge weights. It is designed to be
//! robust and efficient, with proper error handling to ensure usability in a library context.
//!
//! # Features
//! - Dijkstra's algorithm for shortest path computation from a single source
//! - Support for both directed and undirected graphs
//! - Comprehensive error handling for invalid inputs
//!
//! # Examples
//!
//! Basic usage with an undirected graph:
//! ```rust
//! use xgraph::graph::graph::Graph;
//! use xgraph::graph::algorithms::shortest_path::ShortestPath;
//!
//! let mut graph = Graph::<u32, &str, &str>::new(false);
//! let a = graph.add_node("A");
//! let b = graph.add_node("B");
//! graph.add_edge(a, b, 10, "road").unwrap();
//!
//! let distances = graph.dijkstra(a).unwrap();
//! assert_eq!(distances[&b], 10);
//! ```

use crate::graph::graph::Graph;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Add;

/// Error type for shortest path computation failures.
///
/// Represents errors that may occur during shortest path calculations.
#[derive(Debug)]
pub enum ShortestPathError {
    /// Indicates that the starting node does not exist in the graph.
    InvalidStartNode(usize),
    /// Indicates that a neighbor node referenced in the graph does not exist.
    InvalidNeighborNode(usize),
}

impl std::fmt::Display for ShortestPathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShortestPathError::InvalidStartNode(id) => {
                write!(f, "Invalid start node: node ID {} not found", id)
            }
            ShortestPathError::InvalidNeighborNode(id) => {
                write!(f, "Invalid neighbor node: node ID {} not found", id)
            }
        }
    }
}

impl std::error::Error for ShortestPathError {}

/// Represents a state in Dijkstra's algorithm, including the cost to reach the node and the node index.
///
/// Used internally by the priority queue to order nodes by their current shortest path cost.
#[derive(Copy, Clone, Eq, PartialEq)]
struct State<W> {
    /// The current shortest cost to reach this node.
    cost: W,
    /// The index of the node in the graph.
    node: usize,
}

/// Implements ordering for `State` to allow comparison based on cost.
///
/// # Notes
/// The ordering is reversed (higher cost first) because `BinaryHeap` in Rust is a max heap by default,
/// and we need a min heap behavior for Dijkstra's algorithm.
impl<W: Ord> Ord for State<W> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost)
    }
}

/// Implements partial ordering for `State` to allow comparison based on cost.
///
/// # Notes
/// This is required for `Ord` and ensures consistency with the reversed ordering for a min heap.
impl<W: Ord> PartialOrd for State<W> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Trait for shortest path calculations in weighted graphs.
///
/// Provides methods for computing shortest paths with proper error handling.
///
/// # Type Parameters
/// - `W`: Edge weight type, must support addition, ordering, copying, default value, conversion from `u8`, and be `Send`.
/// - `N`: Node data type, must be clonable, equatable, hashable, and debuggable.
/// - `E`: Edge data type, must be clonable and debuggable.
///
/// # Examples
/// Finding shortest paths in a transportation network:
/// ```rust
/// use xgraph::graph::graph::Graph;
/// use xgraph::graph::algorithms::shortest_path::ShortestPath;
///
/// let mut transport = Graph::<u32, &str, &str>::new(false);
/// let cities = ["NYC", "Boston", "DC"].map(|n| transport.add_node(n));
/// transport.add_edge(cities[0], cities[1], 200, "highway").unwrap();
/// transport.add_edge(cities[1], cities[2], 150, "highway").unwrap();
///
/// let nyc_distances = transport.dijkstra(cities[0]).unwrap();
/// assert_eq!(nyc_distances[&cities[2]], 350);
/// ```
pub trait ShortestPath<W, N, E>
where
    W: Add<Output = W> + Ord + Copy + Default + From<u8> + Send,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    /// Calculates shortest paths from a start node using Dijkstra's algorithm.
    ///
    /// Computes the shortest path distances from the given `start` node to all reachable nodes
    /// in the graph, assuming non-negative edge weights.
    ///
    /// # Arguments
    /// - `start`: The ID of the node from which to calculate distances.
    ///
    /// # Returns
    /// - `Ok(HashMap<usize, W>)`: A map with node IDs as keys and their shortest distances from `start` as values.
    /// - `Err(ShortestPathError)`: If the start node is invalid or if an invalid neighbor is encountered.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::graph::algorithms::shortest_path::ShortestPath;
    /// use xgraph::graph::algorithms::shortest_path::ShortestPathError;
    ///
    /// let mut graph = Graph::<u32, i32, &str>::new(true);
    /// let a = graph.add_node(0);
    /// let b = graph.add_node(1);
    /// graph.add_edge(a, b, 5, "edge").unwrap();
    ///
    /// let distances = graph.dijkstra(a).unwrap();
    /// assert_eq!(distances.get(&b), Some(&5));
    /// assert!(matches!(
    ///     graph.dijkstra(999),
    ///     Err(ShortestPathError::InvalidStartNode(999))
    /// ));
    /// ```
    fn dijkstra(&self, start: usize) -> Result<HashMap<usize, W>, ShortestPathError>;
}

impl<W, N, E> ShortestPath<W, N, E> for Graph<W, N, E>
where
    W: Add<Output = W> + Ord + Copy + Default + From<u8> + Send,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    fn dijkstra(&self, start: usize) -> Result<HashMap<usize, W>, ShortestPathError> {
        if !self.nodes.contains(start) {
            return Err(ShortestPathError::InvalidStartNode(start));
        }

        let mut distances = HashMap::new();
        let mut heap = BinaryHeap::new();

        // Initialize the distance to the start node as 0
        distances.insert(start, W::default());
        heap.push(State {
            cost: W::default(),
            node: start,
        });

        // Process nodes until the heap is empty
        while let Some(State { cost, node }) = heap.pop() {
            // If a shorter path to this node has already been found, skip processing
            if let Some(&current_cost) = distances.get(&node) {
                if cost > current_cost {
                    continue;
                }
            } else {
                continue;
            }

            // Update the distance for each neighboring node
            for &(neighbor, weight) in &self.nodes[node].neighbors {
                if !self.nodes.contains(neighbor) {
                    return Err(ShortestPathError::InvalidNeighborNode(neighbor));
                }

                let next_cost = cost + weight;

                match distances.entry(neighbor) {
                    std::collections::hash_map::Entry::Occupied(mut entry) => {
                        if next_cost < *entry.get() {
                            entry.insert(next_cost);
                            heap.push(State {
                                cost: next_cost,
                                node: neighbor,
                            });
                        }
                    }
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(next_cost);
                        heap.push(State {
                            cost: next_cost,
                            node: neighbor,
                        });
                    }
                }
            }
        }

        Ok(distances)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test to check basic functionality of Dijkstra's algorithm.
    #[test]
    fn test_dijkstra_basic() {
        let mut graph = Graph::<u32, (), ()>::new(false);
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        graph.add_edge(n0, n1, 2, ()).unwrap();
        graph.add_edge(n1, n2, 3, ()).unwrap();

        let distances = graph.dijkstra(n0).unwrap();
        assert_eq!(distances[&n0], 0);
        assert_eq!(distances[&n1], 2);
        assert_eq!(distances[&n2], 5);
    }

    /// Test to check handling of an unreachable node in the graph.
    #[test]
    fn test_unreachable_node() {
        let mut graph = Graph::<u64, (), ()>::new(true);
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());

        let distances = graph.dijkstra(n0).unwrap();
        assert_eq!(distances.get(&n1), None);
    }

    /// Test to check error handling for an invalid start node.
    #[test]
    fn test_invalid_start_node() {
        let graph = Graph::<u32, (), ()>::new(false);
        assert!(matches!(
            graph.dijkstra(999),
            Err(ShortestPathError::InvalidStartNode(999))
        ));
    }
}
