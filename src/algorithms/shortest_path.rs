//! Shortest path algorithms for weighted graphs
//!
//! Provides Dijkstra's algorithm implementation for finding shortest paths
//! in graphs with non-negative edge weights.
//!
//! # Examples
//!
//! Basic usage:
//! ```rust
//! use xgraph::graph::graph::Graph;
//! use xgraph::algorithms::shortest_path::ShortestPath;
//!
//! let mut graph = Graph::new(false);
//! let a = graph.add_node("A");
//! let b = graph.add_node("B");
//! graph.add_edge(a, b, 10, "road").unwrap();
//!
//! let distances = graph.dijkstra(a);
//! assert_eq!(distances[&b], 10);
//! ```

use crate::algorithms::search::Search;
use crate::graph::graph::Graph;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Add;

/// Trait for shortest path calculations in weighted graphs
///
/// # Examples
///
/// Finding shortest paths in a transportation network:
/// ```rust
/// use xgraph::graph::graph::Graph;
/// use xgraph::algorithms::shortest_path::ShortestPath;
///
/// let mut transport = Graph::new(false);
/// let cities = ["NYC", "Boston", "DC"].map(|n| transport.add_node(n));
///
/// transport.add_edge(cities[0], cities[1], 200, "highway").unwrap();
/// transport.add_edge(cities[1], cities[2], 150, "highway").unwrap();
///
/// let nyc_distances = transport.dijkstra(cities[0]);
/// assert_eq!(nyc_distances[&cities[2]], 350);
/// ```
pub trait ShortestPath<W, N, E> {
    /// Calculates shortest paths from start node using Dijkstra's algorithm
    ///
    /// # Arguments
    /// * `start` - Node ID to calculate distances from
    ///
    /// # Returns
    /// HashMap with node IDs as keys and shortest distances as values
    ///
    /// # Panics
    /// Doesn't panic but returns empty HashMap for invalid start node
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::shortest_path::ShortestPath;
    ///
    /// let mut graph = Graph::new(true);
    /// let a = graph.add_node(0);
    /// let b = graph.add_node(1);
    /// graph.add_edge(a, b, 5, "edge").unwrap();
    ///
    /// let distances = graph.dijkstra(a);
    /// assert_eq!(distances.get(&b), Some(&5));
    /// ```
    fn dijkstra(&self, start: usize) -> HashMap<usize, W>;
}

/// Represents a state in the Dijkstra's algorithm, including the cost to reach the node and the node index.
#[derive(Copy, Clone, Eq, PartialEq)]
struct State<W> {
    cost: W,
    node: usize,
}

/// Implements Ord trait for State to allow comparison based on the cost.
///
/// # Notes
/// The ordering is reversed (higher cost first) because BinaryHeap in Rust is a max heap by default.
impl<W: Ord> Ord for State<W> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost)
    }
}

/// Implements PartialOrd trait for State to allow comparison based on the cost.
///
/// # Notes
/// The ordering is reversed (higher cost first) because BinaryHeap in Rust is a max heap by default.
impl<W: Ord> PartialOrd for State<W> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Implements the ShortestPath trait for the Graph struct.
///
/// # Generic Parameters
/// * `W` - The type of weights in the graph. It must implement several traits including Add, Ord, Copy, Default, `From<u8>`, and Send.
/// * `N` - The type of nodes in the graph. It must be Clone, Eq, Hash, Debug.
/// * `E` - The type of edges in the graph. It must be Clone, Default, Debug.
impl<W, N, E> ShortestPath<W, N, E> for Graph<W, N, E>
where
    W: Add<Output = W> + Ord + Copy + Default + From<u8> + Send,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    /// Computes the shortest path from the given start node to all other nodes in the graph using Dijkstra's algorithm.
    ///
    /// # Arguments
    /// * `start` - The index of the start node from which the shortest paths are calculated.
    ///
    /// # Returns
    /// A HashMap where the keys are node indices and the values are the shortest distances from the start node.
    fn dijkstra(&self, start: usize) -> HashMap<usize, W> {
        let mut distances = HashMap::new();
        let mut heap = BinaryHeap::new();

        // Initialize the distance to the start node as 0
        if self.has_node(start) {
            distances.insert(start, W::default());
            heap.push(State {
                cost: W::default(),
                node: start,
            });
        }

        // Process nodes until the heap is empty
        while let Some(State { cost, node }) = heap.pop() {
            // If a shorter path to this node has already been found, skip processing
            if let Some(current_cost) = distances.get(&node) {
                if cost > *current_cost {
                    continue;
                }
            } else {
                continue;
            }

            // Update the distance for each neighboring node
            for &(neighbor, weight) in &self.nodes[node].neighbors {
                let next_cost = cost + weight;

                // If the neighbor exists in the graph, update its distance if a shorter path is found
                if self.has_node(neighbor) {
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
        }

        distances
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

        let distances = graph.dijkstra(n0);
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

        let distances = graph.dijkstra(n0);
        assert_eq!(distances.get(&n1), None);
    }
}
