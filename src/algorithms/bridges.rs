//! Bridge detection algorithms for graphs
//!
//! Provides functionality to find bridge edges in a graph using Tarjan's algorithm.
//! A bridge is an edge whose removal increases the number of connected components.
//!
//! # Examples
//!
//! Basic usage:
//! ```rust
//! use xgraph::graph::graph::Graph;
//! use xgraph::algorithms::bridges::Bridges;
//!
//! let mut graph = Graph::<u32, (), ()>::new(false);
//! let n0 = graph.add_node(());
//! let n1 = graph.add_node(());
//! graph.add_edge(n0, n1, 1, ()).unwrap();
//!
//! let bridges = graph.find_bridges();
//! assert_eq!(bridges, vec![(n0, n1)]);
//! ```

use crate::graph::graph::Graph;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

/// Context container for Tarjan's bridge finding algorithm
///
/// Stores the algorithm's state during depth-first search:
/// - Visited nodes
/// - Discovery times
/// - Low values
/// - Parent pointers
/// - Found bridges
/// - Current time step
#[derive(Debug)]
pub struct BridgeContext {
    /// Set of visited node IDs
    visited: HashSet<usize>,
    /// Discovery time for each node
    disc: HashMap<usize, u32>,
    /// Lowest discovery time reachable from each node
    low: HashMap<usize, u32>,
    /// Parent nodes in DFS tree
    parent: HashMap<usize, usize>,
    /// List of found bridges (node pairs)
    bridges: Vec<(usize, usize)>,
    /// Current time step counter
    time: u32,
}

impl BridgeContext {
    /// Creates a new empty BridgeContext
    fn new() -> Self {
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

/// Trait for finding bridge edges in undirected graphs
///
/// A bridge edge is an edge that, when removed, increases the number of
/// connected components in the graph. This implementation uses Tarjan's algorithm
/// with O(V + E) time complexity.
///
/// # Examples
///
/// Finding bridges in a linear graph:
/// ```rust
/// use xgraph::graph::graph::Graph;
/// use xgraph::algorithms::bridges::Bridges;
///
/// let mut graph = Graph::new(false);
/// let nodes = (0..4).map(|_| graph.add_node(())).collect::<Vec<_>>();
///
/// graph.add_edge(nodes[0], nodes[1], 1, ()).unwrap();
/// graph.add_edge(nodes[1], nodes[2], 1, ()).unwrap();
/// graph.add_edge(nodes[2], nodes[3], 1, ()).unwrap();
///
/// assert_eq!(graph.find_bridges(), vec![
///     (nodes[0], nodes[1]),
///     (nodes[1], nodes[2]),
///     (nodes[2], nodes[3])
/// ]);
/// ```
///
/// No bridges in a cyclic graph:
/// ```rust
/// use xgraph::graph::graph::Graph;
/// use xgraph::algorithms::bridges::Bridges;
///
/// let mut graph = Graph::new(false);
/// let nodes = (0..3).map(|_| graph.add_node(())).collect::<Vec<_>>();
///
/// graph.add_edge(nodes[0], nodes[1], 1, ()).unwrap();
/// graph.add_edge(nodes[1], nodes[2], 1, ()).unwrap();
/// graph.add_edge(nodes[2], nodes[0], 1, ()).unwrap();
///
/// assert!(graph.find_bridges().is_empty());
/// ```
pub trait Bridges {
    /// Finds all bridges in the graph using Tarjan's algorithm
    ///
    /// # Returns
    /// Vector of tuples representing bridge edges, sorted by node IDs
    ///
    /// # Examples
    /// ```rust
    /// # use xgraph::graph::graph::Graph;
    /// # use xgraph::algorithms::bridges::Bridges;
    /// let mut graph = Graph::new(false);
    /// let a = graph.add_node("A");
    /// let b = graph.add_node("B");
    /// let c = graph.add_node("C");
    ///
    /// graph.add_edge(a, b, 1, "AB").unwrap();
    /// graph.add_edge(b, c, 1, "BC").unwrap();
    ///
    /// let bridges = graph.find_bridges();
    /// assert_eq!(bridges, vec![(a, b), (b, c)]);
    /// ```
    fn find_bridges(&self) -> Vec<(usize, usize)>;

    /// Normalizes bridge representation by sorting node pairs
    ///
    /// Ensures bridges are always represented with smaller node ID first
    /// and sorts the list lexicographically
    ///
    /// # Arguments
    /// * `bridges` - Mutable reference to vector of bridges
    ///
    /// # Examples
    /// ```rust
    /// # use xgraph::algorithms::bridges::Bridges;
    /// # use xgraph::graph::graph::Graph;
    /// let mut bridges = vec![(2, 1), (3, 0)];
    /// <Graph<(), (), ()> as Bridges>::sort_bridges(&mut bridges);
    /// assert_eq!(bridges, vec![(0, 3), (1, 2)]);
    /// ```
    fn sort_bridges(bridges: &mut Vec<(usize, usize)>);

    /// Internal DFS implementation for bridge detection
    ///
    /// # Arguments
    /// * `node` - Current node being visited
    /// * `context` - Mutable reference to algorithm context
    fn bridge_dfs(&self, node: usize, context: &mut BridgeContext);
}

impl<W, N, E> Bridges for Graph<W, N, E>
where
    W: Copy + Default + PartialEq,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    /// Finds all bridges in the graph using Tarjan's algorithm
    ///
    /// This implementation handles both connected and disconnected graphs,
    /// and works for graphs with self-loops and multiple edges.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::bridges::Bridges;
    ///
    /// let mut graph = Graph::new(false);
    /// let a = graph.add_node(());
    /// let b = graph.add_node(());
    ///
    /// graph.add_edge(a, b, 1, ()).unwrap();
    /// graph.add_edge(a, a, 1, ()).unwrap(); // Self-loop
    ///
    /// assert_eq!(graph.find_bridges(), vec![(a, b)]);
    /// ```
    fn find_bridges(&self) -> Vec<(usize, usize)> {
        let mut context = BridgeContext::new();

        for node in self.all_nodes().map(|(id, _)| id) {
            if !context.visited.contains(&node) {
                self.bridge_dfs(node, &mut context);
            }
        }

        Self::sort_bridges(&mut context.bridges);
        context.bridges
    }

    /// Sorts bridges lexicographically and normalizes node order
    ///
    /// This method ensures:
    /// - For each bridge (u, v), u <= v
    /// - Bridges are sorted in lexicographical order
    fn sort_bridges(bridges: &mut Vec<(usize, usize)>) {
        bridges.iter_mut().for_each(|(u, v)| {
            if u > v {
                std::mem::swap(u, v);
            }
        });
        bridges.sort_unstable();
    }

    /// Performs the depth-first search part of Tarjan's algorithm
    ///
    /// Updates discovery times, low values, and detects bridges through
    /// recursive DFS traversal. Modifies the BridgeContext in-place.
    fn bridge_dfs(&self, node: usize, context: &mut BridgeContext) {
        context.time += 1;
        context.disc.insert(node, context.time);
        context.low.insert(node, context.time);
        context.visited.insert(node);

        for &(neighbor, _) in &self.nodes[node].neighbors {
            if !context.visited.contains(&neighbor) {
                context.parent.insert(neighbor, node);
                self.bridge_dfs(neighbor, context);

                let node_low = *context.low.get(&node).unwrap();
                let neighbor_low = *context.low.get(&neighbor).unwrap();
                context.low.insert(node, node_low.min(neighbor_low));

                if *context.low.get(&neighbor).unwrap() > *context.disc.get(&node).unwrap() {
                    context.bridges.push((node, neighbor));
                }
            } else if context.parent.get(&node) != Some(&neighbor) {
                let node_low = *context.low.get(&node).unwrap();
                let neighbor_disc = *context.disc.get(&neighbor).unwrap();
                context.low.insert(node, node_low.min(neighbor_disc));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test finding bridges in a simple graph.
    #[test]
    fn test_find_bridges_simple() {
        let mut graph = Graph::<u32, (), ()>::new(false);
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        graph.add_edge(n0, n1, 1, ()).unwrap();
        graph.add_edge(n1, n2, 1, ()).unwrap();

        let bridges = graph.find_bridges();
        assert_eq!(bridges, vec![(n0, n1), (n1, n2)]);
    }

    /// Test finding bridges in a graph with no bridges.
    #[test]
    fn test_find_bridges_no_bridges() {
        let mut graph = Graph::<u32, (), ()>::new(false);
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        graph.add_edge(n0, n1, 1, ()).unwrap();
        graph.add_edge(n1, n2, 1, ()).unwrap();
        graph.add_edge(n2, n0, 1, ()).unwrap();

        let bridges = graph.find_bridges();
        assert!(bridges.is_empty());
    }

    /// Test finding bridges in a complex graph.
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

        let graph = Graph::from_adjacency_matrix(&matrix, false, (), ()).unwrap();
        let bridges = graph.find_bridges();
        assert_eq!(bridges, vec![(2, 3), (8, 9)]);
    }

    /// Test finding bridges in an empty graph.
    #[test]
    fn test_find_bridges_empty_graph() {
        let graph = Graph::<u32, (), ()>::new(false);
        assert!(graph.find_bridges().is_empty());
    }

    /// Test finding bridges in a graph with a single node.
    #[test]
    fn test_find_bridges_single_node() {
        let mut graph = Graph::<u32, (), ()>::new(false);
        graph.add_node(());
        assert!(graph.find_bridges().is_empty());
    }
}
