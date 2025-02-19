//! Graph connectivity analysis algorithms
//!
//! Provides functionality to determine:
//! - Weakly connected components (for undirected graphs)
//! - Strongly connected components (for directed graphs)
//! - Overall graph connectivity
//!
//! # Examples
//!
//! Basic usage:
//! ```rust
//! use xgraph::graph::graph::Graph;
//! use xgraph::algorithms::connectivity::Connectivity;
//!
//! // Create a directed graph
//! let mut graph = Graph::<u32, &str, ()>::new(true);
//! let a = graph.add_node("A");
//! let b = graph.add_node("B");
//! graph.add_edge(a, b, 1, ()).unwrap();
//!
//! println!("Strongly Connected Components: {:?}",
//!     graph.find_strongly_connected_components());
//! println!("Is strongly connected: {}", graph.is_strongly_connected());
//! ```

use crate::graph::graph::Graph;
use crate::graph::node::Node;
use std::collections::{HashSet, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;

/// Trait for graph connectivity analysis
///
/// Provides methods to analyze both weak and strong connectivity
/// in directed and undirected graphs.
///
/// # Examples
///
/// Finding connected components in a social network:
/// ```rust
/// use xgraph::graph::graph::Graph;
/// use xgraph::algorithms::connectivity::Connectivity;
///
/// let mut social_graph = Graph::new(false);
/// let alice = social_graph.add_node("Alice");
/// let bob = social_graph.add_node("Bob");
/// let charlie = social_graph.add_node("Charlie");
///
/// social_graph.add_edge(alice, bob, 1, ()).unwrap();
/// social_graph.add_edge(charlie, bob, 1, ()).unwrap();
///
/// let components = social_graph.find_connected_components();
/// assert_eq!(components.len(), 1);
/// ```
pub trait Connectivity<W, N, E> {
    /// Finds weakly connected components using BFS
    ///
    /// Treats the graph as undirected by considering all edge connections
    /// regardless of direction.
    ///
    /// # Returns
    /// Vec of node ID vectors representing components
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::connectivity::Connectivity;
    ///
    /// let mut graph = Graph::new(true);
    /// let a = graph.add_node("A");
    /// let b = graph.add_node("B");
    /// graph.add_edge(a, b, 1, ()).unwrap();
    ///
    /// let components = graph.find_weakly_connected_components();
    /// assert_eq!(components, vec![vec![a, b]]);
    /// ```
    fn find_weakly_connected_components(&self) -> Vec<Vec<usize>>;

    /// Finds strongly connected components using Kosaraju's algorithm
    ///
    /// Two-phase algorithm with O(|V| + |E|) complexity:
    /// 1. DFS to determine finish order
    /// 2. DFS on transposed graph in reverse finish order
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::connectivity::Connectivity;
    ///
    /// let mut graph = Graph::new(true);
    /// let nodes = vec![0, 1, 2].into_iter()
    ///     .map(|i| graph.add_node(i))
    ///     .collect::<Vec<_>>();
    ///
    /// graph.add_edge(nodes[0], nodes[1], 1, ()).unwrap();
    /// graph.add_edge(nodes[1], nodes[2], 1, ()).unwrap();
    /// graph.add_edge(nodes[2], nodes[0], 1, ()).unwrap();
    ///
    /// let scc = graph.find_strongly_connected_components();
    /// assert_eq!(scc.len(), 1);
    /// ```
    fn find_strongly_connected_components(&self) -> Vec<Vec<usize>>;

    /// Automatically selects appropriate connectivity type
    ///
    /// # Returns
    /// Strongly connected components for directed graphs,
    /// weakly connected components for undirected graphs
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::connectivity::Connectivity;
    ///
    /// let mut directed_graph = Graph::<u32, (), ()>::new(true);
    /// let a = directed_graph.add_node(());
    /// let b = directed_graph.add_node(());
    /// directed_graph.add_edge(a, b, 1, ()).unwrap();
    ///
    /// let mut undirected_graph = Graph::<u32, (), ()>::new(false);
    /// let c = undirected_graph.add_node(());
    /// let d = undirected_graph.add_node(());
    /// undirected_graph.add_edge(c, d, 1, ()).unwrap();
    ///
    /// assert_ne!(
    ///     directed_graph.find_connected_components().len(),
    ///     undirected_graph.find_connected_components().len()
    /// );
    /// ```
    fn find_connected_components(&self) -> Vec<Vec<usize>> {
        if self.is_directed() {
            self.find_strongly_connected_components()
        } else {
            self.find_weakly_connected_components()
        }
    }

    /// Checks weak connectivity using BFS
    ///
    /// A graph is weakly connected if there's a path between every pair of nodes
    /// when considering all edges as undirected.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::connectivity::Connectivity;
    ///
    /// let mut graph = Graph::new(true);
    /// let a = graph.add_node(0);
    /// let b = graph.add_node(1);
    /// graph.add_edge(a, b, 1, ()).unwrap();
    ///
    /// assert!(graph.is_weakly_connected());
    /// assert!(!graph.is_strongly_connected());
    /// ```
    fn is_weakly_connected(&self) -> bool;

    /// Checks strong connectivity using Kosaraju's algorithm
    ///
    /// A graph is strongly connected if there's a directed path between
    /// every pair of nodes.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::connectivity::Connectivity;
    ///
    /// let mut graph = Graph::new(true);
    /// let nodes = (0..3).map(|i| graph.add_node(i)).collect::<Vec<_>>();
    ///
    /// for i in 0..3 {
    ///     graph.add_edge(nodes[i], nodes[(i+1)%3], 1, ()).unwrap();
    /// }
    ///
    /// assert!(graph.is_strongly_connected());
    /// ```
    fn is_strongly_connected(&self) -> bool;

    /// Checks overall connectivity based on graph type
    ///
    /// # Returns
    /// Strong connectivity for directed graphs,
    /// weak connectivity for undirected graphs
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::connectivity::Connectivity;
    ///
    /// let mut directed = Graph::<u32, &str, ()>::new(true);
    /// let mut undirected = Graph::<u32, &str, ()>::new(false);
    ///
    /// assert_ne!(directed.is_connected(), undirected.is_connected());
    /// ```
    fn is_connected(&self) -> bool {
        if self.is_directed() {
            self.is_strongly_connected()
        } else {
            self.is_weakly_connected()
        }
    }

    /// Determines graph directionality
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::connectivity::Connectivity;
    ///
    /// let directed = Graph::<u32, (), ()>::new(true);
    /// let undirected = Graph::<u32, (), ()>::new(false);
    ///
    /// assert!(directed.is_directed());
    /// assert!(!undirected.is_directed());
    /// ```
    fn is_directed(&self) -> bool;
}

impl<W, N, E> Connectivity<W, N, E> for Graph<W, N, E>
where
    W: Copy + Default + PartialEq + Debug,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    /// Finds the weakly connected components (ignores edge direction).
    fn find_weakly_connected_components(&self) -> Vec<Vec<usize>> {
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

                    // Gather all neighbors (including reverse edges for directed graphs)
                    let mut neighbors = self
                        .get_neighbors(current)
                        .iter()
                        .map(|(n, _)| *n)
                        .collect::<Vec<_>>();

                    if self.is_directed() {
                        neighbors.extend(self.get_predecessors(current));
                    }

                    for neighbor in neighbors {
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            queue.push_back(neighbor);
                        }
                    }
                }
                components.push(component);
            }
        }
        components
    }

    /// Finds the strongly connected components using Kosaraju's algorithm.
    fn find_strongly_connected_components(&self) -> Vec<Vec<usize>> {
        // First DFS pass to determine finish order
        let mut visited = HashSet::new();
        let mut order = Vec::with_capacity(self.nodes.len());

        for node in self.nodes.iter().map(|(id, _)| id) {
            if !visited.contains(&node) {
                self.dfs_order(node, &mut visited, &mut order);
            }
        }

        // Transpose the graph
        let transposed = self.transpose();

        // Second DFS pass in reverse finish order
        let mut visited = HashSet::new();
        let mut components = Vec::new();

        for &node in order.iter().rev() {
            if !visited.contains(&node) {
                let mut component = Vec::new();
                transposed.dfs_collect(node, &mut visited, &mut component);
                component.sort(); // For test stability
                components.push(component);
            }
        }

        components
    }

    /// Checks whether the graph is weakly connected.
    fn is_weakly_connected(&self) -> bool {
        let components = self.find_weakly_connected_components();
        components.len() == 1
    }

    /// Checks whether the graph is strongly connected.
    fn is_strongly_connected(&self) -> bool {
        if self.nodes.is_empty() {
            return true;
        }

        let start_node = self.nodes.iter().next().unwrap().0;
        self.strong_connectivity_check(start_node)
    }

    /// Checks whether the graph is directed.
    fn is_directed(&self) -> bool {
        self.directed
    }
}

// Private methods implementation for Graph
impl<W, N, E> Graph<W, N, E>
where
    W: Copy + Default + PartialEq + Debug,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    /// Perform DFS to determine finish order of nodes.
    fn dfs_order(&self, node: usize, visited: &mut HashSet<usize>, order: &mut Vec<usize>) {
        visited.insert(node);
        for (neighbor, _) in self.get_neighbors(node) {
            if !visited.contains(&neighbor) {
                self.dfs_order(neighbor, visited, order);
            }
        }
        order.push(node);
    }

    /// Perform DFS to collect nodes of a strongly connected component.
    fn dfs_collect(&self, node: usize, visited: &mut HashSet<usize>, component: &mut Vec<usize>) {
        visited.insert(node);
        component.push(node);
        for (neighbor, _) in self.get_neighbors(node) {
            if !visited.contains(&neighbor) {
                self.dfs_collect(neighbor, visited, component);
            }
        }
    }

    /// Checks the strong connectivity of the graph starting from a given node.
    fn strong_connectivity_check(&self, start: usize) -> bool {
        let mut forward_visited = HashSet::new();
        self.dfs_collect(start, &mut forward_visited, &mut vec![]);

        if forward_visited.len() != self.nodes.len() {
            return false;
        }

        let transposed = self.transpose();
        let mut backward_visited = HashSet::new();
        transposed.dfs_collect(start, &mut backward_visited, &mut vec![]);

        backward_visited.len() == self.nodes.len()
    }

    /// Retrieves the predecessors of a node (only used for directed graphs).
    fn get_predecessors(&self, node: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|(_, e)| e.to == node)
            .map(|(_, e)| e.from)
            .collect()
    }

    /// Transposes the graph by reversing all edge directions.
    fn transpose(&self) -> Self {
        let mut transposed = Graph::new(true);

        // Copy nodes with original IDs
        let mut nodes: Vec<_> = self.nodes.iter().map(|(id, _)| id).collect();
        nodes.sort();

        for &id in &nodes {
            let node = self.nodes.get(id).unwrap();
            transposed.nodes.insert(Node {
                data: node.data.clone(),
                neighbors: Vec::new(),
                attributes: node.attributes.clone(),
            });
        }

        // Add reversed edges through add_edge
        for (_, edge) in self.edges.iter() {
            transposed
                .add_edge(edge.to, edge.from, edge.weight, edge.data.clone())
                .expect("Invalid edge in transpose");

            // Copy edge attributes
            if let Some(attrs) = self.get_all_edge_attributes(edge.from, edge.to) {
                for (k, v) in attrs {
                    transposed.set_edge_attribute(edge.to, edge.from, k.clone(), v.clone());
                }
            }
        }

        transposed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test if the graph is strongly connected.
    #[test]
    fn test_strongly_connected() {
        let mut graph = Graph::<u32, (), ()>::new(true);

        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        graph.add_edge(n0, n1, 1, ()).unwrap();
        graph.add_edge(n1, n2, 1, ()).unwrap();
        graph.add_edge(n2, n0, 1, ()).unwrap();

        assert!(graph.is_strongly_connected());
        let scc = graph.find_strongly_connected_components();
        assert_eq!(scc.len(), 1);
    }

    /// Test if the graph is weakly connected.
    #[test]
    fn test_weak_connectivity() {
        let mut graph = Graph::<u32, (), ()>::new(true);

        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        graph.add_edge(n0, n1, 1, ()).unwrap();

        assert!(!graph.is_strongly_connected());
        assert!(graph.is_weakly_connected());
    }

    /// Test if the transpose function works correctly.
    #[test]
    fn test_transpose() {
        let mut graph = Graph::<u32, (), ()>::new(true);
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        graph.add_edge(n0, n1, 1, ()).unwrap();

        let transposed = graph.transpose();
        assert_eq!(transposed.get_neighbors(n1), vec![(n0, 1)]);
    }
}
