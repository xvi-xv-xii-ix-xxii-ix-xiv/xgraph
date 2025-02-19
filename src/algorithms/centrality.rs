//! Centrality measure algorithms for graph analysis
//!
//! Provides implementations for three key centrality measures:
//! - Degree Centrality: Number of connections per node
//! - Betweenness Centrality: Importance as network bridge
//! - Closeness Centrality: Average distance to other nodes
//!
//! # Examples
//!
//! Basic usage:
//! ```rust
//! use xgraph::graph::graph::Graph;
//! use xgraph::algorithms::centrality::Centrality;
//!
//! let matrix = vec![
//!     vec![0, 1, 1],
//!     vec![1, 0, 1],
//!     vec![1, 1, 0]
//! ];
//! let graph = Graph::from_adjacency_matrix(&matrix, false, 0, 0).unwrap();
//!
//! println!("Degree Centrality: {:?}", graph.degree_centrality());
//! println!("Betweenness Centrality: {:?}", graph.betweenness_centrality());
//! println!("Closeness Centrality: {:?}", graph.closeness_centrality());
//! ```

use crate::graph::graph::Graph;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;

/// Trait for calculating centrality measures in graphs
///
/// # Examples
///
/// Analyzing social network influence:
/// ```rust
/// use xgraph::graph::graph::Graph;
/// use xgraph::algorithms::centrality::Centrality;
///
/// let mut social_graph = Graph::new(false);
/// let alice = social_graph.add_node("Alice");
/// let bob = social_graph.add_node("Bob");
/// let charlie = social_graph.add_node("Charlie");
///
/// social_graph.add_edge(alice, bob, 1, ()).unwrap();
/// social_graph.add_edge(bob, charlie, 1, ()).unwrap();
///
/// let betweenness = social_graph.betweenness_centrality();
/// assert!(betweenness[&bob] > 0.9);
/// ```
pub trait Centrality<W, N, E> {
    /// Calculates degree centrality for all nodes
    ///
    /// Degree centrality is defined as the number of edges incident to a node.
    /// Higher values indicate more connected nodes.
    ///
    /// # Returns
    /// HashMap<NodeId, usize> with centrality scores
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::centrality::Centrality;
    ///
    /// let matrix = vec![
    ///     vec![0, 1, 0],
    ///     vec![1, 0, 1],
    ///     vec![0, 1, 0]
    /// ];
    /// let graph = Graph::from_adjacency_matrix(&matrix, false, 0, 0).unwrap();
    ///
    /// let centrality = graph.degree_centrality();
    /// assert_eq!(centrality[&1], 2);
    /// ```
    fn degree_centrality(&self) -> HashMap<usize, usize>;

    /// Calculates betweenness centrality for all nodes
    ///
    /// Betweenness centrality measures how often a node appears on shortest paths
    /// between other nodes. Uses Brandes' algorithm (O(nm) time complexity).
    ///
    /// # Returns
    /// HashMap<NodeId, f64> with normalized centrality scores
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::centrality::Centrality;
    ///
    /// let matrix = vec![
    ///     vec![0, 1, 0, 0],
    ///     vec![1, 0, 1, 1],
    ///     vec![0, 1, 0, 0],
    ///     vec![0, 1, 0, 0]
    /// ];
    /// let graph = Graph::from_adjacency_matrix(&matrix, false, 0, 0).unwrap();
    ///
    /// let centrality = graph.betweenness_centrality();
    /// assert!((centrality[&1] - 1.0).abs() < 1e-5);
    /// ```
    fn betweenness_centrality(&self) -> HashMap<usize, f64>;

    /// Calculates closeness centrality for all nodes
    ///
    /// Closeness centrality is the reciprocal of the average shortest path distance
    /// to all other nodes. Higher values indicate more central positions.
    ///
    /// # Returns
    /// HashMap<NodeId, f64> with centrality scores
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::centrality::Centrality;
    ///
    /// let matrix = vec![
    ///     vec![0, 1, 1],
    ///     vec![1, 0, 1],
    ///     vec![1, 1, 0]
    /// ];
    /// let graph = Graph::from_adjacency_matrix(&matrix, false, 0, 0).unwrap();
    ///
    /// let centrality = graph.closeness_centrality();
    /// assert!((centrality[&0] - 1.0).abs() < 1e-5);
    /// ```
    fn closeness_centrality(&self) -> HashMap<usize, f64>;
}

impl<W, N, E> Centrality<W, N, E> for Graph<W, N, E>
where
    W: Copy + Default + PartialEq,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    fn degree_centrality(&self) -> HashMap<usize, usize> {
        self.nodes
            .iter()
            .map(|(id, node)| (id, node.neighbors.len()))
            .collect()
    }

    fn betweenness_centrality(&self) -> HashMap<usize, f64> {
        let nodes: Vec<usize> = self.all_nodes().map(|(id, _)| id).collect();
        let num_nodes = nodes.len();
        let mut betweenness = HashMap::new();

        if num_nodes <= 1 {
            return nodes.iter().map(|&n| (n, 0.0)).collect();
        }

        for &s in &nodes {
            let mut stack = Vec::new();
            let mut pred: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
            let mut sigma = vec![0.0; self.nodes.len()];
            let mut dist: Vec<Option<i32>> = vec![None; self.nodes.len()];

            sigma[s] = 1.0;
            dist[s] = Some(0);

            let mut queue = VecDeque::new();
            queue.push_back(s);

            while let Some(v) = queue.pop_front() {
                stack.push(v);
                for &(w, _) in &self.nodes[v].neighbors {
                    if dist[w].is_none() {
                        dist[w] = Some(dist[v].unwrap() + 1);
                        queue.push_back(w);
                    }
                    if dist[w] == Some(dist[v].unwrap() + 1) {
                        sigma[w] += sigma[v];
                        pred[w].push(v);
                    }
                }
            }

            let mut delta = vec![0.0; self.nodes.len()];
            while let Some(w) = stack.pop() {
                for &v in &pred[w] {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
                if w != s {
                    *betweenness.entry(w).or_insert(0.0) += delta[w];
                }
            }
        }

        if !self.directed {
            betweenness.values_mut().for_each(|v| *v /= 2.0);
        }

        let scale = if self.directed {
            1.0 / ((num_nodes - 1) * (num_nodes - 2)) as f64
        } else {
            1.0 / ((num_nodes - 1) * (num_nodes - 2) / 2) as f64
        };

        betweenness.iter_mut().for_each(|(_, v)| *v *= scale);
        betweenness
    }

    fn closeness_centrality(&self) -> HashMap<usize, f64> {
        let nodes: Vec<usize> = self.all_nodes().map(|(id, _)| id).collect();
        let mut closeness = HashMap::new();

        for &node in &nodes {
            let mut dist = HashMap::new();
            let mut queue = VecDeque::new();

            dist.insert(node, 0);
            queue.push_back(node);

            while let Some(v) = queue.pop_front() {
                for &(w, _) in &self.nodes[v].neighbors {
                    if !dist.contains_key(&w) {
                        dist.insert(w, dist[&v] + 1);
                        queue.push_back(w);
                    }
                }
            }

            let total_dist: i32 = dist.values().sum();
            let reachable = dist.len().saturating_sub(1);

            let centrality = if total_dist > 0 && reachable > 0 {
                reachable as f64 / total_dist as f64
            } else {
                0.0
            };

            closeness.insert(node, centrality);
        }

        closeness
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::approx_eq;

    const TOLERANCE: f64 = 1e-5;

    fn assert_approx_eq(a: f64, b: f64) {
        assert!(approx_eq!(f64, a, b, epsilon = TOLERANCE), "{} ≈ {}", a, b);
    }

    #[test]
    fn test_degree_centrality() {
        let matrix = vec![vec![0, 1, 0], vec![1, 0, 1], vec![0, 1, 0]];
        let graph = Graph::from_adjacency_matrix(&matrix, false, (), ()).unwrap();
        let centrality = graph.degree_centrality();
        assert_eq!(centrality[&0], 1);
        assert_eq!(centrality[&1], 2);
        assert_eq!(centrality[&2], 1);
    }

    #[test]
    fn test_betweenness_centrality() {
        let matrix = vec![
            vec![0, 1, 0, 0],
            vec![1, 0, 1, 1],
            vec![0, 1, 0, 0],
            vec![0, 1, 0, 0],
        ];

        let graph = Graph::from_adjacency_matrix(&matrix, false, (), ()).unwrap();
        let betweenness = graph.betweenness_centrality();
        assert_approx_eq(betweenness[&1], 1.0);
    }

    #[test]
    fn test_closeness_centrality() {
        let matrix = vec![vec![0, 1, 1], vec![1, 0, 1], vec![1, 1, 0]];
        let graph = Graph::from_adjacency_matrix(&matrix, false, (), ()).unwrap();
        let closeness = graph.closeness_centrality();
        assert_approx_eq(closeness[&0], 1.0);
    }

    #[test]
    fn test_empty_graph() {
        let graph = Graph::<u32, (), ()>::new(false);
        assert!(graph.degree_centrality().is_empty());
        assert!(graph.betweenness_centrality().is_empty());
        assert!(graph.closeness_centrality().is_empty());
    }
}
