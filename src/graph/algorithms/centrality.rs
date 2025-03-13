//! Centrality measure algorithms for graph analysis
//!
//! Provides implementations for three key centrality measures:
//! - Degree Centrality: Measures node connectivity through edge count
//! - Betweenness Centrality: Quantifies a node's role as a network bridge
//! - Closeness Centrality: Assesses average distance to other nodes
//!
//! All methods return `Result` types to handle potential errors gracefully instead of panicking.
//!
//! # Examples
//!
//! Basic usage with error handling:
//! ```rust
//! use xgraph::graph::graph::Graph;
//! use xgraph::graph::algorithms::centrality::Centrality;
//!
//! let matrix = vec![
//!     vec![0, 1, 1],
//!     vec![1, 0, 1],
//!     vec![1, 1, 0]
//! ];
//! let graph = Graph::from_adjacency_matrix(&matrix, false, 0, 0).unwrap();
//!
//! let degree = graph.degree_centrality().unwrap();
//! let betweenness = graph.betweenness_centrality().unwrap();
//! let closeness = graph.closeness_centrality().unwrap();
//! ```

use crate::graph::graph::Graph;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;

/// Error type for centrality computation failures.
#[derive(Debug)]
pub enum CentralityError {
    /// Indicates an invalid node was encountered during computation.
    InvalidNode(usize),
    /// Indicates an overflow occurred during distance calculation.
    DistanceOverflow,
}

impl std::fmt::Display for CentralityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CentralityError::InvalidNode(id) => {
                write!(f, "Invalid node reference: node ID {} not found", id)
            }
            CentralityError::DistanceOverflow => {
                write!(f, "Distance calculation overflowed")
            }
        }
    }
}

impl std::error::Error for CentralityError {}

/// Trait for calculating centrality measures in graphs
///
/// Provides methods to compute various centrality metrics with proper error handling.
///
/// # Examples
///
/// Analyzing social network influence:
/// ```rust
/// use xgraph::graph::graph::Graph;
/// use xgraph::graph::algorithms::centrality::Centrality;
///
/// let mut social_graph = Graph::new(false);
/// let alice = social_graph.add_node("Alice");
/// let bob = social_graph.add_node("Bob");
/// let charlie = social_graph.add_node("Charlie");
///
/// social_graph.add_edge(alice, bob, 1, ()).unwrap();
/// social_graph.add_edge(bob, charlie, 1, ()).unwrap();
///
/// let betweenness = social_graph.betweenness_centrality().unwrap();
/// assert!(betweenness[&bob] > 0.9);
/// ```
pub trait Centrality<W, N, E> {
    /// Calculates degree centrality for all nodes
    ///
    /// Degree centrality is defined as the number of edges incident to a node.
    /// Higher values indicate more connected nodes.
    ///
    /// # Returns
    /// - `Ok(HashMap<usize, usize>)`: Mapping of node IDs to their degree centrality scores
    /// - `Err(CentralityError)`: If computation fails due to invalid nodes
    fn degree_centrality(&self) -> Result<HashMap<usize, usize>, CentralityError>;

    /// Calculates betweenness centrality for all nodes
    ///
    /// Betweenness centrality measures how often a node appears on shortest paths
    /// between other nodes using Brandes' algorithm (O(nm) time complexity).
    /// Scores are normalized to the range [0, 1].
    ///
    /// # Returns
    /// - `Ok(HashMap<usize, f64>)`: Mapping of node IDs to their normalized betweenness scores
    /// - `Err(CentralityError)`: If computation fails
    fn betweenness_centrality(&self) -> Result<HashMap<usize, f64>, CentralityError>;

    /// Calculates closeness centrality for all nodes
    ///
    /// Closeness centrality is the reciprocal of the average shortest path distance
    /// to all other nodes. Higher values indicate more central positions.
    ///
    /// # Returns
    /// - `Ok(HashMap<usize, f64>)`: Mapping of node IDs to their closeness scores
    /// - `Err(CentralityError)`: If computation fails due to overflow or invalid nodes
    fn closeness_centrality(&self) -> Result<HashMap<usize, f64>, CentralityError>;
}

impl<W, N, E> Centrality<W, N, E> for Graph<W, N, E>
where
    W: Copy + Default + PartialEq,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    fn degree_centrality(&self) -> Result<HashMap<usize, usize>, CentralityError> {
        let mut centrality = HashMap::new();
        for (id, node) in self.nodes.iter() {
            centrality.insert(id, node.neighbors.len());
        }
        Ok(centrality)
    }

    fn betweenness_centrality(&self) -> Result<HashMap<usize, f64>, CentralityError> {
        let nodes: Vec<usize> = self.all_nodes().map(|(id, _)| id).collect();
        let num_nodes = nodes.len();
        let mut betweenness = HashMap::new();

        if num_nodes <= 1 {
            return Ok(nodes.into_iter().map(|n| (n, 0.0)).collect());
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
                let neighbors = self.nodes.get(v).ok_or(CentralityError::InvalidNode(v))?;
                for &(w, _) in &neighbors.neighbors {
                    if dist[w].is_none() {
                        // Fixed: Wrap the result in Some() and specify i32 type
                        dist[w] = Some(
                            dist[v]
                                .unwrap()
                                .checked_add(1_i32)
                                .ok_or(CentralityError::DistanceOverflow)?,
                        );
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
        Ok(betweenness)
    }

    fn closeness_centrality(&self) -> Result<HashMap<usize, f64>, CentralityError> {
        let nodes: Vec<usize> = self.all_nodes().map(|(id, _)| id).collect();
        let mut closeness = HashMap::new();

        for &node in &nodes {
            let mut dist = HashMap::new();
            let mut queue = VecDeque::new();

            dist.insert(node, 0_i32); // Specify type as i32
            queue.push_back(node);

            while let Some(v) = queue.pop_front() {
                let neighbors = self.nodes.get(v).ok_or(CentralityError::InvalidNode(v))?;
                for &(w, _) in &neighbors.neighbors {
                    if !dist.contains_key(&w) {
                        // Fixed: Specify i32 type for checked_add
                        let new_dist = dist[&v]
                            .checked_add(1_i32)
                            .ok_or(CentralityError::DistanceOverflow)?;
                        dist.insert(w, new_dist);
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

        Ok(closeness)
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
        let centrality = graph.degree_centrality().unwrap();
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
        let betweenness = graph.betweenness_centrality().unwrap();
        assert_approx_eq(betweenness[&1], 1.0);
    }

    #[test]
    fn test_closeness_centrality() {
        let matrix = vec![vec![0, 1, 1], vec![1, 0, 1], vec![1, 1, 0]];
        let graph = Graph::from_adjacency_matrix(&matrix, false, (), ()).unwrap();
        let closeness = graph.closeness_centrality().unwrap();
        assert_approx_eq(closeness[&0], 1.0);
    }

    #[test]
    fn test_empty_graph() {
        let graph = Graph::<u32, (), ()>::new(false);
        assert!(graph.degree_centrality().unwrap().is_empty());
        assert!(graph.betweenness_centrality().unwrap().is_empty());
        assert!(graph.closeness_centrality().unwrap().is_empty());
    }
}
