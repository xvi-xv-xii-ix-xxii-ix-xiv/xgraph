//! Heterogeneous centrality measure algorithms for analyzing heterogeneous multigraphs.
//!
//! This module provides implementations of various centrality measures tailored for heterogeneous
//! multigraphs, where nodes and edges can have distinct types. Each measure is available in two forms:
//! one that considers all edges in the graph and another that operates only on edges of specified types.
//! The centrality measures include degree, betweenness, closeness, eigenvector, PageRank, and Katz centrality.

#[cfg(feature = "hgraph")]
use crate::hgraph::h_graph::HeterogeneousGraph;
use std::collections::HashSet;
#[cfg(feature = "hgraph")]
use std::collections::{HashMap, VecDeque};
#[cfg(feature = "hgraph")]
use std::fmt::Debug;
#[cfg(feature = "hgraph")]
use std::hash::Hash;
#[cfg(feature = "hgraph")]
use std::ops::{Add, AddAssign};

// Error handling additions

/// Error type for centrality computation failures.
///
/// Represents errors that may occur during centrality calculations, such as referencing non-existent edges.
#[derive(Debug)]
pub enum CentralityError {
    /// Indicates an edge referenced in the graph does not exist.
    InvalidEdgeReference(usize),
}

impl std::fmt::Display for CentralityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CentralityError::InvalidEdgeReference(id) => {
                write!(f, "Invalid edge reference: edge ID {} not found", id)
            }
        }
    }
}

impl std::error::Error for CentralityError {}

/// Result type alias for centrality computations.
///
/// Wraps the result of centrality calculations, allowing for error handling.
pub type Result<T> = std::result::Result<T, CentralityError>;

// Enhanced trait with documentation and error handling

/// Trait defining centrality measures for heterogeneous graphs.
///
/// Provides methods to compute various centrality metrics on a `HeterogeneousGraph`. Each method
/// comes in two variants: one for all edges and one filtered by edge types. All methods return
/// a `Result` to handle potential errors gracefully instead of panicking.
#[cfg(feature = "hgraph")]
pub trait HeteroCentrality<W, N, E> {
    /// Computes the degree centrality for each node.
    ///
    /// Degree centrality is the number of edges incident to a node, reflecting its direct connectivity.
    ///
    /// # Returns
    /// A `Result` containing a `HashMap` mapping node IDs to their degree centrality values.
    fn degree_centrality(&self) -> Result<HashMap<usize, usize>>;

    /// Computes the degree centrality for each node, considering only specified edge types.
    ///
    /// # Arguments
    /// * `edge_types` - A slice of edge type strings to filter edges by.
    ///
    /// # Returns
    /// A `Result` containing a `HashMap` mapping node IDs to their degree centrality values.
    fn degree_centrality_by_types(&self, edge_types: &[&str]) -> Result<HashMap<usize, usize>>;

    /// Computes the betweenness centrality for each node.
    ///
    /// Betweenness centrality measures the proportion of shortest paths passing through a node,
    /// indicating its role as a bridge in the network.
    ///
    /// # Returns
    /// A `Result` containing a `HashMap` mapping node IDs to their betweenness centrality values.
    fn betweenness_centrality(&self) -> Result<HashMap<usize, f64>>;

    /// Computes the betweenness centrality for each node, considering only specified edge types.
    ///
    /// # Arguments
    /// * `edge_types` - A slice of edge type strings to filter edges by.
    ///
    /// # Returns
    /// A `Result` containing a `HashMap` mapping node IDs to their betweenness centrality values.
    fn betweenness_centrality_by_types(&self, edge_types: &[&str]) -> Result<HashMap<usize, f64>>;

    /// Computes the closeness centrality for each node.
    ///
    /// Closeness centrality measures a node's average distance to all other nodes, reflecting its
    /// accessibility within the network.
    ///
    /// # Returns
    /// A `Result` containing a `HashMap` mapping node IDs to their closeness centrality values.
    fn closeness_centrality(&self) -> Result<HashMap<usize, f64>>;

    /// Computes the closeness centrality for each node, considering only specified edge types.
    ///
    /// # Arguments
    /// * `edge_types` - A slice of edge type strings to filter edges by.
    ///
    /// # Returns
    /// A `Result` containing a `HashMap` mapping node IDs to their closeness centrality values.
    fn closeness_centrality_by_types(&self, edge_types: &[&str]) -> Result<HashMap<usize, f64>>;

    /// Computes the eigenvector centrality for each node.
    ///
    /// Eigenvector centrality measures a node's importance based on the importance of its neighbors,
    /// computed iteratively until convergence.
    ///
    /// # Arguments
    /// * `max_iter` - Maximum number of iterations to perform.
    /// * `tolerance` - Convergence threshold for stopping iterations.
    ///
    /// # Returns
    /// A `Result` containing a `HashMap` mapping node IDs to their eigenvector centrality values.
    fn eigenvector_centrality(
        &self,
        max_iter: usize,
        tolerance: f64,
    ) -> Result<HashMap<usize, f64>>;

    /// Computes the eigenvector centrality for each node, considering only specified edge types.
    ///
    /// # Arguments
    /// * `edge_types` - A slice of edge type strings to filter edges by.
    /// * `max_iter` - Maximum number of iterations to perform.
    /// * `tolerance` - Convergence threshold for stopping iterations.
    ///
    /// # Returns
    /// A `Result` containing a `HashMap` mapping node IDs to their eigenvector centrality values.
    fn eigenvector_centrality_by_types(
        &self,
        edge_types: &[&str],
        max_iter: usize,
        tolerance: f64,
    ) -> Result<HashMap<usize, f64>>;

    /// Computes the PageRank centrality for each node.
    ///
    /// PageRank measures a node's importance based on incoming links, adjusted by a damping factor,
    /// simulating a random walk on the graph.
    ///
    /// # Arguments
    /// * `damping` - Damping factor (typically 0.85) for the random walk.
    /// * `max_iter` - Maximum number of iterations to perform.
    /// * `tolerance` - Convergence threshold for stopping iterations.
    ///
    /// # Returns
    /// A `Result` containing a `HashMap` mapping node IDs to their PageRank values.
    fn pagerank(
        &self,
        damping: f64,
        max_iter: usize,
        tolerance: f64,
    ) -> Result<HashMap<usize, f64>>;

    /// Computes the PageRank centrality for each node, considering only specified edge types.
    ///
    /// # Arguments
    /// * `edge_types` - A slice of edge type strings to filter edges by.
    /// * `damping` - Damping factor (typically 0.85) for the random walk.
    /// * `max_iter` - Maximum number of iterations to perform.
    /// * `tolerance` - Convergence threshold for stopping iterations.
    ///
    /// # Returns
    /// A `Result` containing a `HashMap` mapping node IDs to their PageRank values.
    fn pagerank_by_types(
        &self,
        edge_types: &[&str],
        damping: f64,
        max_iter: usize,
        tolerance: f64,
    ) -> Result<HashMap<usize, f64>>;

    /// Computes the Katz centrality for each node.
    ///
    /// Katz centrality measures a node's influence by considering attenuated contributions from all
    /// paths, parameterized by an attenuation factor `alpha` and a constant `beta`.
    ///
    /// # Arguments
    /// * `alpha` - Attenuation factor for path contributions.
    /// * `beta` - Constant term added to each node's centrality.
    /// * `max_iter` - Maximum number of iterations to perform.
    /// * `tolerance` - Convergence threshold for stopping iterations.
    ///
    /// # Returns
    /// A `Result` containing a `HashMap` mapping node IDs to their Katz centrality values.
    fn katz_centrality(
        &self,
        alpha: f64,
        beta: f64,
        max_iter: usize,
        tolerance: f64,
    ) -> Result<HashMap<usize, f64>>;

    /// Computes the Katz centrality for each node, considering only specified edge types.
    ///
    /// # Arguments
    /// * `edge_types` - A slice of edge type strings to filter edges by.
    /// * `alpha` - Attenuation factor for path contributions.
    /// * `beta` - Constant term added to each node's centrality.
    /// * `max_iter` - Maximum number of iterations to perform.
    /// * `tolerance` - Convergence threshold for stopping iterations.
    ///
    /// # Returns
    /// A `Result` containing a `HashMap` mapping node IDs to their Katz centrality values.
    fn katz_centrality_by_types(
        &self,
        edge_types: &[&str],
        alpha: f64,
        beta: f64,
        max_iter: usize,
        tolerance: f64,
    ) -> Result<HashMap<usize, f64>>;
}

#[cfg(feature = "hgraph")]
impl<W, N, E> HeteroCentrality<W, N, E> for HeterogeneousGraph<W, N, E>
where
    W: Copy + Default + PartialEq + PartialOrd + Add<Output = W> + AddAssign + Into<f64> + Debug,
    N: Clone + Eq + Hash + Debug + crate::hgraph::h_node::NodeType,
    E: Clone + Default + Debug + crate::hgraph::h_edge::EdgeType,
{
    fn degree_centrality(&self) -> Result<HashMap<usize, usize>> {
        Ok(self
            .nodes
            .iter()
            .map(|(id, node)| {
                let degree = node.neighbors.iter().map(|(_, edges)| edges.len()).sum();
                (id, degree)
            })
            .collect())
    }

    fn degree_centrality_by_types(&self, edge_types: &[&str]) -> Result<HashMap<usize, usize>> {
        let centrality = self
            .nodes
            .iter()
            .map(|(id, node)| {
                let degree = node
                    .neighbors
                    .iter()
                    .map(|(_, edges)| {
                        edges
                            .iter()
                            .filter(|&&(edge_id, _)| {
                                self.edges
                                    .get(edge_id)
                                    .map(|edge| {
                                        edge_types.contains(&edge.data.as_string().as_str())
                                    })
                                    .ok_or(CentralityError::InvalidEdgeReference(edge_id))
                                    .unwrap_or(false)
                            })
                            .count()
                    })
                    .sum();
                (id, degree)
            })
            .collect();
        Ok(centrality)
    }

    fn betweenness_centrality(&self) -> Result<HashMap<usize, f64>> {
        let nodes: Vec<usize> = (0..self.nodes.len())
            .filter(|&id| self.nodes.contains(id))
            .collect();
        let num_nodes = nodes.len();
        let mut betweenness = HashMap::new();

        if num_nodes <= 1 {
            return Ok(nodes.iter().map(|&n| (n, 0.0)).collect());
        }

        for &s in &nodes {
            let mut stack = Vec::new();
            let mut pred: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
            let mut sigma = vec![0.0; self.nodes.len()];
            let mut dist: Vec<Option<W>> = vec![None; self.nodes.len()];

            sigma[s] = 1.0;
            dist[s] = Some(W::default());

            let mut queue = VecDeque::new();
            queue.push_back(s);

            while let Some(v) = queue.pop_front() {
                stack.push(v);
                if let Some(node) = self.nodes.get(v) {
                    for &(w, ref edges) in &node.neighbors {
                        let min_edge_weight = edges
                            .iter()
                            .map(|&(_eid, weight)| weight)
                            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                            .unwrap();

                        let new_dist = dist[v].unwrap() + min_edge_weight;
                        if dist[w].is_none() {
                            dist[w] = Some(new_dist);
                            queue.push_back(w);
                        }
                        if dist[w] == Some(new_dist) {
                            sigma[w] += sigma[v];
                            pred[w].push(v);
                        }
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

    fn betweenness_centrality_by_types(&self, edge_types: &[&str]) -> Result<HashMap<usize, f64>> {
        // Step 1: Identify nodes in the subgraph induced by edge_types
        let mut subgraph_nodes = HashSet::new();
        for (id, node) in &self.nodes {
            for &(w, ref edges) in &node.neighbors {
                if edges.iter().any(|&(edge_id, _)| {
                    self.edges
                        .get(edge_id)
                        .map(|edge| edge_types.contains(&edge.data.as_string().as_str()))
                        .ok_or(CentralityError::InvalidEdgeReference(edge_id))
                        .unwrap_or(false)
                }) {
                    subgraph_nodes.insert(id);
                    subgraph_nodes.insert(w);
                }
            }
        }
        let nodes: Vec<usize> = subgraph_nodes.into_iter().collect();
        let num_nodes = nodes.len();
        let mut betweenness = HashMap::new();

        if num_nodes <= 1 {
            return Ok(self.nodes.iter().map(|(id, _)| (id, 0.0)).collect());
        }

        for &s in &nodes {
            let mut stack = Vec::new();
            let mut pred: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
            let mut sigma = vec![0.0; self.nodes.len()];
            let mut dist: Vec<Option<W>> = vec![None; self.nodes.len()];

            sigma[s] = 1.0;
            dist[s] = Some(W::default());

            let mut queue = VecDeque::new();
            queue.push_back(s);

            while let Some(v) = queue.pop_front() {
                stack.push(v);
                if let Some(node) = self.nodes.get(v) {
                    for &(w, ref edges) in &node.neighbors {
                        let filtered_edges: Vec<_> = edges
                            .iter()
                            .filter(|&&(edge_id, _)| {
                                self.edges
                                    .get(edge_id)
                                    .map(|edge| {
                                        edge_types.contains(&edge.data.as_string().as_str())
                                    })
                                    .ok_or(CentralityError::InvalidEdgeReference(edge_id))
                                    .unwrap_or(false)
                            })
                            .collect();

                        if filtered_edges.is_empty() {
                            continue;
                        }

                        let min_edge_weight = filtered_edges
                            .iter()
                            .map(|&(_, weight)| weight)
                            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                            .unwrap();

                        let new_dist = dist[v].unwrap() + *min_edge_weight;
                        if dist[w].is_none() {
                            dist[w] = Some(new_dist);
                            queue.push_back(w);
                        }
                        if dist[w] == Some(new_dist) {
                            sigma[w] += sigma[v];
                            pred[w].push(v);
                        }
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

        // Step 2: Normalize based on subgraph size
        let scale = if self.directed {
            1.0 / ((num_nodes - 1) * (num_nodes - 2)) as f64
        } else {
            1.0 / ((num_nodes - 1) * (num_nodes - 2) / 2) as f64
        };

        // Initialize all nodes with 0.0, then update with scaled values
        let mut result = self
            .nodes
            .iter()
            .map(|(id, _)| (id, 0.0))
            .collect::<HashMap<_, _>>();
        for (&node, &value) in &betweenness {
            result.insert(node, value * scale);
        }

        Ok(result)
    }

    fn closeness_centrality(&self) -> Result<HashMap<usize, f64>> {
        let nodes: Vec<usize> = (0..self.nodes.len())
            .filter(|&id| self.nodes.contains(id))
            .collect();
        let mut closeness = HashMap::new();

        for &node in &nodes {
            let mut dist = HashMap::new();
            let mut queue = VecDeque::new();

            dist.insert(node, W::default());
            queue.push_back(node);

            while let Some(v) = queue.pop_front() {
                if let Some(n) = self.nodes.get(v) {
                    for &(w, ref edges) in &n.neighbors {
                        let min_edge_weight = edges
                            .iter()
                            .map(|&(_eid, weight)| weight)
                            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                            .unwrap();

                        let new_dist = dist[&v] + min_edge_weight;
                        if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(w) {
                            e.insert(new_dist);
                            queue.push_back(w);
                        }
                    }
                }
            }

            let total_dist: W = dist.values().fold(W::default(), |acc, &d| acc + d);
            let reachable = dist.len().saturating_sub(1);
            let total_dist_f64: f64 = total_dist.into();
            let centrality = if total_dist_f64 > 0.0 && reachable > 0 {
                reachable as f64 / total_dist_f64
            } else {
                0.0
            };

            closeness.insert(node, centrality);
        }

        Ok(closeness)
    }

    fn closeness_centrality_by_types(&self, edge_types: &[&str]) -> Result<HashMap<usize, f64>> {
        let nodes: Vec<usize> = (0..self.nodes.len())
            .filter(|&id| self.nodes.contains(id))
            .collect();
        let mut closeness = HashMap::new();

        for &node in &nodes {
            let mut dist = HashMap::new();
            let mut queue = VecDeque::new();

            dist.insert(node, W::default());
            queue.push_back(node);

            while let Some(v) = queue.pop_front() {
                if let Some(n) = self.nodes.get(v) {
                    for &(w, ref edges) in &n.neighbors {
                        // Filter edges by types
                        let filtered_edges: Vec<_> = edges
                            .iter()
                            .filter(|&&(edge_id, _)| {
                                self.edges
                                    .get(edge_id)
                                    .map(|edge| {
                                        edge_types.contains(&edge.data.as_string().as_str())
                                    })
                                    .ok_or(CentralityError::InvalidEdgeReference(edge_id))
                                    .unwrap_or(false)
                            })
                            .collect();

                        if filtered_edges.is_empty() {
                            continue;
                        }

                        let min_edge_weight = filtered_edges
                            .iter()
                            .map(|&(_, weight)| weight)
                            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                            .unwrap();

                        let new_dist = dist[&v] + *min_edge_weight;
                        if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(w) {
                            e.insert(new_dist);
                            queue.push_back(w);
                        }
                    }
                }
            }

            let total_dist: W = dist.values().fold(W::default(), |acc, &d| acc + d);
            let reachable = dist.len().saturating_sub(1);
            let total_dist_f64: f64 = total_dist.into();
            let centrality = if total_dist_f64 > 0.0 && reachable > 0 {
                reachable as f64 / total_dist_f64
            } else {
                0.0
            };

            closeness.insert(node, centrality);
        }

        Ok(closeness)
    }

    fn eigenvector_centrality(
        &self,
        max_iter: usize,
        tolerance: f64,
    ) -> Result<HashMap<usize, f64>> {
        let nodes: Vec<usize> = (0..self.nodes.len())
            .filter(|&id| self.nodes.contains(id))
            .collect();
        let num_nodes = nodes.len();
        let mut centrality: HashMap<usize, f64> = nodes.iter().map(|&n| (n, 1.0)).collect();

        if num_nodes == 0 {
            return Ok(centrality);
        }

        for _ in 0..max_iter {
            let mut new_centrality = HashMap::new();
            let mut max_diff: f64 = 1.0;

            for &v in &nodes {
                let mut score = 1.0;
                if let Some(node) = self.nodes.get(v) {
                    for &(w, ref edges) in &node.neighbors {
                        let total_weight: f64 =
                            edges.iter().map(|&(_, weight)| weight.into()).sum();
                        score += total_weight * centrality[&w];
                    }
                }
                new_centrality.insert(v, score);
                max_diff = max_diff.max((score - centrality[&v]).abs());
            }

            let norm: f64 = new_centrality.values().map(|&v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for v in new_centrality.values_mut() {
                    *v /= norm;
                }
            }

            centrality = new_centrality;

            if max_diff < tolerance {
                break;
            }
        }

        let sum: f64 = centrality.values().sum();
        if sum > 0.0 {
            for v in centrality.values_mut() {
                *v /= sum;
            }
        }

        Ok(centrality)
    }

    fn eigenvector_centrality_by_types(
        &self,
        edge_types: &[&str],
        max_iter: usize,
        tolerance: f64,
    ) -> Result<HashMap<usize, f64>> {
        let nodes: Vec<usize> = (0..self.nodes.len())
            .filter(|&id| self.nodes.contains(id))
            .collect();
        let num_nodes = nodes.len();
        let mut centrality: HashMap<usize, f64> = nodes.iter().map(|&n| (n, 1.0)).collect();

        if num_nodes == 0 {
            return Ok(centrality);
        }

        for _ in 0..max_iter {
            let mut new_centrality = HashMap::new();
            let mut max_diff: f64 = 1.0;

            for &v in &nodes {
                let mut score = 1.0;
                if let Some(node) = self.nodes.get(v) {
                    for &(w, ref edges) in &node.neighbors {
                        // Filter edges by types
                        let total_weight: f64 = edges
                            .iter()
                            .filter(|&&(edge_id, _)| {
                                self.edges
                                    .get(edge_id)
                                    .map(|edge| {
                                        edge_types.contains(&edge.data.as_string().as_str())
                                    })
                                    .ok_or(CentralityError::InvalidEdgeReference(edge_id))
                                    .unwrap_or(false)
                            })
                            .map(|&(_, weight)| weight.into())
                            .sum();
                        score += total_weight * centrality[&w];
                    }
                }
                new_centrality.insert(v, score);
                max_diff = max_diff.max((score - centrality[&v]).abs());
            }

            let norm: f64 = new_centrality.values().map(|&v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for v in new_centrality.values_mut() {
                    *v /= norm;
                }
            }

            centrality = new_centrality;

            if max_diff < tolerance {
                break;
            }
        }

        let sum: f64 = centrality.values().sum();
        if sum > 0.0 {
            for v in centrality.values_mut() {
                *v /= sum;
            }
        }

        Ok(centrality)
    }

    fn pagerank(
        &self,
        damping: f64,
        max_iter: usize,
        tolerance: f64,
    ) -> Result<HashMap<usize, f64>> {
        let nodes: Vec<usize> = (0..self.nodes.len())
            .filter(|&id| self.nodes.contains(id))
            .collect();
        let num_nodes = nodes.len();
        let mut pagerank: HashMap<usize, f64> =
            nodes.iter().map(|&n| (n, 1.0 / num_nodes as f64)).collect();

        if num_nodes == 0 {
            return Ok(pagerank);
        }

        let mut out_weights: HashMap<usize, f64> = HashMap::new();
        for &v in &nodes {
            if let Some(node) = self.nodes.get(v) {
                let total_weight: f64 = node
                    .neighbors
                    .iter()
                    .map(|(_, edges)| edges.iter().map(|&(_, w)| w.into()).sum::<f64>())
                    .sum();
                out_weights.insert(v, total_weight);
            } else {
                out_weights.insert(v, 0.0);
            }
        }

        for _ in 0..max_iter {
            let mut new_pagerank = HashMap::new();
            let mut diff = 0.0;

            let teleport = (1.0 - damping) / num_nodes as f64;

            for &v in &nodes {
                let mut inbound = 0.0;
                for &u in &nodes {
                    if let Some(node) = self.nodes.get(u) {
                        for &(w, ref edges) in &node.neighbors {
                            if w == v {
                                let total_weight: f64 =
                                    edges.iter().map(|&(_, weight)| weight.into()).sum();
                                let out_weight = out_weights[&u];
                                if out_weight > 0.0 {
                                    inbound += (total_weight / out_weight) * pagerank[&u];
                                }
                            }
                        }
                    }
                }
                let new_val = teleport + damping * inbound;
                new_pagerank.insert(v, new_val);
                diff += (new_val - pagerank[&v]).abs();
            }

            pagerank = new_pagerank;
            if diff < tolerance {
                break;
            }
        }

        Ok(pagerank)
    }

    fn pagerank_by_types(
        &self,
        edge_types: &[&str],
        damping: f64,
        max_iter: usize,
        tolerance: f64,
    ) -> Result<HashMap<usize, f64>> {
        let nodes: Vec<usize> = (0..self.nodes.len())
            .filter(|&id| self.nodes.contains(id))
            .collect();
        let num_nodes = nodes.len();
        let mut pagerank: HashMap<usize, f64> =
            nodes.iter().map(|&n| (n, 1.0 / num_nodes as f64)).collect();

        if num_nodes == 0 {
            return Ok(pagerank);
        }

        let mut out_weights: HashMap<usize, f64> = HashMap::new();
        for &v in &nodes {
            if let Some(node) = self.nodes.get(v) {
                let total_weight: f64 = node
                    .neighbors
                    .iter()
                    .map(|(_, edges)| {
                        edges
                            .iter()
                            .filter(|&&(edge_id, _)| {
                                self.edges
                                    .get(edge_id)
                                    .map(|edge| {
                                        edge_types.contains(&edge.data.as_string().as_str())
                                    })
                                    .ok_or(CentralityError::InvalidEdgeReference(edge_id))
                                    .unwrap_or(false)
                            })
                            .map(|&(_, w)| w.into())
                            .sum::<f64>()
                    })
                    .sum();
                out_weights.insert(v, total_weight);
            } else {
                out_weights.insert(v, 0.0);
            }
        }

        for _ in 0..max_iter {
            let mut new_pagerank = HashMap::new();
            let mut diff = 0.0;

            let teleport = (1.0 - damping) / num_nodes as f64;

            for &v in &nodes {
                let mut inbound = 0.0;
                for &u in &nodes {
                    if let Some(node) = self.nodes.get(u) {
                        for &(w, ref edges) in &node.neighbors {
                            if w == v {
                                let total_weight: f64 = edges
                                    .iter()
                                    .filter(|&&(edge_id, _)| {
                                        self.edges
                                            .get(edge_id)
                                            .map(|edge| {
                                                edge_types.contains(&edge.data.as_string().as_str())
                                            })
                                            .ok_or(CentralityError::InvalidEdgeReference(edge_id))
                                            .unwrap_or(false)
                                    })
                                    .map(|&(_, weight)| weight.into())
                                    .sum();
                                let out_weight = out_weights[&u];
                                if out_weight > 0.0 {
                                    inbound += (total_weight / out_weight) * pagerank[&u];
                                }
                            }
                        }
                    }
                }
                let new_val = teleport + damping * inbound;
                new_pagerank.insert(v, new_val);
                diff += (new_val - pagerank[&v]).abs();
            }

            pagerank = new_pagerank;
            if diff < tolerance {
                break;
            }
        }

        Ok(pagerank)
    }

    fn katz_centrality(
        &self,
        alpha: f64,
        beta: f64,
        max_iter: usize,
        tolerance: f64,
    ) -> Result<HashMap<usize, f64>> {
        let nodes: Vec<usize> = (0..self.nodes.len())
            .filter(|&id| self.nodes.contains(id))
            .collect();
        let num_nodes = nodes.len();
        let mut centrality: HashMap<usize, f64> = nodes.iter().map(|&n| (n, beta)).collect();

        if num_nodes == 0 {
            return Ok(centrality);
        }

        for _ in 0..max_iter {
            let mut new_centrality = HashMap::new();
            let mut max_diff: f64 = 0.0;

            for &v in &nodes {
                let mut score = 0.0;
                if let Some(node) = self.nodes.get(v) {
                    for &(w, ref edges) in &node.neighbors {
                        let total_weight: f64 =
                            edges.iter().map(|&(_, weight)| weight.into()).sum();
                        score += alpha * total_weight * centrality[&w];
                    }
                }
                let new_val = score + beta;
                new_centrality.insert(v, new_val);
                max_diff = max_diff.max((new_val - centrality[&v]).abs());
            }

            centrality = new_centrality;
            if max_diff < tolerance {
                break;
            }
        }

        Ok(centrality)
    }

    fn katz_centrality_by_types(
        &self,
        edge_types: &[&str],
        alpha: f64,
        beta: f64,
        max_iter: usize,
        tolerance: f64,
    ) -> Result<HashMap<usize, f64>> {
        let nodes: Vec<usize> = (0..self.nodes.len())
            .filter(|&id| self.nodes.contains(id))
            .collect();
        let num_nodes = nodes.len();
        let mut centrality: HashMap<usize, f64> = nodes.iter().map(|&n| (n, beta)).collect();

        if num_nodes == 0 {
            return Ok(centrality);
        }

        for _ in 0..max_iter {
            let mut new_centrality = HashMap::new();
            let mut max_diff: f64 = 0.0;

            for &v in &nodes {
                let mut score = 0.0;
                if let Some(node) = self.nodes.get(v) {
                    for &(w, ref edges) in &node.neighbors {
                        // Filter edges by types
                        let total_weight: f64 = edges
                            .iter()
                            .filter(|&&(edge_id, _)| {
                                self.edges
                                    .get(edge_id)
                                    .map(|edge| {
                                        edge_types.contains(&edge.data.as_string().as_str())
                                    })
                                    .ok_or(CentralityError::InvalidEdgeReference(edge_id))
                                    .unwrap_or(false)
                            })
                            .map(|&(_, weight)| weight.into())
                            .sum();
                        score += alpha * total_weight * centrality[&w];
                    }
                }
                let new_val = score + beta;
                new_centrality.insert(v, new_val);
                max_diff = max_diff.max((new_val - centrality[&v]).abs());
            }

            centrality = new_centrality;
            if max_diff < tolerance {
                break;
            }
        }

        Ok(centrality)
    }
}

#[cfg(test)]
#[cfg(feature = "hgraph")]
mod tests {
    use super::*;
    use crate::hgraph::h_edge::EdgeType;
    use crate::hgraph::h_node::NodeType;
    use float_cmp::approx_eq;

    const TOLERANCE: f64 = 1e-5;

    fn assert_approx_eq(a: f64, b: f64) {
        assert!(approx_eq!(f64, a, b, epsilon = TOLERANCE), "{} ≈ {}", a, b);
    }

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
    fn test_degree_centrality() {
        let mut graph = HeterogeneousGraph::<u32, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 1, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n0, n1, 2, TestEdge("path".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1, TestEdge("bridge".to_string()))
            .unwrap();

        let centrality = graph.degree_centrality().unwrap();
        assert_eq!(centrality[&n0], 2);
        assert_eq!(centrality[&n1], 3);
        assert_eq!(centrality[&n2], 1);
    }

    #[test]
    fn test_degree_centrality_by_types() {
        let mut graph = HeterogeneousGraph::<u32, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 1, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n0, n1, 2, TestEdge("path".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1, TestEdge("bridge".to_string()))
            .unwrap();

        let centrality = graph
            .degree_centrality_by_types(&["road", "bridge"])
            .unwrap();
        assert_eq!(centrality[&n0], 1); // Only "road" edge
        assert_eq!(centrality[&n1], 2); // "road" and "bridge" edges
        assert_eq!(centrality[&n2], 1); // "bridge" edge
    }

    #[test]
    fn test_betweenness_centrality() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));
        let n3 = graph.add_node(TestNode("D".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1.0, TestEdge("bridge".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n3, 1.0, TestEdge("path".to_string()))
            .unwrap();

        let betweenness = graph.betweenness_centrality().unwrap();
        assert_approx_eq(betweenness[&n1], 1.0);
        assert_approx_eq(betweenness[&n0], 0.0);
    }

    #[test]
    fn test_betweenness_centrality_by_types() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));
        let n3 = graph.add_node(TestNode("D".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1.0, TestEdge("bridge".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n3, 1.0, TestEdge("path".to_string()))
            .unwrap();

        let betweenness = graph
            .betweenness_centrality_by_types(&["road", "bridge"])
            .unwrap();
        assert_approx_eq(betweenness[&n1], 1.0);
        assert_approx_eq(betweenness[&n0], 0.0);
    }

    #[test]
    fn test_closeness_centrality() {
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

        let closeness = graph.closeness_centrality().unwrap();
        assert_approx_eq(closeness[&n1], 1.0);
        assert_approx_eq(closeness[&n0], 2.0 / 3.0);
    }

    #[test]
    fn test_closeness_centrality_by_types() {
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

        let closeness = graph
            .closeness_centrality_by_types(&["road", "bridge"])
            .unwrap();
        assert_approx_eq(closeness[&n1], 1.0);
        assert_approx_eq(closeness[&n0], 2.0 / 3.0);
    }

    #[test]
    fn test_eigenvector_centrality() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 1.3, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1.3, TestEdge("bridge".to_string()))
            .unwrap();

        let centrality = graph.eigenvector_centrality(1000, 1e-8).unwrap();
        assert!(centrality[&n1] > centrality[&n0]);
        assert!(centrality[&n1] > centrality[&n2]);
    }

    #[test]
    fn test_eigenvector_centrality_by_types() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 1.3, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1.3, TestEdge("bridge".to_string()))
            .unwrap();

        let centrality = graph
            .eigenvector_centrality_by_types(&["road", "bridge"], 1000, 1e-8)
            .unwrap();
        assert!(centrality[&n1] > centrality[&n0]);
        assert!(centrality[&n1] > centrality[&n2]);
    }

    #[test]
    fn test_pagerank() {
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
            .add_edge(n2, n1, 2.0, TestEdge("path".to_string()))
            .unwrap();

        let pagerank = graph.pagerank(0.85, 100, 1e-6).unwrap();
        assert!(pagerank[&n1] > pagerank[&n0]);
        assert!(pagerank[&n1] > pagerank[&n2]);
    }

    #[test]
    fn test_pagerank_by_types() {
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
            .add_edge(n2, n1, 2.0, TestEdge("path".to_string()))
            .unwrap();

        let pagerank = graph
            .pagerank_by_types(&["road", "bridge"], 0.85, 100, 1e-6)
            .unwrap();
        assert!(pagerank[&n1] > pagerank[&n0]);
        assert!(pagerank[&n1] > pagerank[&n2]);
    }

    #[test]
    fn test_katz_centrality() {
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

        let katz = graph.katz_centrality(0.1, 1.0, 100, 1e-6).unwrap();
        assert!(katz[&n1] > katz[&n0]);
        assert!(katz[&n1] > katz[&n2]);
    }

    #[test]
    fn test_katz_centrality_by_types() {
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

        let katz = graph
            .katz_centrality_by_types(&["road", "bridge"], 0.1, 1.0, 100, 1e-6)
            .unwrap();
        assert!(katz[&n1] > katz[&n0]);
        assert!(katz[&n1] > katz[&n2]);
    }

    #[test]
    fn test_empty_graph() {
        let graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        assert!(graph.degree_centrality().unwrap().is_empty());
        assert!(graph.betweenness_centrality().unwrap().is_empty());
        assert!(graph.closeness_centrality().unwrap().is_empty());
        assert!(graph.eigenvector_centrality(100, 1e-6).unwrap().is_empty());
        assert!(graph.pagerank(0.85, 100, 1e-6).unwrap().is_empty());
        assert!(graph
            .katz_centrality(0.1, 1.0, 100, 1e-6)
            .unwrap()
            .is_empty());
    }
}
