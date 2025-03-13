//! Community detection using the Leiden algorithm
//!
//! This module implements the Leiden algorithm for community detection in graphs,
//! an improvement over the Louvain method that guarantees well-connected communities.
//! It provides flexible configuration options and handles both directed and undirected graphs.
//!
//! # Features
//! - Detects communities optimizing modularity
//! - Supports deterministic and randomized execution
//! - Handles graph aggregation and refinement
//! - Ensures community connectivity
//!
//! # Examples
//!
//! Basic usage:
//! ```rust
//! use xgraph::graph::graph::Graph;
//! use xgraph::graph::algorithms::leiden_clustering::{CommunityDetection, CommunityConfig};
//!
//! let mut graph = Graph::<f64, (), ()>::new(false);
//! let n0 = graph.add_node(()); let n1 = graph.add_node(()); let n2 = graph.add_node(());
//! graph.add_edge(n0, n1, 1.0, ()).unwrap();
//! graph.add_edge(n1, n2, 1.0, ()).unwrap();
//! graph.add_edge(n0, n2, 1.0, ()).unwrap();
//!
//! let communities = graph.detect_communities(1.0).unwrap();
//! println!("Detected {} communities", communities.len());
//! ```

use crate::graph::graph::Graph;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

/// Error type for community detection failures.
#[derive(Debug)]
pub enum CommunityDetectionError {
    /// Indicates an invalid node was encountered during computation.
    InvalidNode(usize),
    /// Indicates an invalid floating-point result (NaN or infinity) occurred during computation.
    InvalidFloatResult(String),
}

impl std::fmt::Display for CommunityDetectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CommunityDetectionError::InvalidNode(id) => {
                write!(f, "Invalid node reference: node ID {} not found", id)
            }
            CommunityDetectionError::InvalidFloatResult(msg) => {
                write!(
                    f,
                    "Invalid floating-point result in community detection: {}",
                    msg
                )
            }
        }
    }
}

impl std::error::Error for CommunityDetectionError {}

/// Trait for detecting communities in a graph using the Leiden algorithm.
///
/// Implementors can identify clusters based on edge weights and node relationships,
/// optimizing for modularity with error handling.
///
/// # Type Parameters
/// - `W`: Edge weight type, must be convertible to `f64` and support basic operations.
/// - `N`: Node data type, must be clonable, equatable, hashable, and debuggable.
/// - `E`: Edge data type, must be clonable and debuggable.
///
/// # Examples
/// ```rust
/// use xgraph::graph::graph::Graph;
/// use xgraph::graph::algorithms::leiden_clustering::CommunityDetection;
///
/// let mut graph = Graph::<f64, (), ()>::new(false);
/// let n0 = graph.add_node(()); let n1 = graph.add_node(());
/// graph.add_edge(n0, n1, 1.0, ()).unwrap();
///
/// let communities = graph.detect_communities(1.0).unwrap();
/// ```
pub trait CommunityDetection<W, N, E>
where
    W: Copy + Into<f64> + Default + PartialEq,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    fn detect_communities(
        &self,
        gamma: f64,
    ) -> Result<HashMap<usize, Vec<usize>>, CommunityDetectionError>;
    fn detect_communities_with_config(
        &self,
        config: CommunityConfig,
    ) -> Result<HashMap<usize, Vec<usize>>, CommunityDetectionError>;
}

/// Configuration for the Leiden community detection algorithm.
///
/// Allows customization of algorithm behavior including resolution, iterations, and randomness.
///
/// # Fields
/// - `gamma`: Resolution parameter controlling community granularity
/// - `resolution`: Scaling factor for modularity
/// - `iterations`: Maximum number of iterations
/// - `deterministic`: Whether to use deterministic behavior
/// - `seed`: Optional seed for random number generator
#[derive(Debug, Clone)]
pub struct CommunityConfig {
    pub gamma: f64,
    pub resolution: f64,
    pub iterations: usize,
    pub deterministic: bool,
    pub seed: Option<u64>,
}

impl Default for CommunityConfig {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            resolution: 1.0,
            iterations: 10,
            deterministic: false,
            seed: None,
        }
    }
}

/// Internal state for the Leiden clustering algorithm.
///
/// Manages graph structure, community assignments, and weights during clustering.
///
/// # Fields
/// - `adjacency_list`: Graph representation as adjacency list
/// - `node_to_community`: Current community assignments
/// - `community_weights`: Total edge weights per community
/// - `total_weight`: Sum of all edge weights
/// - `hierarchy`: Tracks community assignments across levels
/// - `original_size`: Original number of nodes
/// - `config`: Algorithm configuration
/// - `rng`: Random number generator
struct LeidenState {
    adjacency_list: Vec<Vec<(usize, f64)>>,
    node_to_community: HashMap<usize, usize>,
    community_weights: HashMap<usize, f64>,
    total_weight: f64,
    hierarchy: Vec<HashMap<usize, usize>>,
    original_size: usize,
    config: CommunityConfig,
    rng: StdRng,
}

impl<W, N, E> CommunityDetection<W, N, E> for Graph<W, N, E>
where
    W: Copy + Into<f64> + Default + PartialEq,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    fn detect_communities(
        &self,
        gamma: f64,
    ) -> Result<HashMap<usize, Vec<usize>>, CommunityDetectionError> {
        self.detect_communities_with_config(CommunityConfig {
            gamma,
            ..CommunityConfig::default()
        })
    }

    fn detect_communities_with_config(
        &self,
        config: CommunityConfig,
    ) -> Result<HashMap<usize, Vec<usize>>, CommunityDetectionError> {
        let adjacency_list: Vec<Vec<(usize, f64)>> = self
            .nodes
            .iter()
            .map(|(node, _)| {
                self.get_neighbors(node)
                    .into_iter()
                    .map(|(n, w)| (n, w.into()))
                    .collect()
            })
            .collect();

        let mut state = LeidenState::initialize(&adjacency_list, config)?;
        let mut prev_communities = state.node_to_community.clone();

        for _ in 0..state.config.iterations {
            state.fast_louvain_move_nodes()?;
            state.refine_partition()?;
            state.enforce_gamma_properties()?;

            if state.node_to_community == prev_communities {
                break;
            }
            prev_communities = state.node_to_community.clone();

            if state.should_aggregate() {
                state.aggregate_graph()?;
            }
        }

        state.merge_small_communities(2)?;
        state.resolve_hierarchy()?;
        Ok(state.get_communities())
    }
}

#[allow(dead_code)]
impl LeidenState {
    fn initialize(
        adjacency_list: &[Vec<(usize, f64)>],
        config: CommunityConfig,
    ) -> Result<Self, CommunityDetectionError> {
        let mut node_to_community = HashMap::new();
        let mut community_weights = HashMap::new();
        let mut total_weight = 0.0;

        for (node, neighbors) in adjacency_list.iter().enumerate() {
            node_to_community.insert(node, node);
            let weight = neighbors.iter().map(|(_, w)| w).sum::<f64>();
            community_weights.insert(node, weight);
            total_weight += weight;
        }

        let initial_p = node_to_community.clone();
        let rng = if config.deterministic {
            StdRng::seed_from_u64(config.seed.unwrap_or(42))
        } else {
            StdRng::seed_from_u64(config.seed.unwrap_or_else(rand::random))
        };

        Ok(Self {
            adjacency_list: adjacency_list.to_vec(),
            node_to_community,
            community_weights,
            total_weight,
            hierarchy: vec![initial_p],
            original_size: adjacency_list.len(),
            config,
            rng,
        })
    }

    fn fast_louvain_move_nodes(&mut self) -> Result<(), CommunityDetectionError> {
        let mut nodes: Vec<usize> = (0..self.adjacency_list.len()).collect();
        if self.config.deterministic {
            nodes.sort_unstable();
        } else {
            nodes.shuffle(&mut self.rng);
        }

        let mut queue: VecDeque<usize> = nodes.into_iter().collect();
        let mut processed = HashSet::new();

        while let Some(node) = queue.pop_front() {
            if processed.contains(&node) {
                continue;
            }
            processed.insert(node);

            let current_comm = *self
                .node_to_community
                .get(&node)
                .ok_or(CommunityDetectionError::InvalidNode(node))?;
            let mut eligible_communities = self.find_eligible_communities(node, current_comm);
            if self.config.deterministic {
                eligible_communities.sort_unstable();
            }

            let mut best_comm = current_comm;
            let mut best_delta = f64::MIN;

            for comm in &eligible_communities {
                let delta = self.calculate_move_delta(node, *comm)?;
                if delta > best_delta {
                    best_delta = delta;
                    best_comm = *comm;
                }
            }

            if best_delta > 0.0 && best_comm != current_comm {
                self.move_node(node, best_comm)?;

                let mut neighbors: Vec<usize> = self.adjacency_list[node]
                    .iter()
                    .map(|&(n, _)| n)
                    .filter(|n| !processed.contains(n))
                    .collect();
                if self.config.deterministic {
                    neighbors.sort_unstable();
                }
                for neighbor in neighbors {
                    queue.push_back(neighbor);
                }
            }

            if queue.len() > self.original_size * 2 {
                break;
            }
        }
        Ok(())
    }

    fn generate_order_key(&mut self, node: usize) -> u64 {
        let mut hasher = DefaultHasher::new();
        hasher.write_u64(self.rng.random::<u64>());
        hasher.write_usize(node);
        hasher.finish()
    }

    fn refine_partition(&mut self) -> Result<(), CommunityDetectionError> {
        let mut singleton_partition = self.create_singleton_partition();

        let mut communities: Vec<usize> = self
            .node_to_community
            .values()
            .copied()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        if self.config.deterministic {
            communities.sort_unstable();
        }

        for comm in communities {
            self.refine_community_subset(comm, &mut singleton_partition)?;
        }

        self.node_to_community = singleton_partition;
        self.hierarchy.push(self.node_to_community.clone());
        Ok(())
    }

    fn resolve_hierarchy(&mut self) -> Result<(), CommunityDetectionError> {
        let mut final_communities = HashMap::new();
        for node in 0..self.original_size {
            let mut current = node;
            for level in &self.hierarchy[1..] {
                current = level.get(&current).copied().unwrap_or(current);
            }
            final_communities.insert(node, current);
        }
        self.node_to_community = final_communities;
        Ok(())
    }

    fn should_aggregate(&self) -> bool {
        let current_modularity = self.calculate_modularity().unwrap_or(0.0);
        let communities = self.get_communities();
        communities.values().any(|nodes| nodes.len() > 1) && current_modularity > 0.0
    }

    pub fn get_communities(&self) -> HashMap<usize, Vec<usize>> {
        let mut final_mapping = HashMap::new();
        for node in 0..self.original_size {
            let mut current = node;
            for level in &self.hierarchy[1..] {
                current = level.get(&current).copied().unwrap_or(current);
            }
            final_mapping
                .entry(current)
                .or_insert_with(Vec::new)
                .push(node);
        }
        for nodes in final_mapping.values_mut() {
            nodes.sort_unstable();
        }
        final_mapping
    }

    fn find_eligible_communities(&self, node: usize, original_comm: usize) -> Vec<usize> {
        let mut comms: Vec<usize> = self.adjacency_list[node]
            .iter()
            .map(|(n, _)| self.node_to_community.get(n).copied().unwrap_or(*n))
            .filter(|&comm| comm != original_comm)
            .collect();
        comms.sort_unstable();
        comms.dedup();
        comms
    }

    fn calculate_move_delta(
        &self,
        node: usize,
        new_community: usize,
    ) -> Result<f64, CommunityDetectionError> {
        let old_community = self.node_to_community.get(&node).copied().unwrap_or(node);
        if old_community == new_community {
            return Ok(0.0);
        }

        let k_i = self.community_weights.get(&node).copied().unwrap_or(0.0);
        let sum_tot_new = self
            .community_weights
            .get(&new_community)
            .copied()
            .unwrap_or(0.0);
        let sum_tot_old = self
            .community_weights
            .get(&old_community)
            .copied()
            .unwrap_or(0.0);
        let m = self.total_weight.max(f64::EPSILON);

        let sum_in_new: f64 = self.adjacency_list[node]
            .iter()
            .filter(|(n, _)| self.node_to_community.get(n).copied().unwrap_or(*n) == new_community)
            .map(|(_, w)| w)
            .sum();
        let sum_in_old: f64 = self.adjacency_list[node]
            .iter()
            .filter(|(n, _)| self.node_to_community.get(n).copied().unwrap_or(*n) == old_community)
            .map(|(_, w)| w)
            .sum();

        let delta_new = sum_in_new - self.config.resolution * (sum_tot_new + k_i) * k_i / (2.0 * m);
        let delta_old = sum_in_old - self.config.resolution * (sum_tot_old) * k_i / (2.0 * m);

        let result = delta_new - delta_old;
        if result.is_nan() || result.is_infinite() {
            return Err(CommunityDetectionError::InvalidFloatResult(format!(
                "Delta calculation resulted in {} for node {}",
                if result.is_nan() { "NaN" } else { "infinity" },
                node
            )));
        }
        Ok(result)
    }

    fn move_node(
        &mut self,
        node: usize,
        new_community: usize,
    ) -> Result<(), CommunityDetectionError> {
        let old_community = self.node_to_community.get(&node).copied().unwrap_or(node);
        let node_weight = self.community_weights.get(&node).copied().unwrap_or(0.0);

        let old_weight = self
            .community_weights
            .get_mut(&old_community)
            .ok_or(CommunityDetectionError::InvalidNode(old_community))?;
        *old_weight -= node_weight;
        *self.community_weights.entry(new_community).or_insert(0.0) += node_weight;

        self.node_to_community.insert(node, new_community);

        if let Some(current_level) = self.hierarchy.last_mut() {
            current_level.insert(node, new_community);
        }
        Ok(())
    }

    fn create_singleton_partition(&self) -> HashMap<usize, usize> {
        (0..self.adjacency_list.len()).map(|n| (n, n)).collect()
    }

    fn aggregate_graph(&mut self) -> Result<(), CommunityDetectionError> {
        let mut communities: Vec<(usize, Vec<usize>)> =
            self.get_communities().into_iter().collect();
        if self.config.deterministic {
            communities.sort_by_key(|(_, nodes)| {
                let mut sorted_nodes = nodes.clone();
                sorted_nodes.sort_unstable();
                sorted_nodes
            });
        }

        let mut new_adjacency_list = vec![vec![]; communities.len()];
        let mut index_mapping = HashMap::new();

        for (new_id, (_, nodes)) in communities.iter().enumerate() {
            for &node in nodes {
                index_mapping.insert(node, new_id);
            }
        }

        for (new_id, (_, nodes)) in communities.iter().enumerate() {
            let mut aggregated_edges = HashMap::new();
            for &node in nodes {
                if node < self.adjacency_list.len() {
                    for (neighbor, weight) in &self.adjacency_list[node] {
                        let target_comm = *index_mapping.get(neighbor).unwrap();
                        *aggregated_edges.entry(target_comm).or_insert(0.0) += weight;
                    }
                }
            }
            new_adjacency_list[new_id] = aggregated_edges.into_iter().collect();
        }

        self.adjacency_list = new_adjacency_list;
        self.node_to_community = index_mapping;
        self.community_weights = self.calculate_new_community_weights()?;
        self.total_weight = self.community_weights.values().sum();
        Ok(())
    }

    fn calculate_new_community_weights(
        &self,
    ) -> Result<HashMap<usize, f64>, CommunityDetectionError> {
        Ok(self
            .adjacency_list
            .iter()
            .enumerate()
            .map(|(node, edges)| (node, edges.iter().map(|(_, w)| w).sum()))
            .collect())
    }

    fn calculate_modularity(&self) -> Result<f64, CommunityDetectionError> {
        let mut modularity = 0.0;
        let m = self.total_weight.max(f64::EPSILON);

        for (node, &community) in &self.node_to_community {
            let node_weight = self.community_weights.get(node).copied().unwrap_or(0.0);
            if *node < self.adjacency_list.len() {
                for (neighbor, weight) in &self.adjacency_list[*node] {
                    if community == *self.node_to_community.get(neighbor).unwrap() {
                        let neighbor_weight =
                            self.community_weights.get(neighbor).copied().unwrap_or(0.0);
                        let term = *weight
                            - self.config.resolution
                                * self.config.gamma
                                * (node_weight * neighbor_weight)
                                / (2.0 * m);
                        if term.is_nan() || term.is_infinite() {
                            return Err(CommunityDetectionError::InvalidFloatResult(format!(
                                "Modularity term resulted in {} for node {}",
                                if term.is_nan() { "NaN" } else { "infinity" },
                                node
                            )));
                        }
                        modularity += term;
                    }
                }
            }
        }
        let result = modularity / (2.0 * m);
        if result.is_nan() || result.is_infinite() {
            return Err(CommunityDetectionError::InvalidFloatResult(format!(
                "Modularity resulted in {}",
                if result.is_nan() { "NaN" } else { "infinity" }
            )));
        }
        Ok(result)
    }

    fn get_nodes_in_community(&self, comm: usize) -> Vec<usize> {
        let mut nodes: Vec<usize> = self
            .node_to_community
            .iter()
            .filter(|(_, &c)| c == comm)
            .map(|(&n, _)| n)
            .collect();
        if self.config.deterministic {
            nodes.sort_unstable();
        }
        nodes
    }

    fn enforce_gamma_properties(&mut self) -> Result<(), CommunityDetectionError> {
        let mut communities: Vec<usize> = self
            .node_to_community
            .values()
            .copied()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        if self.config.deterministic {
            communities.sort_unstable();
        }

        for comm in communities {
            if !self.is_community_connected(comm) {
                self.split_disconnected_community(comm)?;
            }
        }
        Ok(())
    }

    fn is_community_connected(&self, community: usize) -> bool {
        let nodes = self.get_nodes_in_community(community);
        if nodes.len() <= 1 {
            return true;
        }

        let mut visited = HashSet::new();
        let mut stack = vec![nodes[0]];

        while let Some(node) = stack.pop() {
            if visited.insert(node) && node < self.adjacency_list.len() {
                for (neighbor, _) in &self.adjacency_list[node] {
                    if nodes.contains(neighbor) && !visited.contains(neighbor) {
                        stack.push(*neighbor);
                    }
                }
            }
        }

        visited.len() == nodes.len()
    }

    fn split_disconnected_community(
        &mut self,
        community: usize,
    ) -> Result<(), CommunityDetectionError> {
        let nodes = self.get_nodes_in_community(community);
        if nodes.is_empty() {
            return Ok(());
        }

        let components = self.find_connected_components(&nodes);
        for (i, component) in components.iter().enumerate() {
            let new_comm = if i == 0 {
                community
            } else {
                self.adjacency_list.len() + i
            };
            for &node in component {
                self.move_node(node, new_comm)?;
            }
        }
        Ok(())
    }

    fn find_connected_components(&self, nodes: &[usize]) -> Vec<Vec<usize>> {
        let node_set: HashSet<_> = nodes.iter().copied().collect();
        let mut visited = HashSet::new();
        let mut components = Vec::new();

        for &node in nodes {
            if !visited.contains(&node) {
                let mut component = Vec::new();
                let mut stack = vec![node];

                while let Some(current) = stack.pop() {
                    if visited.insert(current) && current < self.adjacency_list.len() {
                        component.push(current);
                        for (neighbor, _) in &self.adjacency_list[current] {
                            if node_set.contains(neighbor) && !visited.contains(neighbor) {
                                stack.push(*neighbor);
                            }
                        }
                    }
                }
                if self.config.deterministic {
                    component.sort_unstable();
                }
                components.push(component);
            }
        }
        components
    }

    fn merge_small_communities(&mut self, min_size: usize) -> Result<(), CommunityDetectionError> {
        let communities = self.get_communities();
        let mut to_merge: Vec<usize> = communities
            .iter()
            .filter(|(_, nodes)| nodes.len() < min_size)
            .map(|(comm, _)| *comm)
            .collect();
        if self.config.deterministic {
            to_merge.sort_unstable();
        }

        for comm in to_merge {
            let mut best_target = comm;
            let mut max_connections = 0.0;

            for node in &communities[&comm] {
                if *node < self.adjacency_list.len() {
                    for (neighbor, weight) in &self.adjacency_list[*node] {
                        let target = self
                            .node_to_community
                            .get(neighbor)
                            .copied()
                            .unwrap_or(*neighbor);
                        if target != comm
                            && communities.get(&target).map_or(0, |v| v.len()) >= min_size
                        {
                            let count = *weight;
                            if count > max_connections {
                                max_connections = count;
                                best_target = target;
                            }
                        }
                    }
                }
            }

            if best_target != comm {
                for node in &communities[&comm] {
                    self.move_node(*node, best_target)?;
                }
            }
        }
        Ok(())
    }

    fn refine_community_subset(
        &mut self,
        comm: usize,
        partition: &mut HashMap<usize, usize>,
    ) -> Result<(), CommunityDetectionError> {
        let mut nodes: Vec<usize> = self
            .node_to_community
            .iter()
            .filter(|(_, &c)| c == comm)
            .map(|(&n, _)| n)
            .collect();

        let max_passes = 10;
        let mut changed;

        for _ in 0..max_passes {
            changed = false;

            if self.config.deterministic {
                nodes.sort_unstable();
            } else {
                nodes.shuffle(&mut self.rng);
            }

            for &node in &nodes {
                if node >= self.adjacency_list.len() {
                    continue;
                }

                let current_comm = partition.get(&node).copied().unwrap_or(node);
                let mut best_comm = current_comm;
                let mut best_delta = f64::MIN;

                let k_i = self.community_weights.get(&node).copied().unwrap_or(0.0);
                let m = self.total_weight.max(f64::EPSILON);

                let mut neighbor_comms: Vec<usize> = self.adjacency_list[node]
                    .iter()
                    .map(|(n, _)| self.node_to_community.get(n).copied().unwrap_or(*n))
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect();
                if self.config.deterministic {
                    neighbor_comms.sort_unstable();
                }

                for &neighbor_comm in &neighbor_comms {
                    if neighbor_comm != current_comm {
                        let sum_tot_new = self
                            .community_weights
                            .get(&neighbor_comm)
                            .copied()
                            .unwrap_or(0.0);
                        let sum_tot_old = self
                            .community_weights
                            .get(&current_comm)
                            .copied()
                            .unwrap_or(0.0);
                        let sum_in_new: f64 = self.adjacency_list[node]
                            .iter()
                            .filter(|(n, _)| {
                                self.node_to_community.get(n).copied().unwrap_or(*n)
                                    == neighbor_comm
                            })
                            .map(|(_, w)| w)
                            .sum();
                        let sum_in_old: f64 = self.adjacency_list[node]
                            .iter()
                            .filter(|(n, _)| {
                                self.node_to_community.get(n).copied().unwrap_or(*n) == current_comm
                            })
                            .map(|(_, w)| w)
                            .sum();

                        let delta_new = sum_in_new
                            - self.config.resolution * (sum_tot_new + k_i) * k_i / (2.0 * m);
                        let delta_old =
                            sum_in_old - self.config.resolution * sum_tot_old * k_i / (2.0 * m);
                        let delta = delta_new - delta_old;

                        if delta.is_nan() || delta.is_infinite() {
                            return Err(CommunityDetectionError::InvalidFloatResult(format!(
                                "Delta resulted in {} for node {}",
                                if delta.is_nan() { "NaN" } else { "infinity" },
                                node
                            )));
                        }

                        if delta > best_delta {
                            best_delta = delta;
                            best_comm = neighbor_comm;
                        }
                    }
                }

                if self.config.deterministic {
                    if best_delta > 0.0 && best_comm != current_comm {
                        partition.insert(node, best_comm);
                        changed = true;
                    }
                } else if best_delta > 0.0
                    && self.rng.random::<f64>() < best_delta.exp() / (1.0 + best_delta.exp())
                {
                    partition.insert(node, best_comm);
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }
        Ok(())
    }

    fn calculate_community_connections(
        node: usize,
        comm: usize,
        adjacency_list: &[Vec<(usize, f64)>],
        node_to_community: &HashMap<usize, usize>,
    ) -> f64 {
        if node >= adjacency_list.len() {
            return 0.0;
        }
        adjacency_list[node]
            .iter()
            .filter(|(n, _)| node_to_community.get(n).copied().unwrap_or(*n) == comm)
            .map(|(_, w)| w)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> Graph<f64, String, String> {
        let mut graph = Graph::new(false);
        let node0 = graph.add_node("Node0".to_string());
        let node1 = graph.add_node("Node1".to_string());
        let node2 = graph.add_node("Node2".to_string());
        let _node3 = graph.add_node("Node3".to_string());

        graph
            .add_edge(node0, node1, 1.0, "e01".to_string())
            .unwrap();
        graph
            .add_edge(node1, node2, 1.0, "e12".to_string())
            .unwrap();
        graph
            .add_edge(node0, node2, 1.0, "e02".to_string())
            .unwrap();

        graph
    }

    #[test]
    fn test_community_detection_basic() {
        let graph = create_test_graph();
        let communities = graph.detect_communities(1.0).unwrap();
        assert_eq!(communities.len(), 2);
        let mut found_clique = false;
        let mut found_isolated = false;
        for (_, nodes) in communities {
            if nodes.len() == 3 && nodes.contains(&0) && nodes.contains(&1) && nodes.contains(&2) {
                found_clique = true;
            }
            if nodes.len() == 1 && nodes.contains(&3) {
                found_isolated = true;
            }
        }
        assert!(found_clique);
        assert!(found_isolated);
    }

    #[test]
    fn test_deterministic_behavior() {
        let graph = create_test_graph();
        let config = CommunityConfig {
            deterministic: true,
            seed: Some(42),
            ..CommunityConfig::default()
        };
        let communities1 = graph
            .detect_communities_with_config(config.clone())
            .unwrap();
        let communities2 = graph.detect_communities_with_config(config).unwrap();
        assert_eq!(communities1, communities2);
    }

    #[test]
    fn test_empty_graph() {
        let graph: Graph<f64, String, String> = Graph::new(false);
        let communities = graph.detect_communities(1.0).unwrap();
        assert!(communities.is_empty());
    }

    #[test]
    fn test_single_node() {
        let mut graph: Graph<f64, String, String> = Graph::new(false);
        let node = graph.add_node("Single".to_string());
        let communities = graph.detect_communities(1.0).unwrap();
        assert_eq!(communities.len(), 1);
        assert_eq!(communities.get(&node), Some(&vec![node]));
    }

    #[test]
    fn test_modularity_calculation() {
        let graph = create_test_graph();
        let adjacency_list: Vec<Vec<(usize, f64)>> = graph
            .nodes
            .iter()
            .map(|(node, _)| {
                graph
                    .get_neighbors(node)
                    .into_iter()
                    .map(|(n, w)| (n, w.into()))
                    .collect()
            })
            .collect();
        let state = LeidenState::initialize(&adjacency_list, CommunityConfig::default()).unwrap();
        let modularity = state.calculate_modularity().unwrap();
        assert!(modularity >= 0.0);
    }

    #[test]
    fn test_directed_graph() {
        let mut graph = Graph::new(true);
        let node0 = graph.add_node("Node0".to_string());
        let node1 = graph.add_node("Node1".to_string());
        let node2 = graph.add_node("Node2".to_string());

        graph
            .add_edge(node0, node1, 1.0, "e01".to_string())
            .unwrap();
        graph
            .add_edge(node1, node2, 1.0, "e12".to_string())
            .unwrap();
        graph
            .add_edge(node2, node0, 1.0, "e20".to_string())
            .unwrap();

        let communities = graph.detect_communities(1.0).unwrap();
        assert_eq!(communities.len(), 1);
        let nodes = communities.values().next().unwrap();
        assert_eq!(nodes.len(), 3);
        assert!(nodes.contains(&node0) && nodes.contains(&node1) && nodes.contains(&node2));
    }
}
