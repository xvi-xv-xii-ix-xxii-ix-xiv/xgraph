//! Module for detecting communities in heterogeneous multigraphs using the Leiden algorithm.
//!
//! This module implements the Leiden algorithm tailored for heterogeneous multigraphs, allowing
//! community detection across all edges or filtered by specific edge types. The algorithm optimizes
//! modularity to identify densely connected groups of nodes, supporting both deterministic and
//! non-deterministic runs with configurable parameters.

#[cfg(feature = "hgraph")]
use crate::hgraph::h_graph::HeterogeneousGraph;
#[cfg(feature = "hgraph")]
use rand::rngs::StdRng;
#[cfg(feature = "hgraph")]
use rand::seq::SliceRandom;
#[cfg(feature = "hgraph")]
use rand::{Rng, SeedableRng};
#[cfg(feature = "hgraph")]
use std::collections::hash_map::DefaultHasher;
#[cfg(feature = "hgraph")]
use std::collections::{HashMap, HashSet, VecDeque};
#[cfg(feature = "hgraph")]
use std::fmt::Debug;
#[cfg(feature = "hgraph")]
use std::hash::{Hash, Hasher};

// Error handling additions

/// Error type for community detection failures.
///
/// Represents errors that may occur during the Leiden algorithm execution, such as referencing
/// invalid nodes or encountering computational issues.
#[derive(Debug)]
pub enum CommunityDetectionError {
    /// Indicates a node referenced in the graph does not exist.
    InvalidNodeReference(usize),
    /// Indicates an arithmetic overflow or invalid computation.
    ComputationError(String),
}

impl std::fmt::Display for CommunityDetectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CommunityDetectionError::InvalidNodeReference(id) => {
                write!(f, "Invalid node reference: node ID {} not found", id)
            }
            CommunityDetectionError::ComputationError(msg) => {
                write!(f, "Computation error: {}", msg)
            }
        }
    }
}

impl std::error::Error for CommunityDetectionError {}

/// Result type alias for community detection operations.
///
/// Wraps the result of community detection methods, enabling error handling without panicking.
pub type Result<T> = std::result::Result<T, CommunityDetectionError>;

// Enhanced trait with documentation and error handling

/// Trait defining community detection methods for heterogeneous graphs.
///
/// Provides methods to detect communities using the Leiden algorithm, with options for default
/// parameters, custom configurations, and edge-type filtering. All methods return a `Result` to
/// handle potential errors gracefully instead of panicking.
#[cfg(feature = "hgraph")]
pub trait HeteroCommunityDetection<W, N, E>
where
    W: Copy + Into<f64> + Default + PartialEq + Debug,
    N: Clone + Eq + Hash + Debug + crate::hgraph::h_node::NodeType,
    E: Clone + Default + Debug + crate::hgraph::h_edge::EdgeType + PartialEq,
{
    /// Detects communities using the Leiden algorithm with a specified gamma parameter.
    ///
    /// Uses default configuration except for the provided `gamma`, which controls the preference
    /// for smaller or larger communities.
    ///
    /// # Arguments
    /// * `gamma` - The gamma parameter affecting community size (higher values favor smaller communities).
    ///
    /// # Returns
    /// A `Result` containing a mapping of community IDs to vectors of node IDs.
    fn detect_communities(&self, gamma: f64) -> Result<HashMap<usize, Vec<usize>>>;

    /// Detects communities using the Leiden algorithm with a custom configuration.
    ///
    /// Allows fine-tuned control over the algorithm's behavior via `CommunityConfig`.
    ///
    /// # Arguments
    /// * `config` - Configuration parameters for the Leiden algorithm.
    ///
    /// # Returns
    /// A `Result` containing a mapping of community IDs to vectors of node IDs.
    fn detect_communities_with_config(
        &self,
        config: CommunityConfig,
    ) -> Result<HashMap<usize, Vec<usize>>>;

    /// Detects communities using the Leiden algorithm, considering only specified edge types.
    ///
    /// Filters the graph to include only edges of the given types before running the algorithm.
    ///
    /// # Arguments
    /// * `allowed_edge_types` - A slice of edge types to filter edges by. If empty, all edges are considered.
    /// * `config` - Configuration parameters for the Leiden algorithm.
    ///
    /// # Returns
    /// A `Result` containing a mapping of community IDs to vectors of node IDs.
    fn detect_communities_by_types(
        &self,
        allowed_edge_types: &[E],
        config: CommunityConfig,
    ) -> Result<HashMap<usize, Vec<usize>>>;
}

/// Configuration structure for the Leiden community detection algorithm.
///
/// Defines parameters controlling the algorithm's behavior, such as resolution, iteration count,
/// and determinism.
#[cfg(feature = "hgraph")]
#[derive(Debug, Clone)]
pub struct CommunityConfig {
    /// Gamma parameter affecting community size preference (higher values favor smaller communities).
    pub gamma: f64,
    /// Resolution parameter controlling the granularity of communities.
    pub resolution: f64,
    /// Maximum number of iterations for the algorithm.
    pub iterations: usize,
    /// Whether to enforce deterministic behavior (true) or allow randomness (false).
    pub deterministic: bool,
    /// Optional seed for random number generation; used if `deterministic` is true.
    pub seed: Option<u64>,
    /// Minimum size for communities; smaller ones may be merged.
    pub min_community_size: usize,
}

#[cfg(feature = "hgraph")]
impl Default for CommunityConfig {
    /// Provides default values for `CommunityConfig`.
    fn default() -> Self {
        Self {
            gamma: 1.0,
            resolution: 1.0,
            iterations: 10,
            deterministic: false,
            seed: None,
            min_community_size: 2,
        }
    }
}

/// Internal state for the Leiden algorithm.
///
/// Manages the graph's adjacency list, community assignments, and hierarchical structure during
/// community detection.
#[cfg(feature = "hgraph")]
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

#[cfg(feature = "hgraph")]
impl<W, N, E> HeteroCommunityDetection<W, N, E> for HeterogeneousGraph<W, N, E>
where
    W: Copy + Into<f64> + Default + PartialEq + Debug,
    N: Clone + Eq + Hash + Debug + crate::hgraph::h_node::NodeType,
    E: Clone + Default + Debug + crate::hgraph::h_edge::EdgeType + PartialEq,
{
    fn detect_communities(&self, gamma: f64) -> Result<HashMap<usize, Vec<usize>>> {
        self.detect_communities_with_config(CommunityConfig {
            gamma,
            ..CommunityConfig::default()
        })
    }

    fn detect_communities_with_config(
        &self,
        config: CommunityConfig,
    ) -> Result<HashMap<usize, Vec<usize>>> {
        self.detect_communities_by_types(&[], config)
    }

    fn detect_communities_by_types(
        &self,
        allowed_edge_types: &[E],
        config: CommunityConfig,
    ) -> Result<HashMap<usize, Vec<usize>>> {
        // Step 1: Identify nodes in the subgraph defined by allowed_edge_types
        let mut relevant_nodes = HashSet::new();
        let adjacency_list: Vec<Vec<(usize, f64)>> = self
            .nodes
            .iter()
            .map(|(node, _)| {
                let mut neighbor_weights: HashMap<usize, f64> = HashMap::new();
                for (neighbor, edges) in self.get_neighbors(node) {
                    let total_weight: f64 = edges
                        .iter()
                        .filter(|(edge_id, _)| {
                            if let Some(edge) = self.edges.get(*edge_id) {
                                allowed_edge_types.is_empty()
                                    || allowed_edge_types.contains(&edge.data)
                            } else {
                                false
                            }
                        })
                        .map(|(_, w)| Into::<f64>::into(*w))
                        .sum();
                    if total_weight > 0.0 {
                        neighbor_weights.insert(neighbor, total_weight);
                        relevant_nodes.insert(node);
                        relevant_nodes.insert(neighbor);
                    }
                }
                neighbor_weights.into_iter().collect()
            })
            .collect();

        // Step 2: Restrict adjacency_list to relevant nodes
        let mut node_mapping = HashMap::new();
        let mut reverse_mapping = HashMap::new();
        let mut new_adjacency_list = Vec::new();
        for (old_id, neighbors) in adjacency_list.iter().enumerate() {
            if relevant_nodes.contains(&old_id) {
                let new_id = new_adjacency_list.len();
                node_mapping.insert(old_id, new_id);
                reverse_mapping.insert(new_id, old_id);
                let remapped_neighbors: Vec<(usize, f64)> = neighbors
                    .iter()
                    .filter(|(n, _)| relevant_nodes.contains(n))
                    .filter_map(|(n, w)| node_mapping.get(n).map(|&new_n| (new_n, *w)))
                    .collect();
                new_adjacency_list.push(remapped_neighbors);
            }
        }

        // Step 3: Run Leiden on the relevant nodes
        let mut state = LeidenState::initialize(&new_adjacency_list, config.clone())?;

        let mut prev_communities = state.node_to_community.clone();
        for _ in 0..state.config.iterations {
            state.fast_louvain_move_nodes()?;
            state.refine_partition()?;
            state.enforce_gamma_properties()?;

            let _current_communities = state.get_communities()?;
            if state.node_to_community == prev_communities {
                break;
            }
            prev_communities = state.node_to_community.clone();

            if state.should_aggregate() {
                state.aggregate_graph()?;
            }
        }

        state.merge_small_communities(config.min_community_size)?;
        state.resolve_hierarchy()?;

        // Step 4: Map results back to original IDs
        let communities = state.get_communities()?;
        let mut result = HashMap::new();
        for (comm_id, nodes) in communities {
            let remapped_nodes: Vec<usize> = nodes
                .iter()
                .filter_map(|n| reverse_mapping.get(n).copied())
                .collect();
            if !remapped_nodes.is_empty() {
                result.insert(comm_id, remapped_nodes);
            }
        }

        Ok(result)
    }
}

#[cfg(feature = "hgraph")]
#[allow(dead_code)]
impl LeidenState {
    /// Initializes the Leiden algorithm state from an adjacency list and configuration.
    ///
    /// # Arguments
    /// * `adjacency_list` - List of node neighbors with weights.
    /// * `config` - Configuration for the algorithm.
    ///
    /// # Returns
    /// A `Result` containing the initialized `LeidenState`.
    fn initialize(adjacency_list: &[Vec<(usize, f64)>], config: CommunityConfig) -> Result<Self> {
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

    /// Performs the fast Louvain-style node movement phase.
    ///
    /// Updates node community assignments to maximize modularity.
    ///
    /// # Returns
    /// A `Result` indicating success or an error if computation fails.
    fn fast_louvain_move_nodes(&mut self) -> Result<()> {
        let mut nodes: Vec<usize> = (0..self.adjacency_list.len()).collect();
        if self.config.deterministic {
            nodes.sort_unstable();
        } else {
            nodes.shuffle(&mut self.rng);
        }

        let mut queue: VecDeque<usize> = nodes.into_iter().collect();
        let mut processed = HashSet::new();
        let mut _iteration = 0;

        while let Some(node) = queue.pop_front() {
            if processed.contains(&node) {
                continue;
            }
            processed.insert(node);
            _iteration += 1;

            let current_comm = *self
                .node_to_community
                .get(&node)
                .ok_or(CommunityDetectionError::InvalidNodeReference(node))?;
            let mut eligible_communities = self.find_eligible_communities(node, current_comm);
            eligible_communities.push(node);
            if self.config.deterministic {
                eligible_communities.sort_unstable();
            }

            let mut best_comm = current_comm;
            let mut best_delta = 0.0;

            for &comm in &eligible_communities {
                let delta = self.calculate_move_delta(node, comm)?;
                if delta > best_delta {
                    best_delta = delta;
                    best_comm = comm;
                }
            }

            if best_delta > 0.0 && best_comm != current_comm {
                self.move_node(node, best_comm);

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
        }
        Ok(())
    }

    /// Generates a unique order key for deterministic sorting.
    ///
    /// # Arguments
    /// * `node` - The node ID.
    ///
    /// # Returns
    /// A unique key for ordering.
    fn generate_order_key(&mut self, node: usize) -> u64 {
        let mut hasher = DefaultHasher::new();
        hasher.write_u64(self.rng.random::<u64>());
        hasher.write_usize(node);
        hasher.finish()
    }

    /// Refines the current partition into finer communities.
    ///
    /// # Returns
    /// A `Result` indicating success or an error if refinement fails.
    fn refine_partition(&mut self) -> Result<()> {
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
        let level_map = self.node_to_community.clone();
        self.hierarchy.push(level_map);
        Ok(())
    }

    /// Resolves the hierarchy of community assignments to final communities.
    ///
    /// # Returns
    /// A `Result` indicating success or an error if resolution fails.
    fn resolve_hierarchy(&mut self) -> Result<()> {
        let mut final_communities = HashMap::new();
        for node in 0..self.original_size {
            let mut current = node;
            for level in &self.hierarchy {
                current = level.get(&current).copied().unwrap_or(current);
            }
            final_communities.insert(node, current);
        }
        self.node_to_community = final_communities;
        Ok(())
    }

    /// Determines if the graph should be aggregated into a higher-level graph.
    ///
    /// # Returns
    /// `true` if aggregation is beneficial, `false` otherwise.
    fn should_aggregate(&self) -> bool {
        let current_modularity = self.calculate_modularity();
        let communities = self.get_communities().unwrap_or_default();
        communities.values().any(|nodes| nodes.len() > 1) && current_modularity > 0.0
    }

    /// Retrieves the current community assignments.
    ///
    /// # Returns
    /// A `Result` containing a mapping of community IDs to vectors of node IDs.
    pub fn get_communities(&self) -> Result<HashMap<usize, Vec<usize>>> {
        let mut final_mapping = HashMap::new();

        for node in 0..self.original_size {
            let mut current = node;
            for level in &self.hierarchy {
                current = level.get(&current).copied().unwrap_or(current);
            }
            final_mapping
                .entry(current)
                .or_insert_with(Vec::new)
                .push(node);
        }

        for node in 0..self.original_size {
            if !final_mapping.values().any(|nodes| nodes.contains(&node)) {
                final_mapping.insert(node, vec![node]);
            }
        }

        for nodes in final_mapping.values_mut() {
            nodes.sort_unstable();
        }
        Ok(final_mapping)
    }

    /// Finds eligible communities for a node to move into.
    ///
    /// # Arguments
    /// * `node` - The node ID.
    /// * `original_comm` - The current community of the node.
    ///
    /// # Returns
    /// A vector of community IDs the node could join.
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

    /// Calculates the modularity change if a node moves to a new community.
    ///
    /// # Arguments
    /// * `node` - The node ID.
    /// * `new_community` - The target community ID.
    ///
    /// # Returns
    /// A `Result` containing the modularity delta or an error if computation fails.
    fn calculate_move_delta(&self, node: usize, new_community: usize) -> Result<f64> {
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

        Ok(delta_new - delta_old)
    }

    /// Moves a node to a new community and updates weights.
    ///
    /// # Arguments
    /// * `node` - The node ID to move.
    /// * `new_community` - The target community ID.
    fn move_node(&mut self, node: usize, new_community: usize) {
        let old_community = self.node_to_community.get(&node).copied().unwrap_or(node);
        let node_weight = self.community_weights.get(&node).copied().unwrap_or(0.0);

        *self.community_weights.get_mut(&old_community).unwrap() -= node_weight;
        *self.community_weights.entry(new_community).or_insert(0.0) += node_weight;

        self.node_to_community.insert(node, new_community);

        if let Some(current_level) = self.hierarchy.last_mut() {
            current_level.insert(node, new_community);
        }
    }

    /// Creates a singleton partition where each node is its own community.
    ///
    /// # Returns
    /// A mapping of node IDs to their singleton community IDs.
    fn create_singleton_partition(&self) -> HashMap<usize, usize> {
        (0..self.adjacency_list.len()).map(|n| (n, n)).collect()
    }

    /// Aggregates the graph into a higher-level graph based on current communities.
    ///
    /// # Returns
    /// A `Result` indicating success or an error if aggregation fails.
    fn aggregate_graph(&mut self) -> Result<()> {
        let mut communities: Vec<(usize, Vec<usize>)> =
            self.get_communities()?.into_iter().collect();
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

    /// Calculates weights for each community in the aggregated graph.
    ///
    /// # Returns
    /// A `Result` containing a mapping of community IDs to their total weights.
    fn calculate_new_community_weights(&self) -> Result<HashMap<usize, f64>> {
        let weights: HashMap<usize, f64> = self
            .adjacency_list
            .iter()
            .enumerate()
            .map(|(node, edges)| (node, edges.iter().map(|(_, w)| w).sum()))
            .collect();
        Ok(weights)
    }

    /// Calculates the current modularity of the partition.
    ///
    /// # Returns
    /// The modularity value.
    fn calculate_modularity(&self) -> f64 {
        let mut modularity = 0.0;
        let m = self.total_weight.max(f64::EPSILON);

        for (node, &community) in &self.node_to_community {
            let node_weight = self.community_weights.get(node).copied().unwrap_or(0.0);
            if *node < self.adjacency_list.len() {
                for (neighbor, weight) in &self.adjacency_list[*node] {
                    if community == *self.node_to_community.get(neighbor).unwrap() {
                        let neighbor_weight =
                            self.community_weights.get(neighbor).copied().unwrap_or(0.0);
                        modularity += *weight
                            - self.config.resolution
                                * self.config.gamma
                                * (node_weight * neighbor_weight)
                                / (2.0 * m);
                    }
                }
            }
        }
        modularity / (2.0 * m)
    }

    /// Retrieves nodes belonging to a specific community.
    ///
    /// # Arguments
    /// * `comm` - The community ID.
    ///
    /// # Returns
    /// A vector of node IDs in the community.
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

    /// Enforces gamma properties by splitting disconnected communities.
    ///
    /// # Returns
    /// A `Result` indicating success or an error if enforcement fails.
    fn enforce_gamma_properties(&mut self) -> Result<()> {
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

    /// Checks if a community is connected.
    ///
    /// # Arguments
    /// * `community` - The community ID.
    ///
    /// # Returns
    /// `true` if the community is connected, `false` otherwise.
    fn is_community_connected(&self, community: usize) -> bool {
        let nodes = self.get_nodes_in_community(community);
        if nodes.len() <= 1 {
            return true;
        }

        let mut visited = HashSet::new();
        let mut stack = vec![nodes[0]];
        visited.insert(nodes[0]);

        while let Some(node) = stack.pop() {
            if node < self.adjacency_list.len() {
                for &(neighbor, _) in &self.adjacency_list[node] {
                    if nodes.contains(&neighbor) && visited.insert(neighbor) {
                        stack.push(neighbor);
                    }
                }
            }
        }

        visited.len() == nodes.len()
    }

    /// Splits a disconnected community into connected components.
    ///
    /// # Arguments
    /// * `community` - The community ID to split.
    ///
    /// # Returns
    /// A `Result` indicating success or an error if splitting fails.
    fn split_disconnected_community(&mut self, community: usize) -> Result<()> {
        let nodes = self.get_nodes_in_community(community);
        if nodes.is_empty() {
            return Ok(());
        }

        let components = self.find_connected_components(&nodes)?;
        for (i, component) in components.iter().enumerate() {
            let new_comm = if i == 0 {
                community
            } else {
                self.adjacency_list.len() + i
            };
            for &node in component {
                self.move_node(node, new_comm);
            }
        }
        Ok(())
    }

    /// Finds connected components within a set of nodes.
    ///
    /// # Arguments
    /// * `nodes` - The nodes to analyze.
    ///
    /// # Returns
    /// A `Result` containing a vector of connected components, each as a vector of node IDs.
    fn find_connected_components(&self, nodes: &[usize]) -> Result<Vec<Vec<usize>>> {
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
        Ok(components)
    }

    /// Merges communities smaller than the minimum size into larger ones.
    ///
    /// # Arguments
    /// * `min_size` - The minimum acceptable community size.
    ///
    /// # Returns
    /// A `Result` indicating success or an error if merging fails.
    fn merge_small_communities(&mut self, min_size: usize) -> Result<()> {
        let communities = self.get_communities()?;
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
                    self.move_node(*node, best_target);
                }
            }
        }
        Ok(())
    }

    /// Refines a specific community subset into finer partitions.
    ///
    /// # Arguments
    /// * `comm` - The community ID to refine.
    /// * `partition` - The current partition to update.
    ///
    /// # Returns
    /// A `Result` indicating success or an error if refinement fails.
    fn refine_community_subset(
        &mut self,
        comm: usize,
        partition: &mut HashMap<usize, usize>,
    ) -> Result<()> {
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

    /// Calculates the total weight of connections between a node and a community.
    ///
    /// # Arguments
    /// * `node` - The node ID.
    /// * `comm` - The community ID.
    /// * `adjacency_list` - The graph's adjacency list.
    /// * `node_to_community` - Mapping of nodes to their communities.
    ///
    /// # Returns
    /// The total weight of connections.
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

    #[derive(Clone, Debug, Default, PartialEq)]
    struct TestEdge(String);
    impl EdgeType for TestEdge {
        fn as_string(&self) -> String {
            self.0.clone()
        }
    }

    /// Creates a test graph with two distinct communities.
    fn create_test_graph() -> HeterogeneousGraph<f64, TestNode, TestEdge> {
        let mut graph = HeterogeneousGraph::new(false);
        let node0 = graph.add_node(TestNode("Node0".to_string()));
        let node1 = graph.add_node(TestNode("Node1".to_string()));
        let node2 = graph.add_node(TestNode("Node2".to_string()));
        let node3 = graph.add_node(TestNode("Node3".to_string()));
        let node4 = graph.add_node(TestNode("Node4".to_string()));

        // Community 1: clique of nodes 0, 1, 2
        graph
            .add_edge(node0, node1, 1.0, TestEdge("friend".to_string()))
            .unwrap();
        graph
            .add_edge(node1, node2, 1.0, TestEdge("friend".to_string()))
            .unwrap();
        graph
            .add_edge(node0, node2, 1.0, TestEdge("friend".to_string()))
            .unwrap();

        // Community 2: pair of nodes 3, 4
        graph
            .add_edge(node3, node4, 1.0, TestEdge("friend".to_string()))
            .unwrap();

        graph
    }

    #[test]
    fn test_community_detection_basic() {
        let graph = create_test_graph();
        let communities = graph.detect_communities(0.2).unwrap();

        assert_eq!(communities.len(), 2, "Should detect exactly 2 communities");
        let mut found_clique = false;
        let mut found_pair = false;
        for (_, nodes) in &communities {
            if nodes.len() == 3 && nodes.contains(&0) && nodes.contains(&1) && nodes.contains(&2) {
                found_clique = true;
            }
            if nodes.len() == 2 && nodes.contains(&3) && nodes.contains(&4) {
                found_pair = true;
            }
        }
        assert!(
            found_clique,
            "Clique community (0, 1, 2) not found: {:?}",
            communities
        );
        assert!(
            found_pair,
            "Pair community (3, 4) not found: {:?}",
            communities
        );
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

        assert_eq!(
            communities1, communities2,
            "Deterministic runs should match"
        );
        assert_eq!(communities1.len(), 2, "Should detect 2 communities");
        let mut found_clique = false;
        let mut found_pair = false;
        for (_, nodes) in &communities1 {
            if nodes.len() == 3 && nodes.contains(&0) && nodes.contains(&1) && nodes.contains(&2) {
                found_clique = true;
            }
            if nodes.len() == 2 && nodes.contains(&3) && nodes.contains(&4) {
                found_pair = true;
            }
        }
        assert!(
            found_clique,
            "Clique community (0, 1, 2) not found: {:?}",
            communities1
        );
        assert!(
            found_pair,
            "Pair community (3, 4) not found: {:?}",
            communities1
        );
    }

    #[test]
    fn test_empty_graph() {
        let graph: HeterogeneousGraph<f64, TestNode, TestEdge> = HeterogeneousGraph::new(false);
        let communities = graph.detect_communities(1.0).unwrap();
        assert!(
            communities.is_empty(),
            "Empty graph should have no communities"
        );
    }

    #[test]
    fn test_modularity_calculation() {
        let graph = create_test_graph();
        let adjacency_list: Vec<Vec<(usize, f64)>> = graph
            .nodes
            .iter()
            .map(|(node, _)| {
                let mut neighbor_weights: HashMap<usize, f64> = HashMap::new();
                for (neighbor, edges) in graph.get_neighbors(node) {
                    let total_weight: f64 = edges.iter().map(|(_, w)| Into::<f64>::into(*w)).sum();
                    neighbor_weights.insert(neighbor, total_weight);
                }
                neighbor_weights.into_iter().collect()
            })
            .collect();
        let state = LeidenState::initialize(&adjacency_list, CommunityConfig::default()).unwrap();

        let modularity = state.calculate_modularity();
        assert!(
            modularity >= -1.0 && modularity <= 1.0,
            "Modularity should be between -1 and 1"
        );
    }

    #[test]
    fn test_multiple_edges() {
        let mut graph = create_test_graph();
        graph
            .add_edge(0, 1, 2.0, TestEdge("colleague".to_string()))
            .unwrap();

        let communities = graph.detect_communities(1.0).unwrap();
        assert_eq!(
            communities.len(),
            2,
            "Should detect 2 communities with multiple edges"
        );
        let mut found_clique = false;
        let mut found_pair = false;
        for (_, nodes) in &communities {
            if nodes.len() == 3 && nodes.contains(&0) && nodes.contains(&1) && nodes.contains(&2) {
                found_clique = true;
            }
            if nodes.len() == 2 && nodes.contains(&3) && nodes.contains(&4) {
                found_pair = true;
            }
        }
        assert!(
            found_clique,
            "Clique community (0, 1, 2) not found: {:?}",
            communities
        );
        assert!(
            found_pair,
            "Pair community (3, 4) not found: {:?}",
            communities
        );
    }

    #[test]
    fn test_community_detection_by_types() {
        let graph = create_test_graph();
        let config = CommunityConfig {
            deterministic: true,
            seed: Some(42),
            ..CommunityConfig::default()
        };
        let allowed_edge_types = vec![TestEdge("friend".to_string())];
        let communities = graph
            .detect_communities_by_types(&allowed_edge_types, config)
            .unwrap();

        assert_eq!(communities.len(), 2, "Should detect 2 communities");
        let mut found_clique = false;
        let mut found_pair = false;
        for (_, nodes) in &communities {
            if nodes.len() == 3 && nodes.contains(&0) && nodes.contains(&1) && nodes.contains(&2) {
                found_clique = true;
            }
            if nodes.len() == 2 && nodes.contains(&3) && nodes.contains(&4) {
                found_pair = true;
            }
        }
        assert!(
            found_clique,
            "Clique community (0, 1, 2) not found: {:?}",
            communities
        );
        assert!(
            found_pair,
            "Pair community (3, 4) not found: {:?}",
            communities
        );
    }

    #[test]
    fn test_community_detection_by_types_isolated() {
        let mut graph = create_test_graph();
        graph
            .add_edge(1, 2, 1.0, TestEdge("colleague".to_string()))
            .unwrap();

        let config = CommunityConfig {
            deterministic: true,
            seed: Some(42),
            ..CommunityConfig::default()
        };
        let allowed_edge_types = vec![TestEdge("friend".to_string())];
        let communities = graph
            .detect_communities_by_types(&allowed_edge_types, config)
            .unwrap();

        assert_eq!(communities.len(), 2, "Should detect 2 communities");
        let mut found_clique = false;
        let mut found_pair = false;
        for (_, nodes) in &communities {
            if nodes.len() == 3 && nodes.contains(&0) && nodes.contains(&1) && nodes.contains(&2) {
                found_clique = true;
            }
            if nodes.len() == 2 && nodes.contains(&3) && nodes.contains(&4) {
                found_pair = true;
            }
        }
        assert!(
            found_clique,
            "Clique community (0, 1, 2) not found: {:?}",
            communities
        );
        assert!(
            found_pair,
            "Pair community (3, 4) not found: {:?}",
            communities
        );
    }
}
