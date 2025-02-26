use crate::graph::graph::Graph;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

/// A trait for detecting communities in a graph using the Leiden algorithm.
///
/// Implementors of this trait can identify clusters (communities) in a graph based on edge weights
/// and node relationships, optimizing for modularity.
///
/// # Type Parameters
/// - `W`: The weight type, which must be convertible to `f64` and support basic operations.
/// - `N`: The node type, which must be clonable, equatable, hashable, and debuggable.
/// - `E`: The edge type, which must be clonable and debuggable.
pub trait CommunityDetection<W, N, E>
where
    W: Copy + Into<f64> + Default + PartialEq,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    /// Detects communities using a default configuration with the specified gamma value.
    ///
    /// # Arguments
    /// - `gamma`: The resolution parameter controlling community size (higher values yield smaller communities).
    ///
    /// # Returns
    /// A `HashMap` mapping community IDs to vectors of node indices.
    fn detect_communities(&self, gamma: f64) -> HashMap<usize, Vec<usize>>;

    /// Detects communities with a custom configuration.
    ///
    /// # Arguments
    /// - `config`: A `CommunityConfig` struct specifying the detection parameters.
    ///
    /// # Returns
    /// A `HashMap` mapping community IDs to vectors of node indices.
    fn detect_communities_with_config(&self, config: CommunityConfig)
        -> HashMap<usize, Vec<usize>>;
}

/// Configuration for the Leiden community detection algorithm.
///
/// This struct allows customization of the algorithm's behavior, including resolution, iteration count,
/// and randomness control.
#[derive(Debug, Clone)]
pub struct CommunityConfig {
    /// Resolution parameter (gamma) controlling community granularity.
    pub gamma: f64,
    /// Scaling factor for modularity calculation.
    pub resolution: f64,
    /// Maximum number of iterations for the algorithm.
    pub iterations: usize,
    /// Whether to use deterministic behavior (true) or randomized (false).
    pub deterministic: bool,
    /// Optional seed for the random number generator (used if `deterministic` is true).
    pub seed: Option<u64>,
}

impl Default for CommunityConfig {
    /// Provides a default configuration for `CommunityConfig`.
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
/// This struct manages the graph's adjacency list, community assignments, weights, and hierarchy
/// during the clustering process.
struct LeidenState {
    adjacency_list: Vec<Vec<(usize, f64)>>, // Adjacency list representing the graph.
    node_to_community: HashMap<usize, usize>, // Maps nodes to their current community IDs.
    community_weights: HashMap<usize, f64>, // Total weight of edges connected to each community.
    total_weight: f64,                      // Sum of all edge weights in the graph.
    hierarchy: Vec<HashMap<usize, usize>>, // Tracks community assignments at each aggregation level.
    original_size: usize,                  // Number of nodes in the original graph.
    config: CommunityConfig,               // Configuration for the algorithm.
    rng: StdRng,                           // Random number generator for non-deterministic runs.
}

impl<W, N, E> CommunityDetection<W, N, E> for Graph<W, N, E>
where
    W: Copy + Into<f64> + Default + PartialEq,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    fn detect_communities(&self, gamma: f64) -> HashMap<usize, Vec<usize>> {
        self.detect_communities_with_config(CommunityConfig {
            gamma,
            ..CommunityConfig::default()
        })
    }

    /// Implements the Leiden algorithm for community detection with a custom configuration.
    fn detect_communities_with_config(
        &self,
        config: CommunityConfig,
    ) -> HashMap<usize, Vec<usize>> {
        // Convert the graph's nodes and edges into an adjacency list.
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

        let mut state = LeidenState::initialize(&adjacency_list, config);

        let mut prev_communities = state.node_to_community.clone();
        for iteration in 0..state.config.iterations {
            println!("Starting iteration {}", iteration);
            state.fast_louvain_move_nodes(); // Move nodes to improve modularity.
            state.refine_partition(); // Refine the partition into smaller subsets.
            state.enforce_gamma_properties(); // Ensure connectivity constraints.

            if state.node_to_community == prev_communities {
                println!("No changes in communities at iteration {}", iteration);
                break; // Convergence detected.
            }
            prev_communities = state.node_to_community.clone();

            if state.should_aggregate() {
                println!("Aggregating graph at iteration {}", iteration);
                state.aggregate_graph(); // Aggregate communities into super-nodes.
            }
        }

        state.merge_small_communities(2); // Merge communities below minimum size.
        state.resolve_hierarchy(); // Flatten the hierarchy into final communities.
        state.get_communities() // Return the detected communities.
    }
}

#[allow(dead_code)]
impl LeidenState {
    /// Initializes the Leiden state with an adjacency list and configuration.
    fn initialize(adjacency_list: &[Vec<(usize, f64)>], config: CommunityConfig) -> Self {
        let mut node_to_community = HashMap::new();
        let mut community_weights = HashMap::new();
        let mut total_weight = 0.0;

        // Initially, each node is its own community.
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

        Self {
            adjacency_list: adjacency_list.to_vec(),
            node_to_community,
            community_weights,
            total_weight,
            hierarchy: vec![initial_p],
            original_size: adjacency_list.len(),
            config,
            rng,
        }
    }

    /// Performs the fast Louvain-style node movement phase.
    fn fast_louvain_move_nodes(&mut self) {
        let mut nodes: Vec<usize> = (0..self.adjacency_list.len()).collect();
        if self.config.deterministic {
            nodes.sort_unstable(); // Ensure deterministic order.
        } else {
            nodes.shuffle(&mut self.rng); // Randomize for exploration.
        }

        let mut queue: VecDeque<usize> = nodes.into_iter().collect();
        let mut processed = HashSet::new();
        let mut iteration = 0;

        while let Some(node) = queue.pop_front() {
            if processed.contains(&node) {
                continue; // Skip already processed nodes.
            }
            processed.insert(node);
            iteration += 1;
            if iteration % 100 == 0 {
                println!("Processed {} nodes, queue size: {}", iteration, queue.len());
            }

            let current_comm = *self.node_to_community.get(&node).unwrap();
            let mut eligible_communities = self.find_eligible_communities(node, current_comm);
            if self.config.deterministic {
                eligible_communities.sort_unstable();
            }

            let mut best_comm = current_comm;
            let mut best_delta = f64::MIN;

            // Find the community that maximizes modularity gain.
            for comm in &eligible_communities {
                let delta = self.calculate_move_delta(node, *comm);
                if delta > best_delta {
                    best_delta = delta;
                    best_comm = *comm;
                }
            }

            if best_delta > 0.0 && best_comm != current_comm {
                self.move_node(node, best_comm);

                // Add unprocessed neighbors to the queue.
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
                println!("Queue size limit reached, breaking loop");
                break; // Prevent excessive queue growth.
            }
        }
    }

    /// Generates a deterministic order key for a node (used in non-critical paths).
    fn generate_order_key(&mut self, node: usize) -> u64 {
        let mut hasher = DefaultHasher::new();
        hasher.write_u64(self.rng.random::<u64>());
        hasher.write_usize(node);
        hasher.finish()
    }

    /// Refines the current partition into smaller, more cohesive communities.
    fn refine_partition(&mut self) {
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
            self.refine_community_subset(comm, &mut singleton_partition);
        }

        self.node_to_community = singleton_partition;
        let level_map = self.node_to_community.clone();
        self.hierarchy.push(level_map);
    }

    /// Resolves the hierarchy of community assignments into a final mapping.
    fn resolve_hierarchy(&mut self) {
        let mut final_communities = HashMap::new();
        for node in 0..self.original_size {
            let mut current = node;
            for level in &self.hierarchy[1..] {
                current = level.get(&current).copied().unwrap_or(current);
            }
            final_communities.insert(node, current);
        }
        self.node_to_community = final_communities;
    }

    /// Determines whether to aggregate the graph into super-nodes.
    fn should_aggregate(&self) -> bool {
        let current_modularity = self.calculate_modularity();
        let communities = self.get_communities();
        communities.values().any(|nodes| nodes.len() > 1) && current_modularity > 0.0
    }

    /// Retrieves the final community assignments.
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

    /// Finds communities adjacent to a node, excluding its current community.
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
    fn calculate_move_delta(&self, node: usize, new_community: usize) -> f64 {
        let old_community = self.node_to_community.get(&node).copied().unwrap_or(node);
        if old_community == new_community {
            return 0.0;
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

        delta_new - delta_old
    }

    /// Moves a node to a new community and updates weights.
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

    /// Creates a partition where each node is its own community.
    fn create_singleton_partition(&self) -> HashMap<usize, usize> {
        (0..self.adjacency_list.len()).map(|n| (n, n)).collect()
    }

    /// Aggregates the graph into super-nodes based on current communities.
    fn aggregate_graph(&mut self) {
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
        self.community_weights = self.calculate_new_community_weights();
        self.total_weight = self.community_weights.values().sum();
    }

    /// Recalculates community weights after aggregation.
    fn calculate_new_community_weights(&self) -> HashMap<usize, f64> {
        self.adjacency_list
            .iter()
            .enumerate()
            .map(|(node, edges)| (node, edges.iter().map(|(_, w)| w).sum()))
            .collect()
    }

    /// Calculates the current modularity of the graph.
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

    /// Gets all nodes in a given community.
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

    /// Enforces connectivity properties based on gamma.
    fn enforce_gamma_properties(&mut self) {
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
                self.split_disconnected_community(comm);
            }
        }
    }

    /// Checks if a community is connected.
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

    /// Splits a disconnected community into connected components.
    fn split_disconnected_community(&mut self, community: usize) {
        let nodes = self.get_nodes_in_community(community);
        if nodes.is_empty() {
            return;
        }

        let components = self.find_connected_components(&nodes);
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
    }

    /// Finds connected components within a set of nodes.
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

    /// Merges communities smaller than a minimum size into larger ones.
    fn merge_small_communities(&mut self, min_size: usize) {
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
                    self.move_node(*node, best_target);
                }
            }
        }
    }

    /// Refines a single community subset to improve modularity.
    fn refine_community_subset(&mut self, comm: usize, partition: &mut HashMap<usize, usize>) {
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
    }

    /// Calculates the total edge weight between a node and a community.
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

// Updated Tests module
#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to create a simple undirected graph for testing.
    fn create_test_graph() -> Graph<f64, String, String> {
        let mut graph = Graph::new(false); // Undirected graph
        let node0 = graph.add_node("Node0".to_string());
        let node1 = graph.add_node("Node1".to_string());
        let node2 = graph.add_node("Node2".to_string());
        let _node3 = graph.add_node("Node3".to_string()); // Prefixed with _ since unused

        // Create a clique (0-1-2) and an isolated node (3)
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
        let communities = graph.detect_communities(1.0);

        // Expect two communities: {0, 1, 2} (clique) and {3} (isolated)
        assert_eq!(communities.len(), 2, "Should detect exactly 2 communities");

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
        assert!(found_clique, "Clique community (0, 1, 2) not found");
        assert!(found_isolated, "Isolated node community (3) not found");
    }

    #[test]
    fn test_deterministic_behavior() {
        let graph = create_test_graph();
        let config = CommunityConfig {
            deterministic: true,
            seed: Some(42),
            ..CommunityConfig::default()
        };
        let communities1 = graph.detect_communities_with_config(config.clone());
        let communities2 = graph.detect_communities_with_config(config);

        // Deterministic runs should produce identical results.
        assert_eq!(
            communities1, communities2,
            "Deterministic runs should match"
        );
    }

    #[test]
    fn test_empty_graph() {
        let graph: Graph<f64, String, String> = Graph::new(false);
        let communities = graph.detect_communities(1.0);
        assert!(
            communities.is_empty(),
            "Empty graph should have no communities"
        );
    }

    #[test]
    fn test_single_node() {
        let mut graph: Graph<f64, String, String> = Graph::new(false); // Explicit type annotation
        let node = graph.add_node("Single".to_string());
        let communities = graph.detect_communities(1.0);
        assert_eq!(
            communities.len(),
            1,
            "Single node should form one community"
        );
        assert_eq!(
            communities.get(&node),
            Some(&vec![node]),
            "Node should be in its own community"
        );
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
        let state = LeidenState::initialize(&adjacency_list, CommunityConfig::default()); // Removed mut

        let modularity = state.calculate_modularity();
        assert!(modularity >= 0.0, "Modularity should be non-negative");
    }

    #[test]
    fn test_directed_graph() {
        let mut graph = Graph::new(true); // Directed graph
        let node0 = graph.add_node("Node0".to_string());
        let node1 = graph.add_node("Node1".to_string());
        let node2 = graph.add_node("Node2".to_string());

        // Create a directed cycle: 0 -> 1 -> 2 -> 0
        graph
            .add_edge(node0, node1, 1.0, "e01".to_string())
            .unwrap();
        graph
            .add_edge(node1, node2, 1.0, "e12".to_string())
            .unwrap();
        graph
            .add_edge(node2, node0, 1.0, "e20".to_string())
            .unwrap();

        let communities = graph.detect_communities(1.0);
        assert_eq!(
            communities.len(),
            1,
            "Directed cycle should form one community"
        );
        let nodes = communities.values().next().unwrap();
        assert_eq!(nodes.len(), 3, "All nodes should be in the community");
        assert!(nodes.contains(&node0) && nodes.contains(&node1) && nodes.contains(&node2));
    }
}
