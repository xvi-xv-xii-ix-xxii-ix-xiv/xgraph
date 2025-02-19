//! Graph community detection algorithms
//!
//! Provides the Leiden algorithm implementation for finding high-quality
//! community structures in graphs with various optimization parameters
//!
//! # Examples
//!
//! Basic usage:
//! ```rust
//! use xgraph::graph::graph::Graph;
//! use xgraph::algorithms::leiden_clustering::{CommunityDetection, CommunityConfig};
//!
//! let mut graph = Graph::<f64, (), ()>::new(true);
//! let n0 = graph.add_node(());
//! let n1 = graph.add_node(());
//! graph.add_edge(n0, n1, 1.0, ()).unwrap();
//!
//! let communities = graph.detect_communities(1.0);
//! assert!(!communities.is_empty());
//! ```

use crate::graph::graph::Graph;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;

/// Trait for community detection operations
pub trait CommunityDetection<W, N, E>
where
    W: Copy + Into<f64> + Default + PartialEq,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    /// Detect communities using the Leiden algorithm
    ///
    /// # Arguments
    /// * `gamma` - Resolution parameter (higher values find smaller communities)
    fn detect_communities(&self, gamma: f64) -> HashMap<usize, Vec<usize>>;

    /// Detect communities with additional parameters
    fn detect_communities_with_config(&self, config: CommunityConfig)
        -> HashMap<usize, Vec<usize>>;
}

/// Configuration parameters for community detection
#[derive(Debug, Clone)]
pub struct CommunityConfig {
    pub gamma: f64,
    pub resolution: f64,
    pub iterations: usize,
}

impl Default for CommunityConfig {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            resolution: 1.0,
            iterations: 10,
        }
    }
}

/// Internal state for Leiden algorithm
struct LeidenState {
    adjacency_list: Vec<Vec<(usize, f64)>>,
    node_to_community: HashMap<usize, usize>,
    community_weights: HashMap<usize, f64>,
    total_weight: f64,
    hierarchy: Vec<HashMap<usize, usize>>,
    original_size: usize,
    config: CommunityConfig,
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

    fn detect_communities_with_config(
        &self,
        config: CommunityConfig,
    ) -> HashMap<usize, Vec<usize>> {
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

        for _ in 0..state.config.iterations {
            state.fast_louvain_move_nodes();
            state.refine_partition();
            state.enforce_gamma_properties();

            if state.check_stopping_condition() {
                break;
            }

            if state.should_aggregate() {
                state.aggregate_graph();
            }
        }

        state.resolve_hierarchy();
        state.get_communities()
    }
}

#[allow(dead_code)]
impl LeidenState {
    fn initialize(adjacency_list: &[Vec<(usize, f64)>], config: CommunityConfig) -> Self {
        let mut node_to_community = HashMap::new();
        let mut community_weights = HashMap::new();
        let mut total_weight = 0.0;

        for (node, neighbors) in adjacency_list.iter().enumerate() {
            node_to_community.insert(node, node);
            let weight = neighbors.iter().map(|(_, w)| w).sum();
            community_weights.insert(node, weight);
            total_weight += weight;
        }

        Self {
            adjacency_list: adjacency_list.to_vec(),
            node_to_community,
            community_weights,
            total_weight,
            hierarchy: vec![HashMap::new()],
            original_size: adjacency_list.len(),
            config,
        }
    }

    fn fast_louvain_move_nodes(&mut self) {
        let mut queue: VecDeque<usize> = (0..self.adjacency_list.len()).collect();

        while let Some(node) = queue.pop_front() {
            let current_comm = self.node_to_community[&node];
            let eligible_communities = self.find_eligible_communities(node, current_comm);

            let mut best_comm = current_comm;
            let mut best_delta = f64::MIN;

            for &comm in &eligible_communities {
                let delta = self.calculate_move_delta(node, comm);
                if delta > best_delta {
                    best_delta = delta;
                    best_comm = comm;
                }
            }

            if best_delta > 0.0 && best_comm != current_comm {
                self.move_node(node, best_comm);

                for (neighbor, _) in &self.adjacency_list[node] {
                    queue.push_back(*neighbor);
                }
            }
        }
    }

    fn calculate_new_community_delta(&self, node: usize) -> f64 {
        let current_comm = self.node_to_community[&node];
        let k_i = self.community_weights[&node];
        let m = self.total_weight.max(f64::EPSILON);
        -(self.community_weights[&current_comm] - k_i) * k_i / (2.0 * m)
    }

    fn refine_partition(&mut self) {
        let mut rng = rand::rng();
        let mut singleton_partition = self.create_singleton_partition();

        for comm in self
            .node_to_community
            .values()
            .copied()
            .collect::<HashSet<_>>()
        {
            self.refine_community_subset(comm, &mut singleton_partition, &mut rng);
        }

        for node in 0..self.adjacency_list.len() {
            let delta = self.calculate_new_community_delta(node);
            if delta > 0.0 {
                self.move_node(node, node);
            }
        }

        let mut level_map = HashMap::new();
        for (node, comm) in &singleton_partition {
            level_map.insert(*node, *comm);
        }
        self.hierarchy.push(level_map);
    }

    fn resolve_hierarchy(&mut self) {
        // Reconstruct node_to_community from hierarchy
        let mut final_communities = HashMap::new();
        for node in 0..self.original_size {
            let mut current = node;
            for level in &self.hierarchy {
                current = *level.get(&current).unwrap_or(&current); // Use current as fallback
            }
            final_communities.insert(node, current);
        }
        self.node_to_community = final_communities;
    }

    fn should_aggregate(&self) -> bool {
        let unique_communities = self
            .node_to_community
            .values()
            .collect::<HashSet<_>>()
            .len();
        unique_communities < self.adjacency_list.len() / 2
    }

    pub fn get_communities(&self) -> HashMap<usize, Vec<usize>> {
        let mut final_mapping = HashMap::new();

        // Build mapping through hierarchy
        for node in 0..self.original_size {
            let mut current = node;
            for level in &self.hierarchy {
                current = *level.get(&current).unwrap_or(&current); // Use current as fallback
            }
            final_mapping
                .entry(current)
                .or_insert_with(Vec::new)
                .push(node);
        }

        final_mapping
    }

    fn refine_community_subset(
        &mut self,
        comm: usize,
        partition: &mut HashMap<usize, usize>,
        rng: &mut ThreadRng,
    ) {
        let nodes: Vec<usize> = self
            .node_to_community
            .iter()
            .filter(|(_, &c)| c == comm)
            .map(|(&n, _)| n)
            .collect();

        for &node in &nodes {
            let current_comm = self.node_to_community[&node];
            let mut best_comm = current_comm;
            let mut best_delta = f64::MIN;

            for (neighbor, _) in &self.adjacency_list[node] {
                let neighbor_comm = self.node_to_community[neighbor];
                if neighbor_comm != current_comm {
                    let delta = self.calculate_community_connections(node, neighbor_comm);
                    if delta > best_delta {
                        best_delta = delta;
                        best_comm = neighbor_comm;
                    }
                }
            }

            if best_delta > 0.0
                && rng.random::<f64>() < (best_delta.exp() / (1.0 + best_delta.exp()))
            {
                // Исправлено на актуальный метод генерации
                if best_delta > 0.0
                    && rng.random::<f64>() < (best_delta.exp() / (1.0 + best_delta.exp()))
                {
                    partition.insert(node, best_comm);
                }
            }
        }
    }

    fn find_best_community(&self, node: usize) -> usize {
        let current_community = self.node_to_community[&node];
        let mut best_community = current_community;
        let mut best_delta = f64::MIN;

        let neighbor_communities = self.get_neighbor_communities(node);

        for &comm in &neighbor_communities {
            let delta = self.calculate_move_delta(node, comm);
            if delta > best_delta {
                best_delta = delta;
                best_community = comm;
            }
        }

        best_community
    }

    fn create_singleton_partition(&self) -> HashMap<usize, usize> {
        (0..self.adjacency_list.len()).map(|n| (n, n)).collect()
    }

    fn is_gamma_separated(&self) -> bool {
        let communities: HashSet<usize> = self.node_to_community.values().copied().collect();

        for &comm1 in &communities {
            for &comm2 in &communities {
                if comm1 == comm2 {
                    continue;
                }

                let e = self.calculate_internal_community_connections(comm1);
                let k1 = self.community_weights[&comm1];
                let k2 = self.community_weights[&comm2];

                if e > self.config.gamma * k1 * k2 / (2.0 * self.total_weight) {
                    return false;
                }
            }
        }
        true
    }

    fn calculate_internal_community_connections(&self, comm: usize) -> f64 {
        let nodes = self.get_nodes_in_community(comm);
        let mut e = 0.0;
        for node in &nodes {
            // Используем ссылку
            e += self.adjacency_list[*node]
                .iter()
                .filter(|(n, _)| nodes.contains(n))
                .map(|(_, w)| w)
                .sum::<f64>();
        }
        e
    }

    fn get_nodes_in_community(&self, comm: usize) -> Vec<usize> {
        self.node_to_community
            .iter()
            .filter(|(_, &c)| c == comm)
            .map(|(&n, _)| n)
            .collect()
    }

    fn is_community_gamma_connected(&self, comm: usize) -> bool {
        let nodes = self.get_nodes_in_community(comm);
        if nodes.len() <= 1 {
            return true;
        }

        let internal_connections = self.calculate_internal_community_connections(comm);
        let total_degree = self.community_weights[&comm];
        internal_connections >= self.config.gamma * total_degree * (total_degree - 1.0) / 2.0
    }

    fn enforce_gamma_properties(&mut self) {
        let communities: HashSet<usize> = self.node_to_community.values().copied().collect();
        for comm in communities {
            let connections = self.get_community_connections(comm);
            for (target_comm, &weight) in &connections {
                if *target_comm != comm {
                    let k1 = self.community_weights[&comm];
                    let k2 = self.community_weights[target_comm];
                    if weight > (self.config.gamma * k1 * k2 / (2.0 * self.total_weight)) as usize {
                        self.split_disconnected_community(comm);
                    }
                }
            }
        }
    }

    fn aggregate_graph(&mut self) {
        let communities = self.get_communities();
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
                for (neighbor, weight) in &self.adjacency_list[node] {
                    let target_comm = index_mapping[neighbor];
                    *aggregated_edges.entry(target_comm).or_insert(0.0) += weight;
                }
            }
            new_adjacency_list[new_id] = aggregated_edges.into_iter().collect();
        }

        self.adjacency_list = new_adjacency_list;
        self.node_to_community = index_mapping;
        self.community_weights = self.calculate_new_community_weights();
        self.total_weight = self.community_weights.values().sum();
    }

    fn calculate_modularity(&self) -> f64 {
        let mut modularity = 0.0;
        let m = self.total_weight.max(f64::EPSILON);

        for (node, &community) in &self.node_to_community {
            let node_weight = self.community_weights[node];
            for (neighbor, weight) in &self.adjacency_list[*node] {
                if community == self.node_to_community[neighbor] {
                    let neighbor_weight = self.community_weights[neighbor];
                    modularity += *weight
                        - self.config.resolution
                            * self.config.gamma
                            * (node_weight * neighbor_weight)
                            / (2.0 * m);
                }
            }
        }
        modularity / (2.0 * m)
    }

    fn is_subpartition_gamma_dense(&self, comm: usize) -> bool {
        let nodes = self.get_nodes_in_community(comm);
        if nodes.len() <= 1 {
            return true;
        }

        for i in 1..nodes.len() {
            let part1 = &nodes[..i];
            let part2 = &nodes[i..];

            let e = self.calculate_inter_part_connections(part1, part2);
            let k1: f64 = part1.iter().map(|&n| self.community_weights[&n]).sum();
            let k2: f64 = part2.iter().map(|&n| self.community_weights[&n]).sum();

            if e < self.config.gamma * k1 * k2 / (2.0 * self.total_weight) {
                return false;
            }
        }
        true
    }

    fn calculate_inter_part_connections(&self, part1: &[usize], part2: &[usize]) -> f64 {
        let mut e = 0.0;
        for &n1 in part1 {
            for (n2, w) in &self.adjacency_list[n1] {
                if part2.contains(n2) {
                    e += w;
                }
            }
        }
        e
    }

    fn are_all_communities_gamma_connected(&self) -> bool {
        self.node_to_community
            .values()
            .collect::<HashSet<_>>()
            .iter()
            .all(|&&comm| self.is_community_gamma_connected(comm))
    }

    fn find_eligible_communities(&self, node: usize, original_comm: usize) -> HashSet<usize> {
        let gamma = self.config.gamma;
        let degree_node = self.community_weights[&node];

        self.adjacency_list[node]
            .iter()
            .filter_map(|(neighbor, w)| {
                let comm = self.node_to_community[neighbor];
                (comm != original_comm
                    && *w >= gamma * degree_node * self.community_weights[neighbor])
                    .then_some(comm)
            })
            .collect()
    }

    fn calculate_community_connections(&self, node: usize, comm: usize) -> f64 {
        self.adjacency_list[node]
            .iter()
            .filter(|(n, _)| self.node_to_community[n] == comm)
            .map(|(_, w)| w)
            .sum()
    }

    fn get_neighbor_communities(&self, node: usize) -> HashSet<usize> {
        self.adjacency_list[node]
            .iter()
            .map(|(n, _)| self.node_to_community[n])
            .chain(std::iter::once(self.node_to_community[&node]))
            .collect()
    }

    fn get_community_connections(&self, community: usize) -> HashMap<usize, usize> {
        let mut connections = HashMap::new();
        for node in self.get_nodes_in_community(community) {
            for (neighbor, _) in &self.adjacency_list[node] {
                let target = self.node_to_community[neighbor];
                *connections.entry(target).or_insert(0) += 1;
            }
        }
        connections
    }

    fn calculate_new_community_weights(&self) -> HashMap<usize, f64> {
        self.adjacency_list
            .iter()
            .enumerate()
            .map(|(node, edges)| (node, edges.iter().map(|(_, w)| w).sum()))
            .collect()
    }

    fn calculate_move_delta(&self, node: usize, new_community: usize) -> f64 {
        let old_community = self.node_to_community[&node];
        if old_community == new_community {
            return 0.0;
        }

        let k_i = self.community_weights[&node];
        let sum_tot_new = self.community_weights[&new_community];
        let sum_tot_old = self.community_weights[&old_community] - k_i;
        let m = self.total_weight.max(f64::EPSILON);

        let sum_in: f64 = self.adjacency_list[node]
            .iter()
            .filter(|(n, _)| self.node_to_community[n] == new_community)
            .map(|(_, w)| w)
            .sum();

        (sum_in - self.config.gamma * k_i * sum_tot_new / (2.0 * m))
            - (sum_tot_old * k_i) / (2.0 * m)
    }

    fn move_node(&mut self, node: usize, new_community: usize) {
        let old_community = self.node_to_community[&node];
        let node_weight = self.community_weights[&node];

        *self.community_weights.get_mut(&old_community).unwrap() -= node_weight;
        *self.community_weights.entry(new_community).or_insert(0.0) += node_weight;

        self.node_to_community.insert(node, new_community);

        if let Some(current_level) = self.hierarchy.last_mut() {
            current_level.insert(node, new_community);
        }
    }

    fn check_stopping_condition(&self) -> bool {
        self.adjacency_list.len() == self.node_to_community.len()
    }

    fn reset_partition(&mut self) {
        self.node_to_community.clear();
        self.community_weights.clear();

        for i in 0..self.adjacency_list.len() {
            self.node_to_community.insert(i, i);
            let weight = self.adjacency_list[i].iter().map(|(_, w)| w).sum();
            self.community_weights.insert(i, weight);
        }

        self.total_weight = self.community_weights.values().sum();
    }

    fn normalize_weights(&mut self) {
        let max_weight = self
            .adjacency_list
            .iter()
            .flat_map(|edges| edges.iter().map(|(_, w)| w))
            .fold(0.0f64, |a, &b| a.max(b));

        if max_weight > 0.0 {
            for edges in &mut self.adjacency_list {
                for (_, w) in edges {
                    *w /= max_weight;
                }
            }
            // Обновляем веса сообществ
            for (node, weight) in self.community_weights.iter_mut() {
                *weight = self.adjacency_list[*node].iter().map(|(_, w)| w).sum();
            }
        }
    }

    fn refinement_phase(&mut self) {
        let mut rng = rand::rng();
        let mut nodes: Vec<usize> = (0..self.adjacency_list.len()).collect();
        nodes.shuffle(&mut rng);

        for node in nodes {
            let best_community = self.find_best_community(node);
            if best_community != self.node_to_community[&node] {
                self.move_node(node, best_community);
            }
        }
    }

    fn is_community_connected(&self, community: usize) -> bool {
        let nodes: Vec<usize> = self.get_nodes_in_community(community);
        if nodes.len() <= 1 {
            return true;
        }

        let mut visited = HashSet::new();
        let mut stack = vec![*nodes.first().unwrap()];

        while let Some(node) = stack.pop() {
            if visited.insert(node) {
                for (neighbor, _) in &self.adjacency_list[node] {
                    if nodes.contains(neighbor) && !visited.contains(neighbor) {
                        stack.push(*neighbor);
                    }
                }
            }
        }

        visited.len() == nodes.len()
    }

    fn merge_small_communities(&mut self, min_size: usize) {
        let communities = self.get_communities();
        let mut to_merge = Vec::new();

        for (comm, nodes) in &communities {
            if nodes.len() < min_size {
                to_merge.push(*comm);
            }
        }

        for comm in to_merge {
            let mut best_target = comm;
            let mut max_connections = 0;

            for node in &communities[&comm] {
                for (neighbor, _) in &self.adjacency_list[*node] {
                    let target = self.node_to_community[neighbor];
                    if target != comm {
                        let count = communities.get(&target).map_or(0, |v| v.len());
                        if count > max_connections {
                            max_connections = count;
                            best_target = target;
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

    fn check_community_connectivity(&mut self) {
        let communities: HashSet<_> = self.node_to_community.values().copied().collect();
        for comm in communities {
            if !self.is_community_connected(comm) {
                self.split_disconnected_community(comm);
            }
        }
    }

    fn split_disconnected_community(&mut self, community: usize) {
        let nodes = self.get_nodes_in_community(community);
        if nodes.is_empty() {
            return;
        }

        let components = self.find_connected_components(&nodes);
        for (i, component) in components.iter().enumerate() {
            let new_comm = community + i + 1;
            for &node in component {
                self.move_node(node, new_comm);
            }
        }
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
                    if visited.insert(current) {
                        component.push(current);
                        for (neighbor, _) in &self.adjacency_list[current] {
                            if node_set.contains(neighbor) && !visited.contains(neighbor) {
                                stack.push(*neighbor);
                            }
                        }
                    }
                }
                components.push(component);
            }
        }
        components
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_community_connection() {
        let mut graph = Graph::<f64, (), ()>::new(true);
        let nodes: Vec<_> = (0..10).map(|_| graph.add_node(())).collect();

        let edges = [
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 3),
            (3, 2),
            (3, 4),
            (3, 5),
            (4, 3),
            (4, 5),
            (5, 3),
            (5, 4),
            (6, 7),
            (6, 8),
            (7, 6),
            (7, 8),
            (8, 6),
            (8, 7),
            (8, 9),
            (9, 8),
        ];

        for (src, dst) in edges {
            graph.add_edge(nodes[src], nodes[dst], 1.0, ()).unwrap();
        }

        let communities = graph.detect_communities(0.2);
        assert_eq!(communities.len(), 2);
        assert!(communities.values().all(|c| !c.is_empty()));

        let total_nodes: usize = communities.values().map(|v| v.len()).sum();
        assert_eq!(total_nodes, 10);
    }
}
