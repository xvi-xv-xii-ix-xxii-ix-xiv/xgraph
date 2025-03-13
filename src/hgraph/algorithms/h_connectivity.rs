//! Connectivity analysis algorithms for heterogeneous multigraphs.
//!
//! This module provides implementations of connectivity-related algorithms tailored for heterogeneous
//! multigraphs, where nodes and edges can have distinct types. It includes methods to identify weakly
//! and strongly connected components, check overall connectivity, and perform these analyses either
//! across all edges or filtered by specific edge types.

#[cfg(feature = "hgraph")]
use crate::hgraph::h_graph::HeterogeneousGraph;
#[cfg(feature = "hgraph")]
use std::collections::{HashSet, VecDeque};
#[cfg(feature = "hgraph")]
use std::fmt::Debug;
#[cfg(feature = "hgraph")]
use std::hash::Hash;

// Error handling additions

/// Error type for connectivity computation failures.
///
/// Represents errors that may occur during connectivity analysis, such as referencing non-existent nodes.
#[derive(Debug)]
pub enum ConnectivityError {
    /// Indicates a node referenced in the graph does not exist.
    InvalidNodeReference(usize),
}

impl std::fmt::Display for ConnectivityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectivityError::InvalidNodeReference(id) => {
                write!(f, "Invalid node reference: node ID {} not found", id)
            }
        }
    }
}

impl std::error::Error for ConnectivityError {}

/// Result type alias for connectivity computations.
///
/// Wraps the result of connectivity calculations, allowing for error handling.
pub type Result<T> = std::result::Result<T, ConnectivityError>;

// Enhanced trait with documentation and error handling

/// Trait defining connectivity measures for heterogeneous graphs.
///
/// Provides methods to analyze connectivity in a `HeterogeneousGraph`, including finding connected
/// components and checking connectivity properties. Each method is available in two variants: one
/// considering all edges and another filtered by edge types. All methods return a `Result` to handle
/// potential errors gracefully instead of panicking.
#[cfg(feature = "hgraph")]
pub trait HeteroConnectivity<W, N, E> {
    /// Finds weakly connected components in the graph.
    ///
    /// Weakly connected components are sets of nodes where there is an undirected path between every
    /// pair of nodes, ignoring edge direction in directed graphs.
    ///
    /// # Returns
    /// A `Result` containing a vector of vectors, where each inner vector represents a weakly connected component.
    fn find_weakly_connected_components(&self) -> Result<Vec<Vec<usize>>>;

    /// Finds strongly connected components in the graph.
    ///
    /// Strongly connected components are sets of nodes where there is a directed path from every node
    /// to every other node, applicable only in directed graphs.
    ///
    /// # Returns
    /// A `Result` containing a vector of vectors, where each inner vector represents a strongly connected component.
    fn find_strongly_connected_components(&self) -> Result<Vec<Vec<usize>>>;

    /// Finds connected components based on graph directionality.
    ///
    /// For undirected graphs, this returns weakly connected components. For directed graphs, it returns
    /// strongly connected components.
    ///
    /// # Returns
    /// A `Result` containing a vector of vectors, where each inner vector represents a connected component.
    fn find_connected_components(&self) -> Result<Vec<Vec<usize>>> {
        if self.is_directed() {
            self.find_strongly_connected_components()
        } else {
            self.find_weakly_connected_components()
        }
    }

    /// Checks if the graph is weakly connected.
    ///
    /// A graph is weakly connected if there is an undirected path between every pair of nodes,
    /// ignoring edge direction in directed graphs.
    ///
    /// # Returns
    /// A `Result` containing `true` if the graph is weakly connected, `false` otherwise.
    fn is_weakly_connected(&self) -> Result<bool>;

    /// Checks if the graph is strongly connected.
    ///
    /// A graph is strongly connected if there is a directed path from every node to every other node,
    /// applicable only in directed graphs.
    ///
    /// # Returns
    /// A `Result` containing `true` if the graph is strongly connected, `false` otherwise.
    fn is_strongly_connected(&self) -> Result<bool>;

    /// Checks if the graph is connected based on its directionality.
    ///
    /// For undirected graphs, this checks weak connectivity. For directed graphs, it checks strong connectivity.
    ///
    /// # Returns
    /// A `Result` containing `true` if the graph is connected, `false` otherwise.
    fn is_connected(&self) -> Result<bool> {
        if self.is_directed() {
            self.is_strongly_connected()
        } else {
            self.is_weakly_connected()
        }
    }

    /// Finds weakly connected components considering only specified edge types.
    ///
    /// # Arguments
    /// * `allowed_edge_types` - A slice of edge types to filter edges by. If empty, all edges are considered.
    ///
    /// # Returns
    /// A `Result` containing a vector of vectors, where each inner vector represents a weakly connected component.
    fn find_weakly_connected_components_by_types(
        &self,
        allowed_edge_types: &[E],
    ) -> Result<Vec<Vec<usize>>>;

    /// Finds strongly connected components considering only specified edge types.
    ///
    /// # Arguments
    /// * `allowed_edge_types` - A slice of edge types to filter edges by. If empty, all edges are considered.
    ///
    /// # Returns
    /// A `Result` containing a vector of vectors, where each inner vector represents a strongly connected component.
    fn find_strongly_connected_components_by_types(
        &self,
        allowed_edge_types: &[E],
    ) -> Result<Vec<Vec<usize>>>;

    /// Finds connected components based on graph directionality, considering only specified edge types.
    ///
    /// For undirected graphs, this returns weakly connected components. For directed graphs, it returns
    /// strongly connected components.
    ///
    /// # Arguments
    /// * `allowed_edge_types` - A slice of edge types to filter edges by. If empty, all edges are considered.
    ///
    /// # Returns
    /// A `Result` containing a vector of vectors, where each inner vector represents a connected component.
    fn find_connected_components_by_types(
        &self,
        allowed_edge_types: &[E],
    ) -> Result<Vec<Vec<usize>>> {
        if self.is_directed() {
            self.find_strongly_connected_components_by_types(allowed_edge_types)
        } else {
            self.find_weakly_connected_components_by_types(allowed_edge_types)
        }
    }

    /// Checks if the graph is weakly connected considering only specified edge types.
    ///
    /// # Arguments
    /// * `allowed_edge_types` - A slice of edge types to filter edges by. If empty, all edges are considered.
    ///
    /// # Returns
    /// A `Result` containing `true` if the graph is weakly connected, `false` otherwise.
    fn is_weakly_connected_by_types(&self, allowed_edge_types: &[E]) -> Result<bool>;

    /// Checks if the graph is strongly connected considering only specified edge types.
    ///
    /// # Arguments
    /// * `allowed_edge_types` - A slice of edge types to filter edges by. If empty, all edges are considered.
    ///
    /// # Returns
    /// A `Result` containing `true` if the graph is strongly connected, `false` otherwise.
    fn is_strongly_connected_by_types(&self, allowed_edge_types: &[E]) -> Result<bool>;

    /// Checks if the graph is connected based on its directionality, considering only specified edge types.
    ///
    /// For undirected graphs, this checks weak connectivity. For directed graphs, it checks strong connectivity.
    ///
    /// # Arguments
    /// * `allowed_edge_types` - A slice of edge types to filter edges by. If empty, all edges are considered.
    ///
    /// # Returns
    /// A `Result` containing `true` if the graph is connected, `false` otherwise.
    fn is_connected_by_types(&self, allowed_edge_types: &[E]) -> Result<bool> {
        if self.is_directed() {
            self.is_strongly_connected_by_types(allowed_edge_types)
        } else {
            self.is_weakly_connected_by_types(allowed_edge_types)
        }
    }

    /// Determines if the graph is directed.
    ///
    /// # Returns
    /// `true` if the graph is directed, `false` if undirected.
    fn is_directed(&self) -> bool;
}

#[cfg(feature = "hgraph")]
impl<W, N, E> HeteroConnectivity<W, N, E> for HeterogeneousGraph<W, N, E>
where
    W: Copy + Default + PartialEq + Debug,
    N: Clone + Eq + Hash + Debug + crate::hgraph::h_node::NodeType,
    E: Clone + Default + Debug + crate::hgraph::h_edge::EdgeType + PartialEq,
{
    fn find_weakly_connected_components(&self) -> Result<Vec<Vec<usize>>> {
        self.find_weakly_connected_components_by_types(&[])
    }

    fn find_strongly_connected_components(&self) -> Result<Vec<Vec<usize>>> {
        self.find_strongly_connected_components_by_types(&[])
    }

    fn is_weakly_connected(&self) -> Result<bool> {
        Ok(self.find_weakly_connected_components()?.len() <= 1)
    }

    fn is_strongly_connected(&self) -> Result<bool> {
        if self.nodes.is_empty() {
            return Ok(true);
        }
        let start_node = self.nodes.iter().next().unwrap().0;
        Ok(self.strong_connectivity_check_by_types(start_node, &[]))
    }

    fn find_weakly_connected_components_by_types(
        &self,
        allowed_edge_types: &[E],
    ) -> Result<Vec<Vec<usize>>> {
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

                    let mut neighbors = self
                        .get_neighbors_by_types(current, allowed_edge_types)
                        .into_iter()
                        .map(|(n, _)| n)
                        .collect::<HashSet<_>>();

                    if self.is_directed() {
                        neighbors
                            .extend(self.get_predecessors_by_types(current, allowed_edge_types));
                    }

                    for neighbor in neighbors {
                        if !self.nodes.contains(neighbor) {
                            return Err(ConnectivityError::InvalidNodeReference(neighbor));
                        }
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            queue.push_back(neighbor);
                        }
                    }
                }
                components.push(component);
            }
        }
        Ok(components)
    }

    fn find_strongly_connected_components_by_types(
        &self,
        allowed_edge_types: &[E],
    ) -> Result<Vec<Vec<usize>>> {
        let mut visited = HashSet::new();
        let mut order = Vec::with_capacity(self.nodes.len());

        for node in self.nodes.iter().map(|(id, _)| id) {
            if !visited.contains(&node) {
                self.dfs_order_by_types(node, allowed_edge_types, &mut visited, &mut order);
            }
        }

        let transposed = self.transpose_by_types(allowed_edge_types);
        let mut visited = HashSet::new();
        let mut components = Vec::new();

        for &node in order.iter().rev() {
            if !self.nodes.contains(node) {
                return Err(ConnectivityError::InvalidNodeReference(node));
            }
            if !visited.contains(&node) {
                let mut component = Vec::new();
                transposed.dfs_collect_by_types(
                    node,
                    allowed_edge_types,
                    &mut visited,
                    &mut component,
                );
                component.sort();
                components.push(component);
            }
        }
        Ok(components)
    }

    fn is_weakly_connected_by_types(&self, allowed_edge_types: &[E]) -> Result<bool> {
        Ok(self
            .find_weakly_connected_components_by_types(allowed_edge_types)?
            .len()
            <= 1)
    }

    fn is_strongly_connected_by_types(&self, allowed_edge_types: &[E]) -> Result<bool> {
        if self.nodes.is_empty() {
            return Ok(true);
        }
        let start_node = self.nodes.iter().next().unwrap().0;
        Ok(self.strong_connectivity_check_by_types(start_node, allowed_edge_types))
    }

    fn is_directed(&self) -> bool {
        self.directed
    }
}

#[cfg(feature = "hgraph")]
impl<W, N, E> HeterogeneousGraph<W, N, E>
where
    W: Copy + Default + PartialEq + Debug,
    N: Clone + Eq + Hash + Debug + crate::hgraph::h_node::NodeType,
    E: Clone + Default + Debug + crate::hgraph::h_edge::EdgeType + PartialEq,
{
    /// Performs a depth-first search to determine the order of nodes for strongly connected components.
    ///
    /// # Arguments
    /// * `node` - The starting node ID.
    /// * `allowed_edge_types` - A slice of edge types to filter edges by.
    /// * `visited` - A set tracking visited nodes.
    /// * `order` - A vector storing the DFS finish order.
    fn dfs_order_by_types(
        &self,
        node: usize,
        allowed_edge_types: &[E],
        visited: &mut HashSet<usize>,
        order: &mut Vec<usize>,
    ) {
        visited.insert(node);
        for (neighbor, _edge) in self.get_neighbors_by_types(node, allowed_edge_types) {
            if !visited.contains(&neighbor) {
                self.dfs_order_by_types(neighbor, allowed_edge_types, visited, order);
            }
        }
        order.push(node);
    }

    /// Collects nodes during a depth-first search for strongly connected components.
    ///
    /// # Arguments
    /// * `node` - The starting node ID.
    /// * `allowed_edge_types` - A slice of edge types to filter edges by.
    /// * `visited` - A set tracking visited nodes.
    /// * `component` - A vector collecting nodes in the current component.
    fn dfs_collect_by_types(
        &self,
        node: usize,
        allowed_edge_types: &[E],
        visited: &mut HashSet<usize>,
        component: &mut Vec<usize>,
    ) {
        visited.insert(node);
        component.push(node);
        for (neighbor, _) in self.get_neighbors_by_types(node, allowed_edge_types) {
            if !visited.contains(&neighbor) {
                self.dfs_collect_by_types(neighbor, allowed_edge_types, visited, component);
            }
        }
    }

    /// Checks strong connectivity from a starting node by verifying reachability in both directions.
    ///
    /// # Arguments
    /// * `start` - The starting node ID.
    /// * `allowed_edge_types` - A slice of edge types to filter edges by.
    ///
    /// # Returns
    /// `true` if the graph is strongly connected from the start node, `false` otherwise.
    fn strong_connectivity_check_by_types(&self, start: usize, allowed_edge_types: &[E]) -> bool {
        let mut forward_visited = HashSet::new();
        self.dfs_collect_by_types(start, allowed_edge_types, &mut forward_visited, &mut vec![]);

        if forward_visited.len() != self.nodes.len() {
            return false;
        }

        let transposed = self.transpose_by_types(allowed_edge_types);
        let mut backward_visited = HashSet::new();
        transposed.dfs_collect_by_types(
            start,
            allowed_edge_types,
            &mut backward_visited,
            &mut vec![],
        );

        backward_visited.len() == self.nodes.len()
    }

    /// Retrieves neighbors of a node, filtered by edge types.
    ///
    /// # Arguments
    /// * `node` - The node ID whose neighbors to retrieve.
    /// * `allowed_edge_types` - A slice of edge types to filter edges by. If empty, all edges are included.
    ///
    /// # Returns
    /// A vector of tuples containing neighbor IDs and edge data references.
    fn get_neighbors_by_types(&self, node: usize, allowed_edge_types: &[E]) -> Vec<(usize, &E)> {
        self.edges
            .iter()
            .filter(|(_, e)| {
                e.from == node
                    && (allowed_edge_types.is_empty() || allowed_edge_types.contains(&e.data))
            })
            .map(|(_, e)| (e.to, &e.data))
            .collect()
    }

    /// Retrieves predecessors of a node, filtered by edge types.
    ///
    /// # Arguments
    /// * `node` - The node ID whose predecessors to retrieve.
    /// * `allowed_edge_types` - A slice of edge types to filter edges by. If empty, all edges are included.
    ///
    /// # Returns
    /// A vector of predecessor node IDs.
    fn get_predecessors_by_types(&self, node: usize, allowed_edge_types: &[E]) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|(_, e)| {
                e.to == node
                    && (allowed_edge_types.is_empty() || allowed_edge_types.contains(&e.data))
            })
            .map(|(_, e)| e.from)
            .collect()
    }

    /// Creates a transposed version of the graph, filtered by edge types.
    ///
    /// In the transposed graph, edge directions are reversed.
    ///
    /// # Arguments
    /// * `allowed_edge_types` - A slice of edge types to filter edges by. If empty, all edges are included.
    ///
    /// # Returns
    /// A new `HeterogeneousGraph` with reversed edges.
    fn transpose_by_types(&self, allowed_edge_types: &[E]) -> Self {
        let mut transposed = HeterogeneousGraph::new(self.directed);

        let mut node_map = std::collections::HashMap::new();
        for (id, node) in self.nodes.iter() {
            let new_id = transposed.add_node(node.data.clone());
            node_map.insert(id, new_id);
            for (key, value) in &node.attributes {
                transposed.set_node_attribute(new_id, key.clone(), value.clone());
            }
        }

        for (_edge_id, edge) in self.edges.iter() {
            if allowed_edge_types.is_empty() || allowed_edge_types.contains(&edge.data) {
                let new_from = *node_map.get(&edge.to).unwrap();
                let new_to = *node_map.get(&edge.from).unwrap();
                let new_edge_id = transposed
                    .add_edge(new_from, new_to, edge.weight, edge.data.clone())
                    .expect("Invalid edge in transpose");

                for (key, value) in &edge.attributes {
                    transposed.set_edge_attribute(new_edge_id, key.clone(), value.clone());
                }
            }
        }

        transposed
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

    #[test]
    fn test_weakly_connected_components() {
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

        let wcc = graph.find_weakly_connected_components().unwrap();
        assert_eq!(wcc.len(), 1);
        assert_eq!(wcc[0], vec![n0, n1, n2]);
    }

    #[test]
    fn test_strongly_connected_components() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(true);
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
            .add_edge(n2, n0, 1.0, TestEdge("path".to_string()))
            .unwrap();

        let scc = graph.find_strongly_connected_components().unwrap();
        assert_eq!(scc.len(), 1);
        assert_eq!(scc[0], vec![n0, n1, n2]);
    }

    #[test]
    fn test_weakly_connected_components_by_types() {
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

        let allowed_edge_types = vec![TestEdge("road".to_string())];
        let wcc = graph
            .find_weakly_connected_components_by_types(&allowed_edge_types)
            .unwrap();
        assert_eq!(wcc.len(), 2);
        assert_eq!(wcc[0], vec![n0, n1]);
        assert_eq!(wcc[1], vec![n2]);
    }

    #[test]
    fn test_strongly_connected_components_by_types() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(true);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n0, 1.0, TestEdge("path".to_string()))
            .unwrap();
        graph
            .add_edge(n0, n2, 1.0, TestEdge("path".to_string()))
            .unwrap();
        graph
            .add_edge(n2, n0, 1.0, TestEdge("path".to_string()))
            .unwrap();

        let allowed_edge_types = vec![TestEdge("road".to_string()), TestEdge("path".to_string())];
        let scc = graph
            .find_strongly_connected_components_by_types(&allowed_edge_types)
            .unwrap();

        assert_eq!(scc.len(), 1);
        assert_eq!(scc[0], vec![n0, n1, n2]);
    }

    #[test]
    fn test_is_weakly_connected() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();

        assert!(graph.is_weakly_connected().unwrap());
    }

    #[test]
    fn test_is_strongly_connected() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(true);
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
            .add_edge(n2, n0, 1.0, TestEdge("path".to_string()))
            .unwrap();

        assert!(graph.is_strongly_connected().unwrap());
    }
}
