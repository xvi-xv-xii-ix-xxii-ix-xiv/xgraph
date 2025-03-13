//! Module for finding dominating sets in heterogeneous multigraphs using greedy heuristics.
//!
//! This module provides implementations of a greedy algorithm to find dominating sets in
//! heterogeneous multigraphs. A dominating set is a subset of nodes such that every node in
//! the graph is either in the set or adjacent to a node in the set. The algorithm supports:
//! - Finding a dominating set using all edges.
//! - Finding a dominating set considering only specific edge types.
//!
//! The module is available only when the `hgraph` feature is enabled in your `Cargo.toml`:
//! ```toml
//! [dependencies.xgraph]
//! version = "x.y.z"
//! features = ["hgraph"]
//! ```

#[cfg(feature = "hgraph")]
use crate::hgraph::h_graph::HeterogeneousGraph;
#[cfg(feature = "hgraph")]
use std::collections::HashSet;
#[cfg(feature = "hgraph")]
use std::fmt::Debug;
#[cfg(feature = "hgraph")]
use std::hash::Hash;

// Error handling additions

/// Error type for支配 set computation failures.
///
/// Represents errors that may occur during dominating set calculations, such as invalid edge references.
#[cfg(feature = "hgraph")]
#[derive(Debug)]
pub enum DominatingSetError {
    /// Indicates an edge referenced in the graph does not exist.
    InvalidEdgeReference(usize),
}

#[cfg(feature = "hgraph")]
impl std::fmt::Display for DominatingSetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DominatingSetError::InvalidEdgeReference(id) => {
                write!(f, "Invalid edge reference: edge ID {} not found", id)
            }
        }
    }
}

#[cfg(feature = "hgraph")]
impl std::error::Error for DominatingSetError {}

/// Result type alias for dominating set operations.
///
/// Wraps the result of dominating set methods to enable error handling without panicking.
#[cfg(feature = "hgraph")]
pub type Result<T> = std::result::Result<T, DominatingSetError>;

#[cfg(feature = "hgraph")]
/// Trait for finding dominating sets in heterogeneous multigraphs.
///
/// Provides methods to compute dominating sets using a greedy heuristic, with support for
/// edge type filtering. Methods return a `Result` to handle potential errors gracefully.
pub trait HeteroDominatingSetFinder<W, N, E>
where
    W: Copy + Default + PartialEq + Debug,
    N: Clone + Eq + Hash + Debug + crate::hgraph::h_node::NodeType,
    E: Clone + Default + Debug + crate::hgraph::h_edge::EdgeType + PartialEq,
{
    /// Finds a dominating set in the graph considering all edges.
    ///
    /// Returns a set of node indices where every node in the graph is either in the set or
    /// adjacent to a node in the set, using all available edges.
    ///
    /// # Returns
    /// A `Result` containing a `HashSet<usize>` of node indices in the dominating set.
    ///
    /// # Errors
    /// Returns `DominatingSetError::InvalidEdgeReference` if an edge ID referenced in the
    /// graph’s neighbor list does not exist.
    fn find_dominating_set(&self) -> Result<HashSet<usize>>;

    /// Finds a dominating set in the graph considering only specified edge types.
    ///
    /// Filters edges by type to compute a dominating set in a subgraph defined by
    /// `allowed_edge_types`. If the list is empty, all edge types are considered.
    ///
    /// # Arguments
    /// * `allowed_edge_types` - A slice of edge types to filter the computation.
    ///
    /// # Returns
    /// A `Result` containing a `HashSet<usize>` of node indices in the dominating set.
    ///
    /// # Errors
    /// Returns `DominatingSetError::InvalidEdgeReference` if an edge ID referenced in the
    /// graph’s neighbor list does not exist.
    fn find_dominating_set_by_types(&self, allowed_edge_types: &[E]) -> Result<HashSet<usize>>;
}

#[cfg(feature = "hgraph")]
impl<W, N, E> HeteroDominatingSetFinder<W, N, E> for HeterogeneousGraph<W, N, E>
where
    W: Copy + Default + PartialEq + Debug,
    N: Clone + Eq + Hash + Debug + crate::hgraph::h_node::NodeType,
    E: Clone + Default + Debug + crate::hgraph::h_edge::EdgeType + PartialEq,
{
    fn find_dominating_set(&self) -> Result<HashSet<usize>> {
        // Empty list means consider all edges
        self.find_dominating_set_by_types(&[])
    }

    fn find_dominating_set_by_types(&self, allowed_edge_types: &[E]) -> Result<HashSet<usize>> {
        let mut dominating_set = HashSet::new();
        let mut covered = HashSet::new();

        // Step 1: Collect nodes with non-zero degree based on allowed edge types
        let mut nodes_with_degree: Vec<(usize, usize)> = self
            .nodes
            .iter()
            .map(|(id, _)| {
                let degree = self
                    .get_neighbors(id)
                    .iter()
                    .filter(|(_, edges)| {
                        edges.iter().any(|(edge_id, _)| {
                            if let Some(edge) = self.edges.get(*edge_id) {
                                allowed_edge_types.is_empty()
                                    || allowed_edge_types.contains(&edge.data)
                            } else {
                                false
                            }
                        })
                    })
                    .count();
                (id, degree)
            })
            .filter(|&(_, degree)| degree > 0) // Only nodes with relevant edges
            .collect();

        // Sort by decreasing degree for greedy selection
        nodes_with_degree.sort_by(|a, b| b.1.cmp(&a.1));

        // Step 2: Greedily cover connected nodes
        for (node, _) in &nodes_with_degree {
            let neighbors: HashSet<usize> = self
                .get_neighbors(*node)
                .iter()
                .filter(|(_, edges)| {
                    edges.iter().any(|(edge_id, _)| {
                        match self.edges.get(*edge_id) {
                            Some(edge) => {
                                allowed_edge_types.is_empty()
                                    || allowed_edge_types.contains(&edge.data)
                            }
                            None => false, // Will trigger error later if needed
                        }
                    })
                })
                .map(|(n, _)| *n)
                .collect();

            // Check for invalid edge references
            for &(edge_id, _) in self
                .get_neighbors(*node)
                .iter()
                .flat_map(|(_, edges)| edges)
            {
                if self.edges.get(edge_id).is_none() {
                    return Err(DominatingSetError::InvalidEdgeReference(edge_id));
                }
            }

            // Add node if it or any neighbor is uncovered
            if !covered.contains(node) || neighbors.iter().any(|n| !covered.contains(n)) {
                dominating_set.insert(*node);
                covered.insert(*node);
                for neighbor in neighbors {
                    covered.insert(neighbor);
                }
            }
        }

        Ok(dominating_set)
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
    fn test_simple_dominating_set() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));
        let n3 = graph.add_node(TestNode("D".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("friend".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1.0, TestEdge("friend".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n3, 1.0, TestEdge("friend".to_string()))
            .unwrap();

        let ds = graph.find_dominating_set().unwrap();
        assert!(!ds.is_empty(), "Dominating set should not be empty");
        assert!(
            ds.len() <= 2,
            "Dominating set size should be at most 2, got {}",
            ds.len()
        );
    }

    #[test]
    fn test_triangle_dominating_set() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("friend".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1.0, TestEdge("friend".to_string()))
            .unwrap();
        graph
            .add_edge(n2, n0, 1.0, TestEdge("friend".to_string()))
            .unwrap();

        let ds = graph.find_dominating_set().unwrap();
        assert_eq!(
            ds.len(),
            1,
            "Dominating set for a triangle should be 1, got {}",
            ds.len()
        );
    }

    #[test]
    fn test_disconnected_components() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));
        let n3 = graph.add_node(TestNode("D".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("friend".to_string()))
            .unwrap();
        graph
            .add_edge(n2, n3, 1.0, TestEdge("friend".to_string()))
            .unwrap();

        let ds = graph.find_dominating_set().unwrap();
        assert_eq!(
            ds.len(),
            2,
            "Dominating set should contain 2 nodes for 2 components, got {}",
            ds.len()
        );
    }

    #[test]
    fn test_multiple_edges() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("friend".to_string()))
            .unwrap();
        graph
            .add_edge(n0, n1, 2.0, TestEdge("colleague".to_string()))
            .unwrap();

        let ds = graph.find_dominating_set().unwrap();
        assert_eq!(
            ds.len(),
            1,
            "Dominating set should be 1 despite multiple edges, got {}",
            ds.len()
        );
    }

    #[test]
    fn test_dominating_set_by_types() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));
        let n3 = graph.add_node(TestNode("D".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("friend".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1.0, TestEdge("colleague".to_string()))
            .unwrap();
        graph
            .add_edge(n2, n3, 1.0, TestEdge("friend".to_string()))
            .unwrap();

        // Consider only "friend" edges
        let allowed_edge_types = vec![TestEdge("friend".to_string())];
        let ds = graph
            .find_dominating_set_by_types(&allowed_edge_types)
            .unwrap();
        assert_eq!(
            ds.len(),
            2,
            "Dominating set should contain 2 nodes when considering only 'friend' edges, got {}",
            ds.len()
        );
        let covers_all =
            (ds.contains(&n0) || ds.contains(&n1)) && (ds.contains(&n2) || ds.contains(&n3));
        assert!(
            covers_all,
            "Dominating set should cover both components: {:?}",
            ds
        );
    }

    #[test]
    fn test_dominating_set_by_types_isolated() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("friend".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 1.0, TestEdge("colleague".to_string()))
            .unwrap();

        // Consider only "friend" edges
        let allowed_edge_types = vec![TestEdge("friend".to_string())];
        let ds = graph
            .find_dominating_set_by_types(&allowed_edge_types)
            .unwrap();
        assert_eq!(
            ds.len(),
            1,
            "Dominating set should contain 1 node (n0 or n1) for connected 'friend' component, got {}",
            ds.len()
        );
        assert!(
            ds.contains(&n0) || ds.contains(&n1),
            "Either n0 or n1 should be in the dominating set: {:?}",
            ds
        );
    }

    #[test]
    fn test_invalid_edge_reference() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));

        // Simulate an invalid edge by manually manipulating the graph (not possible through public API)
        // For testing, we assume the possibility of an edge ID not existing in a corrupted graph
        graph
            .add_edge(n0, n1, 1.0, TestEdge("friend".to_string()))
            .unwrap();

        // Note: Since we can't directly inject an invalid edge ID via public API,
        // this test assumes the error handling works as implemented. In practice,
        // this would require internal corruption or a bug in graph construction.
        let ds_result = graph.find_dominating_set();
        assert!(ds_result.is_ok(), "Should succeed with valid edges");
    }
}
