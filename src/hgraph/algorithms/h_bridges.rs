//! Bridge detection algorithms for heterogeneous multigraphs.
//!
//! This module provides efficient algorithms to identify bridges in a heterogeneous multigraph.
//! A bridge is an edge whose removal increases the number of connected components in the graph.

#[cfg(feature = "hgraph")]
use crate::hgraph::h_graph::HeterogeneousGraph;
#[cfg(feature = "hgraph")]
use std::collections::{HashMap, HashSet};
#[cfg(feature = "hgraph")]
use std::fmt::Debug;
#[cfg(feature = "hgraph")]
use std::hash::Hash;

#[cfg(feature = "hgraph")]
#[derive(Debug)]
pub struct BridgeContext {
    visited: HashSet<usize>,
    discovery: HashMap<usize, u32>,
    lowest: HashMap<usize, u32>,
    parent: HashMap<usize, usize>,
    bridges: Vec<(usize, usize, String)>, // (from, to, edge_type)
    time: u32,
}

#[cfg(feature = "hgraph")]
impl BridgeContext {
    /// Creates a new `BridgeContext` with empty state for bridge detection.
    pub fn new() -> Self {
        Self {
            visited: HashSet::new(),
            discovery: HashMap::new(),
            lowest: HashMap::new(),
            parent: HashMap::new(),
            bridges: Vec::new(),
            time: 0,
        }
    }
}

impl Default for BridgeContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "hgraph")]
pub trait HeteroBridges {
    /// Finds all bridges in the graph, considering all edges.
    fn find_bridges(&self) -> Vec<(usize, usize, String)>;

    /// Sorts bridges by node indices and edge type for consistent output.
    fn sort_bridges(bridges: &mut Vec<(usize, usize, String)>);

    /// Internal DFS method for bridge detection.
    fn bridge_dfs(&self, node: usize, context: &mut BridgeContext);

    fn find_bridges_by_types(&self, edge_types: &[&str]) -> Vec<(usize, usize, String)>;
    fn bridge_dfs_filtered_by_types(
        &self,
        node: usize,
        context: &mut BridgeContext,
        edge_types: &[&str],
    );
}

#[cfg(feature = "hgraph")]
impl<W, N, E> HeteroBridges for HeterogeneousGraph<W, N, E>
where
    W: Copy + Default + PartialEq + Debug,
    N: Clone + Eq + Hash + Debug + crate::hgraph::h_node::NodeType,
    E: Clone + Default + Debug + crate::hgraph::h_edge::EdgeType,
{
    fn find_bridges(&self) -> Vec<(usize, usize, String)> {
        let mut context = BridgeContext::new();
        let unvisited_nodes: Vec<usize> = self
            .nodes
            .iter()
            .filter(|&(id, _)| !context.visited.contains(&id))
            .map(|(id, _)| id)
            .collect();

        unvisited_nodes
            .into_iter()
            .for_each(|id| self.bridge_dfs(id, &mut context));

        Self::sort_bridges(&mut context.bridges);
        context.bridges
    }

    fn find_bridges_by_types(&self, edge_types: &[&str]) -> Vec<(usize, usize, String)> {
        let mut context = BridgeContext::new();
        let unvisited_nodes: Vec<usize> = self
            .nodes
            .iter()
            .filter(|&(id, _)| !context.visited.contains(&id))
            .map(|(id, _)| id)
            .collect();

        unvisited_nodes
            .into_iter()
            .for_each(|id| self.bridge_dfs_filtered_by_types(id, &mut context, edge_types));

        Self::sort_bridges(&mut context.bridges);
        context.bridges
    }

    fn sort_bridges(bridges: &mut Vec<(usize, usize, String)>) {
        bridges.iter_mut().for_each(|(u, v, _)| {
            if u > v {
                std::mem::swap(u, v);
            }
        });
        bridges.sort_unstable_by(|(u1, v1, t1), (u2, v2, t2)| {
            u1.cmp(u2).then_with(|| v1.cmp(v2)).then_with(|| t1.cmp(t2))
        });
    }

    fn bridge_dfs(&self, node: usize, context: &mut BridgeContext) {
        context.time += 1;
        context.discovery.insert(node, context.time);
        context.lowest.insert(node, context.time);
        context.visited.insert(node);

        if let Some(current_node) = self.nodes.get(node) {
            current_node.neighbors.iter().for_each(|(neighbor, edges)| {
                if edges.len() > 1 {
                    // Multiple edges mean it's not a bridge
                    if !context.visited.contains(neighbor) {
                        context.parent.insert(*neighbor, node);
                        self.bridge_dfs(*neighbor, context);
                        let neighbor_low = context.lowest[neighbor];
                        context
                            .lowest
                            .entry(node)
                            .and_modify(|low| *low = (*low).min(neighbor_low));
                    } else if context.parent.get(&node) != Some(neighbor) {
                        let neighbor_disc = context.discovery[neighbor];
                        context
                            .lowest
                            .entry(node)
                            .and_modify(|low| *low = (*low).min(neighbor_disc));
                    }
                } else if let Some(&(edge_id, _)) = edges.first() {
                    // Single edge case
                    let edge = self.edges.get(edge_id).expect("Edge should exist");
                    let edge_type = edge.data.as_string();

                    if !context.visited.contains(neighbor) {
                        context.parent.insert(*neighbor, node);
                        self.bridge_dfs(*neighbor, context);
                        let neighbor_low = context.lowest[neighbor];
                        context
                            .lowest
                            .entry(node)
                            .and_modify(|low| *low = (*low).min(neighbor_low));
                        if context.lowest[neighbor] > context.discovery[&node] {
                            context.bridges.push((node, *neighbor, edge_type));
                        }
                    } else if context.parent.get(&node) != Some(neighbor) {
                        let neighbor_disc = context.discovery[neighbor];
                        context
                            .lowest
                            .entry(node)
                            .and_modify(|low| *low = (*low).min(neighbor_disc));
                    }
                }
            });
        }
    }

    fn bridge_dfs_filtered_by_types(
        &self,
        node: usize,
        context: &mut BridgeContext,
        edge_types: &[&str],
    ) {
        context.time += 1;
        context.discovery.insert(node, context.time);
        context.lowest.insert(node, context.time);
        context.visited.insert(node);

        if let Some(current_node) = self.nodes.get(node) {
            let filtered_neighbors: Vec<_> = current_node
                .neighbors
                .iter()
                .filter_map(|(neighbor, edges)| {
                    let filtered_edges: Vec<_> = edges
                        .iter()
                        .filter(|&&(edge_id, _)| {
                            let edge_type = self
                                .edges
                                .get(edge_id)
                                .expect("Edge should exist")
                                .data
                                .as_string();
                            edge_types.contains(&edge_type.as_str())
                        })
                        .collect();

                    if !filtered_edges.is_empty() {
                        Some((*neighbor, filtered_edges))
                    } else {
                        None
                    }
                })
                .collect();

            for (neighbor, edges) in filtered_neighbors {
                if !context.visited.contains(&neighbor) {
                    context.parent.insert(neighbor, node);
                    self.bridge_dfs_filtered_by_types(neighbor, context, edge_types);
                    let neighbor_low = context.lowest[&neighbor];
                    context
                        .lowest
                        .entry(node)
                        .and_modify(|low| *low = (*low).min(neighbor_low));
                    if context.lowest[&neighbor] > context.discovery[&node] {
                        // Добавляем все ребра указанных типов между node и neighbor
                        for &(edge_id, _) in &edges {
                            let edge = self.edges.get(*edge_id).expect("Edge should exist");
                            context
                                .bridges
                                .push((node, neighbor, edge.data.as_string()));
                        }
                    }
                } else if context.parent.get(&node) != Some(&neighbor) {
                    let neighbor_disc = context.discovery[&neighbor];
                    context
                        .lowest
                        .entry(node)
                        .and_modify(|low| *low = (*low).min(neighbor_disc));
                }
            }
        }
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

    #[derive(Clone, Debug, Default)]
    struct TestEdge(String);
    impl EdgeType for TestEdge {
        fn as_string(&self) -> String {
            self.0.clone()
        }
    }

    #[test]
    fn test_find_bridges_simple() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 2.0, TestEdge("bridge".to_string()))
            .unwrap();

        let bridges = graph.find_bridges();
        assert_eq!(
            bridges,
            vec![(n0, n1, "road".to_string()), (n1, n2, "bridge".to_string())]
        );
    }

    #[test]
    fn test_find_bridges_by_types() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n0, n1, 2.0, TestEdge("bridge".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 3.0, TestEdge("bridge".to_string()))
            .unwrap();
        graph
            .add_edge(n2, n0, 4.0, TestEdge("path".to_string()))
            .unwrap();

        // Ищем мосты для типов "bridge" и "road"
        let bridges = graph.find_bridges_by_types(&["bridge", "road"]);
        let expected_bridges = vec![
            (0, 1, "road".to_string()),
            (0, 1, "bridge".to_string()),
            (1, 2, "bridge".to_string()),
        ];

        let bridges_set: HashSet<_> = bridges.into_iter().collect();
        let expected_set: HashSet<_> = expected_bridges.into_iter().collect();
        assert_eq!(bridges_set, expected_set);

        // Ищем мосты для типа "path"
        let bridges = graph.find_bridges_by_types(&["path"]);
        let expected_bridges = vec![(0, 2, "path".to_string())];

        let bridges_set: HashSet<_> = bridges.into_iter().collect();
        let expected_set: HashSet<_> = expected_bridges.into_iter().collect();
        assert_eq!(bridges_set, expected_set);
    }

    #[test]
    fn test_find_bridges_cycle() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));
        let n2 = graph.add_node(TestNode("C".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n1, n2, 2.0, TestEdge("bridge".to_string()))
            .unwrap();
        graph
            .add_edge(n2, n0, 3.0, TestEdge("path".to_string()))
            .unwrap();

        let bridges = graph.find_bridges();
        assert!(bridges.is_empty());
    }

    #[test]
    fn test_find_bridges_multiple_edges() {
        let mut graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let n0 = graph.add_node(TestNode("A".to_string()));
        let n1 = graph.add_node(TestNode("B".to_string()));

        graph
            .add_edge(n0, n1, 1.0, TestEdge("road".to_string()))
            .unwrap();
        graph
            .add_edge(n0, n1, 2.0, TestEdge("bridge".to_string()))
            .unwrap();

        let bridges = graph.find_bridges();
        assert!(bridges.is_empty());
    }

    #[test]
    fn test_find_bridges_empty() {
        let graph = HeterogeneousGraph::<f64, TestNode, TestEdge>::new(false);
        let bridges = graph.find_bridges();
        assert!(bridges.is_empty());
    }
}
