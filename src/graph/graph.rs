//! A comprehensive graph implementation supporting directed/undirected graphs,
//! generic node/edge types, weights, and attributes.
//!
//! This module provides a flexible and efficient graph data structure with support for:
//! - Directed and undirected graphs
//! - Generic node and edge data types
//! - Weighted edges
//! - Node and edge attributes
//!
//! # Examples
//!
//! Basic usage:
//!
//! ```rust
//! use xgraph::graph::Graph;
//!
//! let mut graph = Graph::<i32, &str, &str>::new(false);
//! let london = graph.add_node("London");
//! let paris = graph.add_node("Paris");
//! graph.add_edge(london, paris, 343, "Eurostar").unwrap();
//! ```

use crate::{Edge, Node};
use slab::Slab;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

/// Unique identifier for nodes in the graph.
pub type NodeId = usize;

/// Error type for graph operations.
#[derive(Debug)]
pub enum GraphError {
    /// Indicates that a node with the given ID does not exist.
    InvalidNodeId(NodeId),
    /// Indicates that an edge between the specified nodes does not exist.
    EdgeNotFound { from: NodeId, to: NodeId },
    /// Indicates an error during adjacency matrix creation or parsing.
    MatrixError(String),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::InvalidNodeId(id) => write!(f, "Invalid node ID: {}", id),
            GraphError::EdgeNotFound { from, to } => {
                write!(f, "Edge not found between nodes {} and {}", from, to)
            }
            GraphError::MatrixError(msg) => write!(f, "Matrix error: {}", msg),
        }
    }
}

impl std::error::Error for GraphError {}

/// A generic graph structure with configurable node/edge types and weights.
///
/// # Type Parameters
/// - `W`: Edge weight type (must implement `Copy + PartialEq`)
/// - `N`: Node data type (must implement `Clone + Eq + Hash + Debug`)
/// - `E`: Edge data type (must implement `Clone + Debug`)
///
/// # Fields
/// - `nodes`: Storage for nodes using slab allocation
/// - `edges`: Storage for edges using slab allocation
/// - `directed`: Directionality flag (true for directed graphs)
///
/// # Examples
///
/// Creating a directed graph:
/// ```rust
/// use xgraph::graph::Graph;
/// let mut directed_graph = Graph::<f64, String, ()>::new(true);
/// ```
#[derive(Debug, Clone)]
pub struct Graph<W, N, E>
where
    W: Copy + PartialEq,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Debug,
{
    pub nodes: Slab<Node<W, N>>,
    pub edges: Slab<Edge<W, E>>,
    pub directed: bool,
}

impl<W, N, E> Graph<W, N, E>
where
    W: Copy + Default + PartialEq,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    /// Creates a new empty graph with specified directionality.
    ///
    /// # Arguments
    /// - `directed`: Set to `true` for a directed graph, `false` for an undirected graph.
    ///
    /// # Returns
    /// A new `Graph` instance.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::Graph;
    /// let undirected_graph = Graph::<i32, &str, ()>::new(false);
    /// ```
    pub fn new(directed: bool) -> Self {
        Self {
            nodes: Slab::with_capacity(1024),
            edges: Slab::with_capacity(4096),
            directed,
        }
    }

    /// Adds a node to the graph with associated data.
    ///
    /// # Arguments
    /// - `data`: The data to store in the node.
    ///
    /// # Returns
    /// The `NodeId` of the newly created node.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::Graph;
    /// let mut graph = Graph::<u32, &str, ()>::new(false);
    /// let node_id = graph.add_node("Node Data");
    /// ```
    pub fn add_node(&mut self, data: N) -> NodeId {
        self.nodes.insert(Node {
            data,
            neighbors: Vec::new(),
            attributes: HashMap::new(),
        })
    }

    /// Removes a node and all its connected edges from the graph.
    ///
    /// # Arguments
    /// - `node`: The `NodeId` of the node to remove.
    ///
    /// # Returns
    /// - `Ok(true)` if the node existed and was removed.
    /// - `Ok(false)` if the node did not exist.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::Graph;
    /// let mut graph = Graph::<u32, &str, ()>::new(false);
    /// let node = graph.add_node("Data");
    /// assert!(graph.remove_node(node).unwrap());
    /// ```
    pub fn remove_node(&mut self, node: NodeId) -> Result<bool, GraphError> {
        if !self.nodes.contains(node) {
            return Ok(false);
        }
        let edges_to_remove: Vec<_> = self
            .edges
            .iter()
            .filter(|(_, e)| e.from == node || e.to == node)
            .map(|(id, _)| id)
            .collect();

        for edge_id in edges_to_remove {
            self.edges.remove(edge_id);
        }
        self.nodes.remove(node);
        Ok(true)
    }

    /// Adds an edge between two nodes with a weight and data.
    ///
    /// # Arguments
    /// - `from`: Source node ID.
    /// - `to`: Target node ID.
    /// - `weight`: Edge weight.
    /// - `edge_data`: Edge metadata.
    ///
    /// # Returns
    /// - `Ok(())` on success.
    /// - `Err(GraphError::InvalidNodeId)` if either node does not exist.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::Graph;
    /// let mut graph = Graph::<u32, &str, &str>::new(false);
    /// let a = graph.add_node("A");
    /// let b = graph.add_node("B");
    /// graph.add_edge(a, b, 1, "connection").unwrap();
    /// ```
    pub fn add_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        weight: W,
        edge_data: E,
    ) -> Result<(), GraphError> {
        let (from, to) = self.normalize_edge(from, to);

        if !self.nodes.contains(from) {
            return Err(GraphError::InvalidNodeId(from));
        }
        if !self.nodes.contains(to) {
            return Err(GraphError::InvalidNodeId(to));
        }

        let edge = Edge {
            from,
            to,
            weight,
            data: edge_data,
            attributes: HashMap::new(),
        };

        let _edge_id = self.edges.insert(edge);

        self.nodes[from].neighbors.push((to, weight));
        if !self.directed {
            self.nodes[to].neighbors.push((from, weight));
        }

        Ok(())
    }

    /// Adds multiple nodes to the graph in a batch.
    ///
    /// # Arguments
    /// - `data_iter`: An iterator yielding node data to add.
    ///
    /// # Returns
    /// A vector of `NodeId`s for the added nodes.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::Graph;
    /// let mut graph = Graph::<u32, &str, &str>::new(false);
    /// let ids = graph.add_nodes_batch(vec!["A", "B", "C"].into_iter());
    /// assert_eq!(ids.len(), 3);
    /// ```
    pub fn add_nodes_batch(&mut self, data_iter: impl Iterator<Item = N>) -> Vec<NodeId> {
        data_iter.map(|data| self.add_node(data)).collect()
    }

    /// Adds multiple edges to the graph in a batch.
    ///
    /// # Arguments
    /// - `edges`: A vector of tuples `(from, to, weight, edge_data)`.
    ///
    /// # Returns
    /// - `Ok(())` if all edges were added successfully.
    /// - `Err(GraphError)` with details of the first failure.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::Graph;
    /// let mut graph = Graph::<u32, &str, &str>::new(false);
    /// let a = graph.add_node("A");
    /// let b = graph.add_node("B");
    /// let edges = vec![(a, b, 1, "AB")];
    /// assert!(graph.add_edges_batch(edges).is_ok());
    /// ```
    pub fn add_edges_batch(
        &mut self,
        edges: Vec<(NodeId, NodeId, W, E)>,
    ) -> Result<(), GraphError> {
        for (from, to, weight, data) in edges {
            self.add_edge(from, to, weight, data)?;
        }
        Ok(())
    }

    /// Removes an edge between two nodes.
    ///
    /// # Arguments
    /// - `from`: Source node ID.
    /// - `to`: Target node ID.
    ///
    /// # Returns
    /// - `Ok(())` if the edge was removed.
    /// - `Err(GraphError::EdgeNotFound)` if no edge exists between the nodes.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::Graph;
    /// let mut graph = Graph::<u32, &str, &str>::new(false);
    /// let a = graph.add_node("A");
    /// let b = graph.add_node("B");
    /// graph.add_edge(a, b, 1, "").unwrap();
    /// assert!(graph.remove_edge(a, b).is_ok());
    /// ```
    pub fn remove_edge(&mut self, from: NodeId, to: NodeId) -> Result<(), GraphError> {
        let directed = self.directed;
        let edge_ids: Vec<_> = self
            .edges
            .iter()
            .filter(|(_, e)| Self::edge_matches(e, from, to, directed))
            .map(|(id, _)| id)
            .collect();

        if edge_ids.is_empty() {
            return Err(GraphError::EdgeNotFound { from, to });
        }

        for edge_id in edge_ids {
            self.edges.remove(edge_id);
        }

        self.nodes[from].neighbors.retain(|&(n, _)| n != to);
        if !self.directed {
            self.nodes[to].neighbors.retain(|&(n, _)| n != from);
        }

        Ok(())
    }

    /// Creates a graph from an adjacency matrix.
    ///
    /// # Arguments
    /// - `matrix`: A 2D vector representing edge weights.
    /// - `directed`: Graph directionality.
    /// - `default_node`: Default node data for all nodes.
    /// - `default_edge`: Default edge data for all edges.
    ///
    /// # Returns
    /// - `Ok(Graph)` on success.
    /// - `Err(GraphError::MatrixError)` if the matrix is invalid.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::Graph;
    /// let matrix = vec![vec![0, 1], vec![1, 0]];
    /// let graph = Graph::from_adjacency_matrix(&matrix, false, "Node", "").unwrap();
    /// ```
    pub fn from_adjacency_matrix(
        matrix: &[Vec<W>],
        directed: bool,
        default_node: N,
        default_edge: E,
    ) -> Result<Self, GraphError>
    where
        W: PartialEq + Default,
        N: Clone,
        E: Clone,
    {
        if matrix.is_empty() || matrix.iter().any(|row| row.len() != matrix.len()) {
            return Err(GraphError::MatrixError(
                "Matrix must be non-empty and square".to_string(),
            ));
        }

        let mut graph = Graph::new(directed);
        let n = matrix.len();

        for _ in 0..n {
            graph.add_node(default_node.clone());
        }

        let edges: Vec<_> = (0..n)
            .flat_map(|i| {
                let range = if directed { 0..n } else { i..n };
                let edge_data = default_edge.clone();
                range.filter_map(move |j| {
                    if matrix[i][j] != W::default() {
                        Some((i, j, matrix[i][j], edge_data.clone()))
                    } else {
                        None
                    }
                })
            })
            .collect();

        graph.add_edges_batch(edges)?;
        Ok(graph)
    }

    /// Converts the graph to an adjacency matrix.
    ///
    /// # Returns
    /// A 2D vector representing the adjacency matrix.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::Graph;
    /// let mut graph = Graph::<i32, &str, &str>::new(false);
    /// let a = graph.add_node("A");
    /// let b = graph.add_node("B");
    /// graph.add_edge(a, b, 1, "").unwrap();
    /// let matrix = graph.to_adjacency_matrix();
    /// assert_eq!(matrix[a][b], 1);
    /// ```
    pub fn to_adjacency_matrix(&self) -> Vec<Vec<W>>
    where
        W: Default + Copy,
    {
        let size = self.nodes.len();
        let mut matrix = vec![vec![W::default(); size]; size];

        for (_, edge) in self.edges.iter() {
            matrix[edge.from][edge.to] = edge.weight;
            if !self.directed {
                matrix[edge.to][edge.from] = edge.weight;
            }
        }
        matrix
    }

    /// Sets an attribute on a node.
    ///
    /// # Arguments
    /// - `node`: Node ID.
    /// - `key`: Attribute key.
    /// - `value`: Attribute value.
    ///
    /// # Returns
    /// - `Ok(true)` if the attribute was set.
    /// - `Err(GraphError::InvalidNodeId)` if the node does not exist.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::Graph;
    /// let mut graph = Graph::<u32, &str, ()>::new(false);
    /// let node = graph.add_node("Data");
    /// graph.set_node_attribute(node, "color".into(), "red".into()).unwrap();
    /// ```
    pub fn set_node_attribute(
        &mut self,
        node: NodeId,
        key: String,
        value: String,
    ) -> Result<bool, GraphError> {
        if let Some(node) = self.nodes.get_mut(node) {
            node.attributes.insert(key, value);
            Ok(true)
        } else {
            Err(GraphError::InvalidNodeId(node))
        }
    }

    /// Sets an attribute on an edge.
    ///
    /// # Arguments
    /// - `from`: Source node ID.
    /// - `to`: Target node ID.
    /// - `key`: Attribute key.
    /// - `value`: Attribute value.
    ///
    /// # Returns
    /// - `Ok(true)` if the attribute was set.
    /// - `Err(GraphError::EdgeNotFound)` if no edge exists.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::Graph;
    /// let mut graph = Graph::<u32, &str, &str>::new(false);
    /// let a = graph.add_node("A");
    /// let b = graph.add_node("B");
    /// graph.add_edge(a, b, 1, "").unwrap();
    /// graph.set_edge_attribute(a, b, "type".into(), "road".into()).unwrap();
    /// ```
    pub fn set_edge_attribute(
        &mut self,
        from: NodeId,
        to: NodeId,
        key: String,
        value: String,
    ) -> Result<bool, GraphError> {
        let mut found = false;
        let directed = self.directed;
        for (_, edge) in self.edges.iter_mut() {
            if Self::edge_matches(edge, from, to, directed) {
                edge.attributes.insert(key.clone(), value.clone());
                found = true;
            }
        }
        if found {
            Ok(true)
        } else {
            Err(GraphError::EdgeNotFound { from, to })
        }
    }

    /// Gets a node attribute value.
    ///
    /// # Arguments
    /// - `node`: Node ID.
    /// - `key`: Attribute key.
    ///
    /// # Returns
    /// - `Some(&String)` if the attribute exists.
    /// - `None` if the node or attribute does not exist.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::Graph;
    /// let mut graph = Graph::<u32, &str, ()>::new(false);
    /// let node = graph.add_node("Data");
    /// graph.set_node_attribute(node, "color".into(), "blue".into()).unwrap();
    /// assert_eq!(graph.get_node_attribute(node, "color"), Some(&"blue".into()));
    /// ```
    pub fn get_node_attribute(&self, node: NodeId, key: &str) -> Option<&String> {
        self.nodes.get(node).and_then(|n| n.attributes.get(key))
    }

    /// Gets an edge attribute value.
    ///
    /// # Arguments
    /// - `from`: Source node ID.
    /// - `to`: Target node ID.
    /// - `key`: Attribute key.
    ///
    /// # Returns
    /// - `Some(&String)` if the attribute exists.
    /// - `None` if the edge or attribute does not exist.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::Graph;
    /// let mut graph = Graph::<u32, &str, &str>::new(false);
    /// let a = graph.add_node("A");
    /// let b = graph.add_node("B");
    /// graph.add_edge(a, b, 1, "").unwrap();
    /// graph.set_edge_attribute(a, b, "type".into(), "rail".into()).unwrap();
    /// assert_eq!(graph.get_edge_attribute(a, b, "type"), Some(&"rail".into()));
    /// ```
    pub fn get_edge_attribute(&self, from: NodeId, to: NodeId, key: &str) -> Option<&String> {
        self.edges
            .iter()
            .find(|(_, e)| Self::edge_matches(e, from, to, self.directed))
            .and_then(|(_, e)| e.attributes.get(key))
    }

    /// Gets all attributes for an edge.
    ///
    /// # Arguments
    /// - `from`: Source node ID.
    /// - `to`: Target node ID.
    ///
    /// # Returns
    /// - `Some(&HashMap<String, String>)` if the edge exists.
    /// - `None` if the edge does not exist.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::Graph;
    /// let mut graph = Graph::<u32, &str, &str>::new(false);
    /// let a = graph.add_node("A");
    /// let b = graph.add_node("B");
    /// graph.add_edge(a, b, 1, "").unwrap();
    /// graph.set_edge_attribute(a, b, "type".into(), "air".into()).unwrap();
    /// let attrs = graph.get_all_edge_attributes(a, b).unwrap();
    /// assert_eq!(attrs.get("type"), Some(&"air".into()));
    /// ```
    pub fn get_all_edge_attributes(
        &self,
        from: NodeId,
        to: NodeId,
    ) -> Option<&HashMap<String, String>> {
        self.edges
            .iter()
            .find(|(_, e)| Self::edge_matches(e, from, to, self.directed))
            .map(|(_, e)| &e.attributes)
    }

    /// Gets all edges in the graph.
    ///
    /// # Returns
    /// A vector of tuples `(from, to, weight, edge_data)` representing all edges.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::Graph;
    /// let mut graph = Graph::<u32, &str, &str>::new(false);
    /// let a = graph.add_node("A");
    /// let b = graph.add_node("B");
    /// graph.add_edge(a, b, 1, "AB").unwrap();
    /// let edges = graph.get_all_edges();
    /// assert_eq!(edges.len(), 1);
    /// ```
    pub fn get_all_edges(&self) -> Vec<(NodeId, NodeId, W, E)> {
        self.edges
            .iter()
            .map(|(_, e)| (e.from, e.to, e.weight, e.data.clone()))
            .collect()
    }

    /// Gets all neighbors of a node.
    ///
    /// # Arguments
    /// - `node`: Node ID.
    ///
    /// # Returns
    /// A vector of tuples `(neighbor_id, weight)` for all neighbors.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::Graph;
    /// let mut graph = Graph::<u32, &str, &str>::new(false);
    /// let a = graph.add_node("A");
    /// let b = graph.add_node("B");
    /// graph.add_edge(a, b, 1, "").unwrap();
    /// let neighbors = graph.get_neighbors(a);
    /// assert_eq!(neighbors.len(), 1);
    /// ```
    pub fn get_neighbors(&self, node: NodeId) -> Vec<(NodeId, W)> {
        self.nodes
            .get(node)
            .map(|n| n.neighbors.iter().map(|&(id, w)| (id, w)).collect())
            .unwrap_or_default()
    }

    /// Validates the consistency of the graph.
    ///
    /// Checks that all edges reference valid nodes and all neighbor lists are consistent.
    ///
    /// # Returns
    /// - `true` if the graph is valid.
    /// - `false` if inconsistencies are found.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::Graph;
    /// let mut graph = Graph::<u32, &str, &str>::new(false);
    /// let a = graph.add_node("A");
    /// let b = graph.add_node("B");
    /// graph.add_edge(a, b, 1, "").unwrap();
    /// assert!(graph.validate_graph());
    /// ```
    pub fn validate_graph(&self) -> bool {
        let edges_valid = self
            .edges
            .iter()
            .all(|(_, e)| self.nodes.contains(e.from) && self.nodes.contains(e.to));

        let neighbors_valid = self
            .nodes
            .iter()
            .all(|(_, n)| n.neighbors.iter().all(|&(id, _)| self.nodes.contains(id)));

        edges_valid && neighbors_valid
    }

    /// Returns an iterator over all nodes in the graph.
    ///
    /// # Returns
    /// An iterator yielding tuples `(NodeId, &N)` for each node.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::Graph;
    /// let mut graph = Graph::<u32, &str, ()>::new(false);
    /// graph.add_node("A");
    /// graph.add_node("B");
    /// for (id, data) in graph.all_nodes() {
    ///     println!("Node {}: {:?}", id, data);
    /// }
    /// ```
    pub fn all_nodes(&self) -> impl Iterator<Item = (NodeId, &N)> + '_ {
        self.nodes.iter().map(|(id, node)| (id, &node.data))
    }

    /// Normalizes edge direction for undirected graphs.
    fn normalize_edge(&self, mut from: NodeId, mut to: NodeId) -> (NodeId, NodeId) {
        if !self.directed && from > to {
            std::mem::swap(&mut from, &mut to);
        }
        (from, to)
    }

    /// Checks if an edge matches the given from/to nodes considering directionality.
    fn edge_matches(edge: &Edge<W, E>, from: NodeId, to: NodeId, directed: bool) -> bool {
        if directed {
            edge.from == from && edge.to == to
        } else {
            (edge.from == from && edge.to == to) || (edge.from == to && edge.to == from)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed_types() {
        let mut graph = Graph::<u32, String, &str>::new(false);
        let paris = graph.add_node("Paris".into());
        let london = graph.add_node("London".into());
        assert!(graph.add_edge(paris, london, 343, "Eurostar").is_ok());
        assert_eq!(graph.get_all_edges()[0].3, "Eurostar");
    }

    #[test]
    fn test_directed_graph() {
        let mut graph = Graph::<f64, i32, String>::new(true);
        let a = graph.add_node(1);
        let b = graph.add_node(2);
        graph.add_edge(a, b, 1.5, "AB".into()).unwrap();
        graph.add_edge(b, a, 2.5, "BA".into()).unwrap();
        assert_eq!(graph.get_all_edges().len(), 2);
    }

    #[test]
    fn test_attributes() {
        let mut graph = Graph::<i32, &str, ()>::new(false);
        let n1 = graph.add_node("Node1");
        let n2 = graph.add_node("Node2");
        graph
            .set_node_attribute(n1, "color".into(), "red".into())
            .unwrap();
        graph.add_edge(n1, n2, 10, ()).unwrap();
        graph
            .set_edge_attribute(n1, n2, "type".into(), "road".into())
            .unwrap();
        assert_eq!(graph.get_node_attribute(n1, "color"), Some(&"red".into()));
        assert_eq!(
            graph.get_edge_attribute(n1, n2, "type"),
            Some(&"road".into())
        );
    }

    #[test]
    fn test_empty_graph() {
        let graph: Graph<u32, (), ()> = Graph::new(false);
        assert!(graph.nodes.is_empty());
        assert!(graph.edges.is_empty());
    }

    #[test]
    fn test_complete_graph() {
        let mut graph = Graph::<u32, i32, ()>::new(false);
        let nodes: Vec<_> = (0..5).map(|i| graph.add_node(i)).collect();
        for &a in &nodes {
            for &b in &nodes {
                if a != b {
                    graph.add_edge(a, b, 1, ()).unwrap();
                }
            }
        }
        assert_eq!(graph.edges.len(), 20);
    }

    #[test]
    fn test_disconnected_graph() {
        let mut graph = Graph::<u32, &str, ()>::new(false);
        graph.add_node("A");
        graph.add_node("B");
        assert_eq!(graph.get_all_edges().len(), 0);
    }

    #[test]
    fn test_graph_validation() {
        let mut graph = Graph::<u32, &str, ()>::new(false);
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        graph.add_edge(a, b, 1, ()).unwrap();
        assert!(graph.validate_graph());
    }

    #[test]
    fn test_varied_node_edge_types() {
        let mut graph = Graph::<f32, &str, char>::new(false);
        let a = graph.add_node("Alpha");
        let b = graph.add_node("Beta");
        let c = graph.add_node("Gamma");
        graph.add_edge(a, b, 2.5, 'X').unwrap();
        graph.add_edge(b, c, 3.1, 'Y').unwrap();
        assert_eq!(graph.get_all_edges().len(), 2);
    }

    #[test]
    fn test_mixed_weight_types() {
        let mut graph = Graph::<i64, u32, &str>::new(true);
        let n1 = graph.add_node(100);
        let n2 = graph.add_node(200);
        graph.add_edge(n1, n2, 50, "Highway").unwrap();
        assert_eq!(graph.get_neighbors(n1).len(), 1);
    }
}

// Assuming these are defined in a separate module (graph.rs)
pub mod graph {
    use super::*;
    /// Represents a node in the graph.
    #[derive(Debug, Clone)]
    pub struct Node<W, N> {
        pub data: N,
        pub neighbors: Vec<(NodeId, W)>,
        pub attributes: HashMap<String, String>,
    }

    /// Represents an edge in the graph.
    #[derive(Debug, Clone)]
    pub struct Edge<W, E> {
        pub from: NodeId,
        pub to: NodeId,
        pub weight: W,
        pub data: E,
        pub attributes: HashMap<String, String>,
    }
}
