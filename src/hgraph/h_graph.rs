use crate::hgraph::{h_edge::HEdge, h_node::HNode};
use slab::Slab;
use std::collections::HashMap;
use std::fmt::Debug;

/// Type alias for node identifiers in the graph.
pub type NodeId = usize;

/// Type alias for edge identifiers in the graph.
pub type EdgeId = usize;

/// A heterogeneous multigraph implementation based on traits.
///
/// This structure represents a graph where nodes and edges can have custom data types,
/// supporting multiple edges between the same pair of nodes (multigraph behavior).
/// It uses `Slab` for efficient storage and retrieval of nodes and edges.
#[derive(Debug)]
pub struct HeterogeneousGraph<W, N, E>
where
    W: Copy + PartialEq + Debug,
    N: crate::hgraph::h_node::NodeType,
    E: crate::hgraph::h_edge::EdgeType,
{
    /// Storage for the graph's nodes.
    pub nodes: Slab<HNode<W, N>>,
    /// Storage for the graph's edges.
    pub edges: Slab<HEdge<W, E>>,
    /// Indicates whether the graph is directed.
    pub directed: bool,
}

impl<W, N, E> HeterogeneousGraph<W, N, E>
where
    W: Copy + Default + PartialEq + Debug,
    N: crate::hgraph::h_node::NodeType,
    E: crate::hgraph::h_edge::EdgeType + Default,
{
    /// Creates a new heterogeneous multigraph.
    ///
    /// # Arguments
    /// * `directed` - A boolean indicating whether the graph is directed.
    ///
    /// # Returns
    /// A new instance of `HeterogeneousGraph`.
    pub fn new(directed: bool) -> Self {
        Self {
            nodes: Slab::with_capacity(1024),
            edges: Slab::with_capacity(4096),
            directed,
        }
    }

    /// Adds a node with the specified data.
    ///
    /// # Arguments
    /// * `data` - The data associated with the node.
    ///
    /// # Returns
    /// The `NodeId` of the newly added node.
    pub fn add_node(&mut self, data: N) -> NodeId {
        self.nodes.insert(HNode::new(data))
    }

    /// Adds an edge with the specified data and returns its `EdgeId`.
    ///
    /// # Arguments
    /// * `from` - The `NodeId` of the source node.
    /// * `to` - The `NodeId` of the target node.
    /// * `weight` - The weight associated with the edge.
    /// * `edge_data` - The data associated with the edge.
    ///
    /// # Returns
    /// A `Result` containing the `EdgeId` if successful, or an error message if the node IDs are invalid.
    pub fn add_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        weight: W,
        edge_data: E,
    ) -> Result<EdgeId, String> {
        let (from, to) = self.normalize_edge(from, to);

        if !self.nodes.contains(from) || !self.nodes.contains(to) {
            return Err(format!("Invalid node IDs: {} or {}", from, to));
        }

        let edge_id = self.edges.vacant_key(); // Get the next available key
        let edge = HEdge::new(edge_id, from, to, weight, edge_data);
        let inserted_id = self.edges.insert(edge);
        assert_eq!(edge_id, inserted_id); // Ensure the IDs match

        self.nodes[from].add_neighbor(to, edge_id, weight);
        if !self.directed {
            self.nodes[to].add_neighbor(from, edge_id, weight);
        }

        Ok(edge_id)
    }

    /// Removes an edge by its `EdgeId`.
    ///
    /// # Arguments
    /// * `edge_id` - The `EdgeId` of the edge to remove.
    ///
    /// # Returns
    /// A boolean indicating whether the edge was successfully removed.
    pub fn remove_edge(&mut self, edge_id: EdgeId) -> bool {
        if let Some(edge) = self.edges.try_remove(edge_id) {
            self.nodes[edge.from].remove_edge(edge.to, edge_id);
            if !self.directed {
                self.nodes[edge.to].remove_edge(edge.from, edge_id);
            }
            true
        } else {
            false
        }
    }

    /// Retrieves the data of a node.
    ///
    /// # Arguments
    /// * `node` - The `NodeId` of the node.
    ///
    /// # Returns
    /// An `Option` containing a reference to the node's data if the node exists, or `None` otherwise.
    pub fn get_node_data(&self, node: NodeId) -> Option<&N> {
        self.nodes.get(node).map(|n| &n.data)
    }

    /// Retrieves the data of an edge by its `EdgeId`.
    ///
    /// # Arguments
    /// * `edge_id` - The `EdgeId` of the edge.
    ///
    /// # Returns
    /// An `Option` containing a reference to the edge's data if the edge exists, or `None` otherwise.
    pub fn get_edge_data_by_id(&self, edge_id: EdgeId) -> Option<&E> {
        self.edges.get(edge_id).map(|e| &e.data)
    }

    /// Retrieves all edges between two nodes.
    ///
    /// # Arguments
    /// * `from` - The `NodeId` of the source node.
    /// * `to` - The `NodeId` of the target node.
    ///
    /// # Returns
    /// A vector of tuples containing the `EdgeId`, a reference to the edge's data, and the edge's weight.
    pub fn get_edges_between(&self, from: NodeId, to: NodeId) -> Vec<(EdgeId, &E, W)> {
        self.edges
            .iter()
            .filter(|(_, e)| self.edge_matches(e, from, to))
            .map(|(id, e)| (id, &e.data, e.weight))
            .collect()
    }

    /// Sets an attribute for a node.
    ///
    /// # Arguments
    /// * `node` - The `NodeId` of the node.
    /// * `key` - The key of the attribute.
    /// * `value` - The value of the attribute.
    ///
    /// # Returns
    /// A boolean indicating whether the attribute was successfully set.
    pub fn set_node_attribute(&mut self, node: NodeId, key: String, value: String) -> bool {
        if let Some(node) = self.nodes.get_mut(node) {
            node.attributes.insert(key, value);
            true
        } else {
            false
        }
    }

    /// Sets an attribute for an edge by its `EdgeId`.
    ///
    /// # Arguments
    /// * `edge_id` - The `EdgeId` of the edge.
    /// * `key` - The key of the attribute.
    /// * `value` - The value of the attribute.
    ///
    /// # Returns
    /// A boolean indicating whether the attribute was successfully set.
    pub fn set_edge_attribute(&mut self, edge_id: EdgeId, key: String, value: String) -> bool {
        if let Some(edge) = self.edges.get_mut(edge_id) {
            edge.attributes.insert(key, value);
            true
        } else {
            false
        }
    }

    /// Retrieves all neighboring nodes with their edges.
    ///
    /// # Arguments
    /// * `node` - The `NodeId` of the node.
    ///
    /// # Returns
    /// A vector of tuples containing the `NodeId` of the neighbor and a vector of tuples containing the `EdgeId` and weight of the edges connecting the nodes.
    pub fn get_neighbors(&self, node: NodeId) -> Vec<(NodeId, Vec<(EdgeId, W)>)> {
        self.nodes
            .get(node)
            .map(|n| n.neighbors.clone())
            .unwrap_or_default()
    }

    /// Validates the integrity of the graph.
    ///
    /// # Returns
    /// A boolean indicating whether the graph is valid.
    pub fn validate_graph(&self) -> bool {
        let edges_valid = self
            .edges
            .iter()
            .all(|(_, e)| self.nodes.contains(e.from) && self.nodes.contains(e.to));

        let neighbors_valid = self.nodes.iter().all(|(_, n)| {
            n.neighbors.iter().all(|&(id, ref edges)| {
                self.nodes.contains(id) && edges.iter().all(|&(eid, _)| self.edges.contains(eid))
            })
        });

        edges_valid && neighbors_valid
    }

    /// Normalizes edge direction for undirected graphs.
    fn normalize_edge(&self, mut from: NodeId, mut to: NodeId) -> (NodeId, NodeId) {
        if !self.directed && from > to {
            std::mem::swap(&mut from, &mut to);
        }
        (from, to)
    }

    /// Checks if an edge matches the given `from` and `to` nodes.
    fn edge_matches(&self, edge: &HEdge<W, E>, from: NodeId, to: NodeId) -> bool {
        if self.directed {
            edge.from == from && edge.to == to
        } else {
            (edge.from == from && edge.to == to) || (edge.from == to && edge.to == from)
        }
    }

    /// Returns all edges in the graph as a vector of (from, to, weight, edge_data).
    ///
    /// # Returns
    /// A vector of tuples containing the `NodeId` of the source node, the `NodeId` of the target node, the weight of the edge, and a reference to the edge's data.
    pub fn get_all_edges(&self) -> Vec<(NodeId, NodeId, W, &E)> {
        self.edges
            .iter()
            .map(|(_, edge)| (edge.from, edge.to, edge.weight, &edge.data))
            .collect()
    }

    /// Returns all attributes for edges between two nodes.
    ///
    /// # Arguments
    /// * `from` - The `NodeId` of the source node.
    /// * `to` - The `NodeId` of the target node.
    ///
    /// # Returns
    /// An `Option` containing a reference to the attributes of the edges between the nodes if any exist, or `None` otherwise.
    pub fn get_all_edge_attributes(
        &self,
        from: NodeId,
        to: NodeId,
    ) -> Option<&HashMap<String, String>> {
        self.edges
            .iter()
            .find(|(_, edge)| self.edge_matches(edge, from, to))
            .map(|(_, edge)| &edge.attributes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hgraph::h_edge::EdgeType;
    use crate::hgraph::h_node::NodeType;

    #[derive(Clone, Eq, PartialEq, Hash, Debug)]
    struct City {
        name: String,
        population: u32,
    }

    impl NodeType for City {
        fn as_string(&self) -> String {
            format!("{} ({})", self.name, self.population)
        }
    }

    #[derive(Clone, Debug, Default)]
    struct Road {
        length: f64,
        road_type: String,
    }

    impl EdgeType for Road {
        fn as_string(&self) -> String {
            format!("{} km, {}", self.length, self.road_type)
        }
    }

    #[test]
    fn test_basic_graph() {
        let mut graph = HeterogeneousGraph::<f64, City, Road>::new(false);
        let n1 = graph.add_node(City {
            name: "London".to_string(),
            population: 8_982_000,
        });
        let n2 = graph.add_node(City {
            name: "Paris".to_string(),
            population: 2_165_000,
        });

        let edge_id = graph
            .add_edge(
                n1,
                n2,
                343.0,
                Road {
                    length: 343.0,
                    road_type: "Eurostar".to_string(),
                },
            )
            .unwrap();

        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.get_node_data(n1).unwrap().name, "London");
        assert_eq!(
            graph.get_edge_data_by_id(edge_id).unwrap().road_type,
            "Eurostar"
        );
        assert!(graph.validate_graph());
    }

    #[test]
    fn test_multigraph() {
        let mut graph = HeterogeneousGraph::<f64, City, Road>::new(false);
        let london = graph.add_node(City {
            name: "London".to_string(),
            population: 8_982_000,
        });
        let paris = graph.add_node(City {
            name: "Paris".to_string(),
            population: 2_165_000,
        });

        let edge1 = graph
            .add_edge(
                london,
                paris,
                343.0,
                Road {
                    length: 343.0,
                    road_type: "Eurostar".to_string(),
                },
            )
            .unwrap();
        let _edge2 = graph
            .add_edge(
                london,
                paris,
                450.0,
                Road {
                    length: 450.0,
                    road_type: "Flight".to_string(),
                },
            )
            .unwrap();

        let edges = graph.get_edges_between(london, paris);
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[0].1.road_type, "Eurostar");
        assert_eq!(edges[1].1.road_type, "Flight");

        graph.remove_edge(edge1);
        assert_eq!(graph.get_edges_between(london, paris).len(), 1);
        assert!(graph.validate_graph());
    }

    #[test]
    fn test_directed_graph() {
        let mut graph = HeterogeneousGraph::<f64, City, Road>::new(true);
        let n1 = graph.add_node(City {
            name: "A".to_string(),
            population: 1000,
        });
        let n2 = graph.add_node(City {
            name: "B".to_string(),
            population: 2000,
        });

        let edge_id = graph
            .add_edge(
                n1,
                n2,
                10.0,
                Road {
                    length: 10.0,
                    road_type: "Highway".to_string(),
                },
            )
            .unwrap();

        let edges_forward = graph.get_edges_between(n1, n2);
        let edges_backward = graph.get_edges_between(n2, n1);
        assert_eq!(edges_forward.len(), 1);
        assert_eq!(edges_backward.len(), 0);
        assert_eq!(graph.get_neighbors(n1)[0].0, n2);
        assert!(graph.get_neighbors(n2).is_empty());
        assert_eq!(
            graph.get_edge_data_by_id(edge_id).unwrap().road_type,
            "Highway"
        );
    }

    #[test]
    fn test_attributes() {
        let mut graph = HeterogeneousGraph::<f64, City, Road>::new(false);
        let n1 = graph.add_node(City {
            name: "X".to_string(),
            population: 500,
        });
        let n2 = graph.add_node(City {
            name: "Y".to_string(),
            population: 600,
        });
        let edge_id = graph
            .add_edge(
                n1,
                n2,
                5.0,
                Road {
                    length: 5.0,
                    road_type: "Path".to_string(),
                },
            )
            .unwrap();

        assert!(graph.set_node_attribute(n1, "type".to_string(), "city".to_string()));
        assert!(graph.set_edge_attribute(edge_id, "condition".to_string(), "good".to_string()));
        assert!(!graph.set_node_attribute(999, "key".to_string(), "value".to_string()));
        assert!(!graph.set_edge_attribute(999, "key".to_string(), "value".to_string()));

        assert_eq!(
            graph.nodes[n1].attributes.get("type"),
            Some(&"city".to_string())
        );
        assert_eq!(
            graph.edges[edge_id].attributes.get("condition"),
            Some(&"good".to_string())
        );
    }

    #[test]
    fn test_invalid_edge_addition() {
        let mut graph = HeterogeneousGraph::<f64, City, Road>::new(false);
        let n1 = graph.add_node(City {
            name: "A".to_string(),
            population: 100,
        });

        let result = graph.add_edge(
            n1,
            999,
            1.0,
            Road {
                length: 1.0,
                road_type: "Test".to_string(),
            },
        );
        assert!(result.is_err());
        assert_eq!(graph.edges.len(), 0);
    }
}
