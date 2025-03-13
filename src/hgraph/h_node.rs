use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

/// Type alias for node identifiers in the graph.
pub type NodeId = usize;

/// Type alias for edge identifiers in the graph.
pub type EdgeId = usize;

/// Trait defining the requirements for node data types in a heterogeneous graph.
///
/// This trait ensures that node data can be cloned, compared for equality, hashed, and debugged.
/// Additionally, it provides a method to convert the node data into a string representation,
/// which is useful for I/O operations or debugging.
///
/// # Requirements
/// - Implementors must be `Clone`, `Eq`, `Hash`, and `Debug`.
/// - The `as_string` method must return a string representation of the data.
///
/// # Examples
/// ```rust
/// use xgraph::hgraph::h_node::NodeType;
///
/// // Custom node type
/// #[derive(Clone, Eq, PartialEq, Hash, Debug)]
/// struct Person {
///     name: String,
///     age: u32,
/// }
///
/// impl NodeType for Person {
///     fn as_string(&self) -> String {
///         format!("{} ({})", self.name, self.age)
///     }
/// }
///
/// let person = Person { name: "Alice".to_string(), age: 30 };
/// assert_eq!(person.as_string(), "Alice (30)");
/// ```
pub trait NodeType: Clone + Eq + Hash + Debug {
    /// Returns a string representation of the node data.
    ///
    /// This method is used for serialization (e.g., to CSV) or debugging purposes.
    ///
    /// # Returns
    /// A `String` representing the node data.
    fn as_string(&self) -> String;
}

/// A structure representing a node in a heterogeneous multigraph.
///
/// This struct holds the node's data, its neighbors (with multiple edges support), and attributes.
/// It is designed to work within a `HeterogeneousGraph`, supporting multiple edges between the same pair of nodes.
///
/// # Type Parameters
/// - `W`: The type of the edge weights (e.g., `f64`, `u32`).
/// - `N`: The type of the node data, which must implement `NodeType`.
///
/// # Fields
/// - `data`: The node's data of type `N`.
/// - `neighbors`: A vector of tuples `(NodeId, Vec<(EdgeId, W)>)`, where each tuple represents a neighboring node
///   and a list of edges (with their IDs and weights) connecting to it.
/// - `attributes`: A hash map of string key-value pairs for additional node metadata.
///
/// # Examples
/// ```rust
/// use xgraph::hgraph::h_node::{HNode, NodeType};
/// use std::collections::HashMap;
///
/// #[derive(Clone, Eq, PartialEq, Hash, Debug)]
/// struct City(String);
/// impl NodeType for City {
///     fn as_string(&self) -> String { self.0.clone() }
/// }
///
/// let mut node: HNode<f64, City> = HNode::new(City("London".to_string()));
/// node.add_neighbor(1, 0, 100.0); // Add neighbor with edge ID 0 and weight 100.0
/// node.attributes.insert("population".to_string(), "9000000".to_string());
///
/// assert_eq!(node.data.as_string(), "London");
/// assert_eq!(node.neighbors[0], (1, vec![(0, 100.0)]));
/// assert_eq!(node.attributes.get("population"), Some(&"9000000".to_string()));
/// ```
#[derive(Debug)]
pub struct HNode<W, N: NodeType> {
    /// The data associated with the node.
    pub data: N,
    /// List of neighbors: `(NodeId, Vec<(EdgeId, W)>)` for multigraph support.
    pub neighbors: Vec<(NodeId, Vec<(EdgeId, W)>)>,
    /// A map of attributes as key-value pairs.
    pub attributes: HashMap<String, String>,
}

impl<W, N: NodeType> HNode<W, N> {
    /// Creates a new node with the specified data.
    ///
    /// Initializes an empty list of neighbors and an empty attributes map.
    ///
    /// # Arguments
    /// - `data`: The data to associate with the node.
    ///
    /// # Returns
    /// A new `HNode` instance.
    pub fn new(data: N) -> Self {
        Self {
            data,
            neighbors: Vec::new(),
            attributes: HashMap::new(),
        }
    }

    /// Adds a neighbor to the node with a specific edge and weight.
    ///
    /// If the neighbor already exists, the new edge is appended to its list of edges.
    /// Otherwise, a new neighbor entry is created with the edge.
    ///
    /// # Arguments
    /// - `neighbor`: The `NodeId` of the neighboring node.
    /// - `edge_id`: The `EdgeId` of the edge connecting to the neighbor.
    /// - `weight`: The weight of the edge.
    pub fn add_neighbor(&mut self, neighbor: NodeId, edge_id: EdgeId, weight: W) {
        // Check if the neighbor already exists
        if let Some((_, edges)) = self.neighbors.iter_mut().find(|(id, _)| *id == neighbor) {
            // Append the new edge to the existing neighbor's edge list
            edges.push((edge_id, weight));
        } else {
            // Create a new neighbor entry with the edge
            self.neighbors.push((neighbor, vec![(edge_id, weight)]));
        }
    }

    /// Removes an edge to a specific neighbor based on its edge ID.
    ///
    /// If the neighbor exists and the edge ID matches, the edge is removed from the neighbor's edge list.
    /// If the edge list becomes empty, the neighbor remains in the list (not removed).
    ///
    /// # Arguments
    /// - `neighbor`: The `NodeId` of the neighbor.
    /// - `edge_id`: The `EdgeId` of the edge to remove.
    pub fn remove_edge(&mut self, neighbor: NodeId, edge_id: EdgeId) {
        // Find the neighbor and filter out the edge with the given edge_id
        if let Some((_, edges)) = self.neighbors.iter_mut().find(|(id, _)| *id == neighbor) {
            edges.retain(|&(id, _)| id != edge_id);
        }
    }
}

/// Implementations of `NodeType` for common primitive types.
///
/// These implementations allow primitive types to be used directly as node data in a heterogeneous graph.
impl NodeType for String {
    fn as_string(&self) -> String {
        self.clone()
    }
}

impl NodeType for i32 {
    fn as_string(&self) -> String {
        self.to_string()
    }
}

impl NodeType for i64 {
    fn as_string(&self) -> String {
        self.to_string()
    }
}

impl NodeType for u64 {
    fn as_string(&self) -> String {
        self.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple test node type for use in tests.
    #[derive(Clone, Eq, PartialEq, Hash, Debug)]
    struct TestNode(String);

    impl NodeType for TestNode {
        fn as_string(&self) -> String {
            self.0.clone()
        }
    }

    /// Tests the creation of a new node and basic property access.
    #[test]
    fn test_node_creation() {
        let node: HNode<f64, TestNode> = HNode::new(TestNode("Node1".to_string()));
        assert_eq!(node.data.as_string(), "Node1");
        assert!(node.neighbors.is_empty());
        assert!(node.attributes.is_empty());
    }

    /// Tests adding neighbors and edges to a node.
    #[test]
    fn test_add_neighbor() {
        let mut node: HNode<u32, TestNode> = HNode::new(TestNode("Node1".to_string()));

        // Add first edge to neighbor 2
        node.add_neighbor(2, 0, 10);
        assert_eq!(node.neighbors, vec![(2, vec![(0, 10)])]);

        // Add second edge to the same neighbor
        node.add_neighbor(2, 1, 20);
        assert_eq!(node.neighbors, vec![(2, vec![(0, 10), (1, 20)])]);

        // Add edge to a different neighbor
        node.add_neighbor(3, 2, 30);
        assert_eq!(
            node.neighbors,
            vec![(2, vec![(0, 10), (1, 20)]), (3, vec![(2, 30)])]
        );
    }

    /// Tests removing edges from a node's neighbor list.
    #[test]
    fn test_remove_edge() {
        let mut node: HNode<f64, TestNode> = HNode::new(TestNode("Node1".to_string()));

        // Add multiple edges to neighbor 2
        node.add_neighbor(2, 0, 1.0);
        node.add_neighbor(2, 1, 2.0);
        node.add_neighbor(3, 2, 3.0);

        // Remove edge with ID 1 from neighbor 2
        node.remove_edge(2, 1);
        assert_eq!(node.neighbors[0], (2, vec![(0, 1.0)]));
        assert_eq!(node.neighbors[1], (3, vec![(2, 3.0)]));

        // Remove non-existent edge (no change expected)
        node.remove_edge(2, 99);
        assert_eq!(node.neighbors[0], (2, vec![(0, 1.0)]));

        // Remove edge from non-existent neighbor (no panic expected)
        node.remove_edge(4, 0);
        assert_eq!(node.neighbors.len(), 2);
    }

    /// Tests attribute management for a node.
    #[test]
    fn test_attributes() {
        let mut node: HNode<i32, TestNode> = HNode::new(TestNode("Node1".to_string()));

        // Add attributes
        node.attributes
            .insert("key1".to_string(), "value1".to_string());
        node.attributes
            .insert("key2".to_string(), "value2".to_string());

        assert_eq!(node.attributes.get("key1"), Some(&"value1".to_string()));
        assert_eq!(node.attributes.get("key2"), Some(&"value2".to_string()));
        assert_eq!(node.attributes.len(), 2);

        // Overwrite an attribute
        node.attributes
            .insert("key1".to_string(), "new_value".to_string());
        assert_eq!(node.attributes.get("key1"), Some(&"new_value".to_string()));
    }

    /// Tests `NodeType` implementations for primitive types.
    #[test]
    fn test_node_type_primitives() {
        let string_node: HNode<f64, String> = HNode::new("Test".to_string());
        assert_eq!(string_node.data.as_string(), "Test");

        let i32_node: HNode<f64, i32> = HNode::new(42);
        assert_eq!(i32_node.data.as_string(), "42");

        let i64_node: HNode<f64, i64> = HNode::new(1234567890);
        assert_eq!(i64_node.data.as_string(), "1234567890");

        let u64_node: HNode<f64, u64> = HNode::new(9876543210);
        assert_eq!(u64_node.data.as_string(), "9876543210");
    }
}
