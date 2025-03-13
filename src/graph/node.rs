use std::collections::HashMap;

/// A type alias for node IDs, typically used for indexing nodes in a graph.
///
/// This is an unsigned integer type (`usize`) to ensure non-negative indexing.
pub type NodeId = usize;

/// Error type for node-related operations.
///
/// This enum can be extended in the future if operations that might fail are added.
#[derive(Debug)]
pub enum NodeError {
    /// Placeholder for potential future errors (e.g., invalid attribute operations).
    GenericError(String),
}

impl std::fmt::Display for NodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeError::GenericError(msg) => write!(f, "Node error: {}", msg),
        }
    }
}

impl std::error::Error for NodeError {}

/// A struct representing a graph node with data, neighbors, and attributes.
///
/// This struct is generic over the weight type `W` and node data type `N`, allowing flexibility
/// in how nodes are defined and used within a graph.
///
/// # Type Parameters
/// - `W`: The type of the weight associated with edges to neighbors.
/// - `N`: The type of the data stored in the node (must implement `Debug`, `Clone`, `PartialEq`, `Eq`).
///
/// # Fields
/// - `data`: The data associated with the node.
/// - `neighbors`: A list of neighboring nodes, represented as tuples of `(NodeId, W)`.
/// - `attributes`: A collection of additional attributes for the node, stored as key-value pairs.
///
/// # Examples
///
/// Creating a new node:
/// ```rust
/// use xgraph::graph::node::Node;
/// let node = Node::<i32, &str>::new("City");
/// assert_eq!(node.data, "City");
/// assert!(node.neighbors.is_empty());
/// assert!(node.attributes.is_empty());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node<W, N> {
    /// The data associated with the node.
    pub data: N,

    /// The list of neighbors, represented as a tuple of node ID and weight.
    pub neighbors: Vec<(NodeId, W)>,

    /// A collection of additional attributes for the node, represented as key-value pairs.
    pub attributes: HashMap<String, String>,
}

impl<W, N: Default> Default for Node<W, N> {
    /// Returns a default-initialized `Node`.
    ///
    /// The default node has:
    /// - `data` set to the default value of type `N` (requires `N: Default`).
    /// - `neighbors` as an empty vector.
    /// - `attributes` as an empty `HashMap`.
    ///
    /// # Returns
    /// A new `Node` instance with default values.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::node::Node;
    /// let node: Node<i32, String> = Node::default();
    /// assert_eq!(node.data, String::new());
    /// assert!(node.neighbors.is_empty());
    /// assert!(node.attributes.is_empty());
    /// ```
    fn default() -> Self {
        Self {
            data: N::default(),
            neighbors: Vec::new(),
            attributes: HashMap::new(),
        }
    }
}

impl<W, N> Node<W, N> {
    /// Creates a new `Node` with the provided data, initializing neighbors and attributes as empty.
    ///
    /// # Arguments
    /// - `data`: The data to associate with the node.
    ///
    /// # Returns
    /// A new `Node` instance with the given data, an empty neighbors list, and an empty attributes map.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::node::Node;
    /// let node = Node::<f64, i32>::new(42);
    /// assert_eq!(node.data, 42);
    /// assert!(node.neighbors.is_empty());
    /// assert!(node.attributes.is_empty());
    /// ```
    pub fn new(data: N) -> Self {
        Node {
            data,
            neighbors: Vec::new(),
            attributes: HashMap::new(),
        }
    }

    /// Adds a neighbor to the node with the specified ID and weight.
    ///
    /// # Arguments
    /// - `neighbor_id`: The ID of the neighboring node.
    /// - `weight`: The weight of the edge to the neighbor.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::node::Node;
    /// let mut node = Node::<i32, &str>::new("A");
    /// node.add_neighbor(1, 10);
    /// assert_eq!(node.neighbors, vec![(1, 10)]);
    /// ```
    pub fn add_neighbor(&mut self, neighbor_id: NodeId, weight: W) {
        self.neighbors.push((neighbor_id, weight));
    }

    /// Gets an attribute value by key.
    ///
    /// # Arguments
    /// - `key`: The key of the attribute to retrieve.
    ///
    /// # Returns
    /// - `Some(&String)` if the attribute exists.
    /// - `None` if the attribute does not exist.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::node::Node;
    /// let mut node = Node::<i32, &str>::new("A");
    /// node.attributes.insert("color".to_string(), "blue".to_string());
    /// assert_eq!(node.get_attribute("color"), Some(&"blue".to_string()));
    /// ```
    pub fn get_attribute(&self, key: &str) -> Option<&String> {
        self.attributes.get(key)
    }

    /// Sets an attribute with the given key and value.
    ///
    /// # Arguments
    /// - `key`: The key of the attribute to set.
    /// - `value`: The value of the attribute to set.
    ///
    /// # Returns
    /// - `Ok(())` on success.
    /// - Currently no error cases, but structured for future expansion.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::node::Node;
    /// let mut node = Node::<i32, &str>::new("A");
    /// node.set_attribute("color".to_string(), "red".to_string()).unwrap();
    /// assert_eq!(node.get_attribute("color"), Some(&"red".to_string()));
    /// ```
    pub fn set_attribute(&mut self, key: String, value: String) -> Result<(), NodeError> {
        self.attributes.insert(key, value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = Node::<i32, &str>::new("Test");
        assert_eq!(node.data, "Test");
        assert!(node.neighbors.is_empty());
        assert!(node.attributes.is_empty());
    }

    #[test]
    fn test_default_node() {
        let node: Node<f64, String> = Node::default();
        assert_eq!(node.data, String::new());
        assert!(node.neighbors.is_empty());
        assert!(node.attributes.is_empty());
    }

    #[test]
    fn test_add_neighbor() {
        let mut node = Node::<u32, i32>::new(1);
        node.add_neighbor(2, 10);
        node.add_neighbor(3, 20);
        assert_eq!(node.neighbors, vec![(2, 10), (3, 20)]);
    }

    #[test]
    fn test_attributes() {
        let mut node = Node::<f32, &str>::new("Node");
        node.set_attribute("size".to_string(), "large".to_string())
            .unwrap();
        assert_eq!(node.get_attribute("size"), Some(&"large".to_string()));
        assert_eq!(node.get_attribute("color"), None);
    }
}
