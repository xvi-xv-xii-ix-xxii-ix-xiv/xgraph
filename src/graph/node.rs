use std::collections::HashMap;

/// A type alias for node IDs, typically used for indexing nodes.
type NodeId = usize;

/// A struct representing a graph node with data, neighbors, and attributes.
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
    /// Returns a default-initialized Node where:
    /// - `data` is set to the default value of the type `N`
    /// - `neighbors` is an empty vector
    /// - `attributes` is an empty hash map
    fn default() -> Self {
        Self {
            data: N::default(),
            neighbors: Vec::new(),
            attributes: HashMap::new(),
        }
    }
}

impl<W, N> Node<W, N> {
    /// Creates a new `Node` with the provided data, leaving neighbors and attributes empty.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to associate with the node.
    ///
    /// # Returns
    ///
    /// A new `Node` instance with the given data, and empty neighbors and attributes.
    pub fn new(data: N) -> Self {
        Node {
            data,
            neighbors: Vec::new(),
            attributes: HashMap::new(),
        }
    }
}
