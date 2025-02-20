use std::collections::HashMap;

/// A type alias for node IDs, typically used for indexing nodes.
type NodeId = usize;

/// A struct representing an edge between two nodes with a weight and additional data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edge<W, E> {
    /// The ID of the node from which the edge originates.
    pub from: NodeId,

    /// The ID of the node to which the edge points.
    pub to: NodeId,

    /// The weight or cost associated with the edge.
    pub weight: W,

    /// The additional data associated with the edge.
    pub data: E,

    /// A collection of attributes for the edge, represented as key-value pairs.
    pub attributes: HashMap<String, String>,
}

impl<W: Default, E: Default> Default for Edge<W, E> {
    /// Returns a default-initialized `Edge` where:
    /// - `from` and `to` are both set to `0`
    /// - `weight` is set to the default value of type `W`
    /// - `data` is set to the default value of type `E`
    /// - `attributes` is an empty hash map
    fn default() -> Self {
        Edge {
            from: 0,
            to: 0,
            weight: W::default(),
            data: E::default(),
            attributes: HashMap::new(),
        }
    }
}

impl<W, E> Edge<W, E> {
    /// Creates a new `Edge` with the provided parameters, leaving the attributes empty.
    ///
    /// # Arguments
    ///
    /// * `from` - The node ID from which the edge originates.
    /// * `to` - The node ID to which the edge points.
    /// * `weight` - The weight or cost of the edge.
    /// * `data` - Additional data associated with the edge.
    ///
    /// # Returns
    ///
    /// A new `Edge` instance with the specified properties, and empty attributes.
    pub fn new(from: NodeId, to: NodeId, weight: W, data: E) -> Self {
        Edge {
            from,
            to,
            weight,
            data,
            attributes: HashMap::new(),
        }
    }
}
