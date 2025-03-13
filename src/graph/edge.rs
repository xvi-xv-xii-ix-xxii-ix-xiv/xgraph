use std::collections::HashMap;

/// A type alias for node IDs, typically used for indexing nodes in a graph.
///
/// This is an unsigned integer type (`usize`) to ensure non-negative indexing.
pub type NodeId = usize;

/// Error type for edge-related operations.
///
/// This enum can be extended in the future if operations that might fail are added.
#[derive(Debug)]
pub enum EdgeError {
    /// Placeholder for potential future errors (e.g., invalid node IDs or attribute operations).
    GenericError(String),
}

impl std::fmt::Display for EdgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EdgeError::GenericError(msg) => write!(f, "Edge error: {}", msg),
        }
    }
}

impl std::error::Error for EdgeError {}

/// A struct representing an edge between two nodes with a weight and additional data.
///
/// This struct is generic over the weight type `W` and edge data type `E`, providing flexibility
/// for various graph implementations.
///
/// # Type Parameters
/// - `W`: The type of the weight associated with the edge (must implement `Debug`, `Clone`, `PartialEq`, `Eq`).
/// - `E`: The type of the additional data stored in the edge (must implement `Debug`, `Clone`, `PartialEq`, `Eq`).
///
/// # Fields
/// - `from`: The ID of the node from which the edge originates.
/// - `to`: The ID of the node to which the edge points.
/// - `weight`: The weight or cost associated with the edge.
/// - `data`: The additional data associated with the edge.
/// - `attributes`: A collection of attributes for the edge, stored as key-value pairs.
///
/// # Examples
///
/// Creating a new edge:
/// ```rust
/// use xgraph::graph::edge::Edge;
/// let edge = Edge::<i32, &str>::new(0, 1, 10, "road");
/// assert_eq!(edge.from, 0);
/// assert_eq!(edge.to, 1);
/// assert_eq!(edge.weight, 10);
/// assert_eq!(edge.data, "road");
/// assert!(edge.attributes.is_empty());
/// ```
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
    /// Returns a default-initialized `Edge`.
    ///
    /// The default edge has:
    /// - `from` and `to` set to `0`.
    /// - `weight` set to the default value of type `W` (requires `W: Default`).
    /// - `data` set to the default value of type `E` (requires `E: Default`).
    /// - `attributes` as an empty `HashMap`.
    ///
    /// # Returns
    /// A new `Edge` instance with default values.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::edge::Edge;
    /// let edge: Edge<i32, String> = Edge::default();
    /// assert_eq!(edge.from, 0);
    /// assert_eq!(edge.to, 0);
    /// assert_eq!(edge.weight, 0);
    /// assert_eq!(edge.data, String::new());
    /// assert!(edge.attributes.is_empty());
    /// ```
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
    /// Creates a new `Edge` with the provided parameters, initializing attributes as empty.
    ///
    /// # Arguments
    /// - `from`: The node ID from which the edge originates.
    /// - `to`: The node ID to which the edge points.
    /// - `weight`: The weight or cost of the edge.
    /// - `data`: Additional data associated with the edge.
    ///
    /// # Returns
    /// A new `Edge` instance with the specified properties and an empty attributes map.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::edge::Edge;
    /// let edge = Edge::<f64, &str>::new(1, 2, 3.5, "path");
    /// assert_eq!(edge.from, 1);
    /// assert_eq!(edge.to, 2);
    /// assert_eq!(edge.weight, 3.5);
    /// assert_eq!(edge.data, "path");
    /// assert!(edge.attributes.is_empty());
    /// ```
    pub fn new(from: NodeId, to: NodeId, weight: W, data: E) -> Self {
        Edge {
            from,
            to,
            weight,
            data,
            attributes: HashMap::new(),
        }
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
    /// use xgraph::graph::edge::Edge;
    /// let mut edge = Edge::<i32, &str>::new(0, 1, 5, "link");
    /// edge.attributes.insert("type".to_string(), "road".to_string());
    /// assert_eq!(edge.get_attribute("type"), Some(&"road".to_string()));
    /// assert_eq!(edge.get_attribute("color"), None);
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
    /// use xgraph::graph::edge::Edge;
    /// let mut edge = Edge::<i32, &str>::new(0, 1, 5, "link");
    /// edge.set_attribute("type".to_string(), "rail".to_string()).unwrap();
    /// assert_eq!(edge.get_attribute("type"), Some(&"rail".to_string()));
    /// ```
    pub fn set_attribute(&mut self, key: String, value: String) -> Result<(), EdgeError> {
        self.attributes.insert(key, value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_creation() {
        let edge = Edge::<i32, &str>::new(0, 1, 10, "connection");
        assert_eq!(edge.from, 0);
        assert_eq!(edge.to, 1);
        assert_eq!(edge.weight, 10);
        assert_eq!(edge.data, "connection");
        assert!(edge.attributes.is_empty());
    }

    #[test]
    fn test_default_edge() {
        let edge: Edge<i32, String> = Edge::default();
        assert_eq!(edge.from, 0);
        assert_eq!(edge.to, 0);
        assert_eq!(edge.weight, 0);
        assert_eq!(edge.data, String::new());
        assert!(edge.attributes.is_empty());
    }

    #[test]
    fn test_attributes() {
        let mut edge = Edge::<f32, &str>::new(1, 2, 3.5, "path");
        edge.set_attribute("color".to_string(), "blue".to_string())
            .unwrap();
        assert_eq!(edge.get_attribute("color"), Some(&"blue".to_string()));
        assert_eq!(edge.get_attribute("size"), None);
    }

    #[test]
    fn test_edge_equality() {
        let edge1 = Edge::<u32, &str>::new(0, 1, 5, "link");
        let edge2 = Edge::<u32, &str>::new(0, 1, 5, "link");
        assert_eq!(edge1, edge2);
    }
}
