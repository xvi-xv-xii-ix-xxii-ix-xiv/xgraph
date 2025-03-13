use std::collections::HashMap;
use std::fmt::Debug;

/// Type alias for node identifiers in the graph.
pub type NodeId = usize;

/// Type alias for edge identifiers in the graph.
pub type EdgeId = usize;

/// Trait defining the requirements for edge data types in a heterogeneous graph.
///
/// This trait ensures that edge data can be cloned and debugged, and provides a method
/// to convert the edge data into a string representation, which is useful for I/O operations or debugging.
///
/// # Requirements
/// - Implementors must be `Clone` and `Debug`.
/// - The `as_string` method must return a string representation of the data.
///
/// # Examples
/// ```rust
/// use xgraph::hgraph::h_edge::EdgeType;
///
/// // Custom edge type
/// #[derive(Clone, Debug)]
/// struct Connection {
///     kind: String,
///     since: u32,
/// }
///
/// impl EdgeType for Connection {
///     fn as_string(&self) -> String {
///         format!("{} since {}", self.kind, self.since)
///     }
/// }
///
/// let conn = Connection { kind: "friend".to_string(), since: 2020 };
/// assert_eq!(conn.as_string(), "friend since 2020");
/// ```
pub trait EdgeType: Clone + Debug {
    /// Returns a string representation of the edge data.
    ///
    /// This method is used for serialization (e.g., to CSV) or debugging purposes.
    ///
    /// # Returns
    /// A `String` representing the edge data.
    fn as_string(&self) -> String;
}

/// A structure representing an edge in a heterogeneous multigraph.
///
/// This struct holds the edge's unique identifier, source and target nodes, weight, data,
/// and attributes. It is designed to work within a `HeterogeneousGraph`, supporting multiple edges
/// between the same pair of nodes by using a unique `id`.
///
/// # Type Parameters
/// - `W`: The type of the edge weight (e.g., `f64`, `u32`).
/// - `E`: The type of the edge data, which must implement `EdgeType`.
///
/// # Fields
/// - `id`: A unique identifier for the edge.
/// - `from`: The `NodeId` of the source node.
/// - `to`: The `NodeId` of the target node.
/// - `weight`: The weight of the edge.
/// - `data`: The edge's data of type `E`.
/// - `attributes`: A hash map of string key-value pairs for additional edge metadata.
///
/// # Examples
/// ```rust
/// use xgraph::hgraph::h_edge::{HEdge, EdgeType};
/// use std::collections::HashMap;
///
/// #[derive(Clone, Debug)]
/// struct Relation(String);
/// impl EdgeType for Relation {
///     fn as_string(&self) -> String { self.0.clone() }
/// }
///
/// let mut edge: HEdge<f64, Relation> = HEdge::new(0, 1, 2, 1.5, Relation("friend".to_string()));
/// edge.attributes.insert("since".to_string(), "2020".to_string());
///
/// assert_eq!(edge.id, 0);
/// assert_eq!(edge.from, 1);
/// assert_eq!(edge.to, 2);
/// assert_eq!(edge.weight, 1.5);
/// assert_eq!(edge.data.as_string(), "friend");
/// assert_eq!(edge.attributes.get("since"), Some(&"2020".to_string()));
/// ```
#[derive(Debug)]
pub struct HEdge<W, E: EdgeType> {
    /// Unique identifier for the edge.
    pub id: EdgeId,
    /// Source node identifier.
    pub from: NodeId,
    /// Target node identifier.
    pub to: NodeId,
    /// Weight of the edge.
    pub weight: W,
    /// Data associated with the edge.
    pub data: E,
    /// Map of attributes as key-value pairs.
    pub attributes: HashMap<String, String>,
}

impl<W, E: EdgeType> HEdge<W, E> {
    /// Creates a new edge with the specified parameters.
    ///
    /// Initializes an edge with a unique ID, source and target nodes, weight, and data,
    /// with an empty attributes map.
    ///
    /// # Arguments
    /// - `id`: The unique identifier for the edge.
    /// - `from`: The `NodeId` of the source node.
    /// - `to`: The `NodeId` of the target node.
    /// - `weight`: The weight of the edge.
    /// - `data`: The data to associate with the edge.
    ///
    /// # Returns
    /// A new `HEdge` instance.
    pub fn new(id: EdgeId, from: NodeId, to: NodeId, weight: W, data: E) -> Self {
        Self {
            id,
            from,
            to,
            weight,
            data,
            attributes: HashMap::new(), // Initialize with an empty attributes map
        }
    }
}

/// Implementations of `EdgeType` for common primitive types.
///
/// These implementations allow primitive types to be used directly as edge data in a heterogeneous graph.
impl EdgeType for String {
    fn as_string(&self) -> String {
        self.clone()
    }
}

impl EdgeType for u32 {
    fn as_string(&self) -> String {
        self.to_string()
    }
}

impl EdgeType for f64 {
    fn as_string(&self) -> String {
        self.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple test edge type for use in tests.
    #[derive(Clone, Debug)]
    struct TestEdge(String);

    impl EdgeType for TestEdge {
        fn as_string(&self) -> String {
            self.0.clone()
        }
    }

    /// Tests the creation of a new edge and basic property access.
    #[test]
    fn test_edge_creation() {
        let edge: HEdge<f64, TestEdge> = HEdge::new(0, 1, 2, 3.5, TestEdge("test".to_string()));

        assert_eq!(edge.id, 0);
        assert_eq!(edge.from, 1);
        assert_eq!(edge.to, 2);
        assert_eq!(edge.weight, 3.5);
        assert_eq!(edge.data.as_string(), "test");
        assert!(edge.attributes.is_empty());
    }

    /// Tests adding and retrieving attributes for an edge.
    #[test]
    fn test_edge_attributes() {
        let mut edge: HEdge<u32, TestEdge> =
            HEdge::new(1, 0, 1, 10, TestEdge("connection".to_string()));

        // Add attributes
        edge.attributes
            .insert("type".to_string(), "road".to_string());
        edge.attributes
            .insert("length".to_string(), "100".to_string());

        assert_eq!(edge.attributes.get("type"), Some(&"road".to_string()));
        assert_eq!(edge.attributes.get("length"), Some(&"100".to_string()));
        assert_eq!(edge.attributes.len(), 2);

        // Overwrite an attribute
        edge.attributes
            .insert("type".to_string(), "bridge".to_string());
        assert_eq!(edge.attributes.get("type"), Some(&"bridge".to_string()));
    }

    /// Tests `EdgeType` implementations for primitive types.
    #[test]
    fn test_edge_type_primitives() {
        let string_edge: HEdge<i32, String> = HEdge::new(0, 1, 2, 5, "relation".to_string());
        assert_eq!(string_edge.data.as_string(), "relation");

        let u32_edge: HEdge<f64, u32> = HEdge::new(1, 0, 1, 2.5, 42);
        assert_eq!(u32_edge.data.as_string(), "42");

        let f64_edge: HEdge<i64, f64> = HEdge::new(2, 2, 3, 10, 3.14);
        assert_eq!(f64_edge.data.as_string(), "3.14");
    }

    /// Tests edge creation with a custom edge type.
    #[test]
    fn test_custom_edge_type() {
        #[derive(Clone, Debug)]
        struct CustomEdge {
            label: String,
            priority: i32,
        }

        impl EdgeType for CustomEdge {
            fn as_string(&self) -> String {
                format!("{}:{}", self.label, self.priority)
            }
        }

        let edge: HEdge<f64, CustomEdge> = HEdge::new(
            0,
            1,
            2,
            1.0,
            CustomEdge {
                label: "path".to_string(),
                priority: 5,
            },
        );

        assert_eq!(edge.data.as_string(), "path:5");
        assert_eq!(edge.from, 1);
        assert_eq!(edge.to, 2);
        assert_eq!(edge.weight, 1.0);
    }
}
