use crate::graph::graph::Graph;
use std::hash::Hash;

/// A trait for converting a graph between different node and edge data types.
///
/// This trait provides methods to transform a `Graph<W, N, E>` into a new graph with different
/// node (`N2`) and edge (`E2`) types, or specifically into a graph with `String` as the data type
/// for both nodes and edges.
///
/// # Type Parameters
/// - `W`: The weight type of the graph edges (e.g., `u32`, `f64`).
/// - `N`: The original node data type.
/// - `E`: The original edge data type.
///
/// # Requirements
/// - The weight type `W` must implement `Copy`, `PartialEq`, and `Default`.
/// - The original node type `N` must implement `Clone`, `Eq`, `Hash`, and `Debug`.
/// - The original edge type `E` must implement `Clone`, `Debug`, and `Default`.
pub trait GraphConversion<W, N, E> {
    /// Converts the graph into a new graph with specified node and edge types.
    ///
    /// This method creates a new graph with the same structure (nodes, edges, weights, and direction),
    /// but with node data of type `N2` and edge data of type `E2`. Node and edge data are initialized
    /// using their default values, while attributes are preserved.
    ///
    /// # Type Parameters
    /// - `N2`: The new node data type, must implement `Clone`, `Eq`, `Hash`, `Debug`, and `Default`.
    /// - `E2`: The new edge data type, must implement `Clone`, `Debug`, and `Default`.
    ///
    /// # Returns
    /// A new `Graph<W, N2, E2>` with the converted structure and attributes.
    fn convert_to<N2, E2>(&self) -> Graph<W, N2, E2>
    where
        W: Copy + PartialEq + Default,
        N2: Clone + Eq + Hash + std::fmt::Debug + Default,
        E2: Clone + std::fmt::Debug + Default;

    /// Converts the graph into a graph with `String` node and edge data types.
    ///
    /// This method transforms the graph into one where node data is the string representation of
    /// the node ID, and edge data is an empty string. Attributes are preserved from the original graph.
    ///
    /// # Returns
    /// A new `Graph<W, String, String>` with the converted structure and attributes.
    fn to_string_graph(&self) -> Graph<W, String, String>
    where
        W: Copy + PartialEq + Default;
}

impl<W, N, E> GraphConversion<W, N, E> for Graph<W, N, E>
where
    W: Copy + PartialEq + Default,
    N: Clone + Eq + Hash + std::fmt::Debug,
    E: Clone + std::fmt::Debug + Default,
{
    fn convert_to<N2, E2>(&self) -> Graph<W, N2, E2>
    where
        W: Copy + PartialEq + Default,
        N2: Clone + Eq + Hash + std::fmt::Debug + Default,
        E2: Clone + std::fmt::Debug + Default,
    {
        // Create a new graph with the same directionality
        let mut new_graph = Graph::<W, N2, E2>::new(self.directed);

        // Convert nodes: use default values for new node data, copy attributes
        for (_id, node) in self.nodes.iter() {
            let new_id = new_graph.add_node(N2::default());
            for (key, value) in &node.attributes {
                new_graph.set_node_attribute(new_id, key.clone(), value.clone());
            }
        }

        // Convert edges: use default values for new edge data, copy attributes
        for (from, to, weight, _) in self.get_all_edges() {
            new_graph.add_edge(from, to, weight, E2::default()).unwrap();
            if let Some(attrs) = self.get_all_edge_attributes(from, to) {
                for (key, value) in attrs {
                    new_graph.set_edge_attribute(from, to, key.clone(), value.clone());
                }
            }
        }

        new_graph
    }

    fn to_string_graph(&self) -> Graph<W, String, String>
    where
        W: Copy + PartialEq + Default,
    {
        // Create a new graph with the same directionality
        let mut new_graph = Graph::<W, String, String>::new(self.directed);

        // Convert nodes: use node ID as string data, copy attributes
        for (id, node) in self.nodes.iter() {
            let new_id = new_graph.add_node(id.to_string());
            for (key, value) in &node.attributes {
                new_graph.set_node_attribute(new_id, key.clone(), value.clone());
            }
        }

        // Convert edges: use empty string as edge data, copy attributes
        for (from, to, weight, _) in self.get_all_edges() {
            new_graph
                .add_edge(from, to, weight, "".to_string())
                .unwrap();
            if let Some(attrs) = self.get_all_edge_attributes(from, to) {
                for (key, value) in attrs {
                    new_graph.set_edge_attribute(from, to, key.clone(), value.clone());
                }
            }
        }

        new_graph
    }
}
