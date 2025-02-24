use crate::graph::graph::Graph;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::hash::Hash;

/// A trait for CSV input/output operations on graphs.
///
/// This trait defines methods to save a graph to CSV files and load a graph from CSV files.
/// It supports saving and loading node and edge data along with their attributes in a structured format.
///
/// # Type Parameters
/// - `W`: The weight type of the graph edges (e.g., `u32`, `f64`).
/// - `N`: The node data type.
/// - `E`: The edge data type.
///
/// # Requirements
/// - For `save_to_csv`: `W`, `N`, and `E` must implement `Display`.
/// - For `load_from_csv`: `W`, `N`, and `E` must implement `FromStr` and `Default`, with debuggable parse errors.
pub trait CsvIO<W, N, E> {
    /// Saves the graph to CSV files.
    ///
    /// This method writes the graph's nodes and edges to two separate CSV files:
    /// - `nodes_file`: Contains node IDs and their attributes as columns.
    /// - `edges_file`: Contains edge details (`from`, `to`, `weight`) and their attributes as columns.
    ///
    /// Attributes are dynamically determined from the graph and written as additional columns.
    /// Missing attributes for a node or edge are represented as empty strings.
    ///
    /// # Arguments
    /// - `nodes_file`: The path to the file where nodes will be saved.
    /// - `edges_file`: The path to the file where edges will be saved.
    ///
    /// # Returns
    /// An `io::Result<()>` indicating success or failure.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::io::csv_io::CsvIO;
    ///
    /// let mut graph: Graph<u32, String, String> = Graph::new(false);
    /// let n1 = graph.add_node("A".to_string());
    /// let n2 = graph.add_node("B".to_string());
    /// graph.add_edge(n1, n2, 1, "edge".to_string()).unwrap();
    /// graph.set_node_attribute(n1, "color".to_string(), "red".to_string());
    ///
    /// graph.save_to_csv("nodes.csv", "edges.csv").unwrap();
    /// ```
    fn save_to_csv(&self, nodes_file: &str, edges_file: &str) -> io::Result<()>
    where
        W: Copy + PartialEq + std::fmt::Display,
        N: Clone + Eq + Hash + std::fmt::Debug + std::fmt::Display,
        E: Clone + std::fmt::Debug + std::fmt::Display;

    /// Loads a graph from CSV files.
    ///
    /// This method reads a graph from two CSV files:
    /// - `nodes_file`: Contains node IDs and attributes.
    /// - `edges_file`: Contains edge details (`from`, `to`, `weight`) and attributes.
    ///
    /// The node and edge data are parsed from the CSV using `FromStr`, with node IDs used as temporary
    /// data if parsing fails. Attributes are loaded dynamically based on the CSV headers.
    ///
    /// # Arguments
    /// - `nodes_file`: The path to the file containing nodes.
    /// - `edges_file`: The path to the file containing edges.
    /// - `directed`: A boolean indicating whether the loaded graph should be directed.
    ///
    /// # Returns
    /// An `io::Result<Self>` containing the loaded graph or an error.
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::io::csv_io::CsvIO;
    ///
    /// // Assuming "nodes.csv" and "edges.csv" exist from a previous save
    /// let graph = Graph::<u32, String, String>::load_from_csv("nodes.csv", "edges.csv", false).unwrap();
    /// assert_eq!(graph.nodes.len(), 2);
    /// assert_eq!(graph.edges.len(), 1);
    /// ```
    fn load_from_csv(nodes_file: &str, edges_file: &str, directed: bool) -> io::Result<Self>
    where
        Self: Sized,
        W: Copy + PartialEq + Default + std::str::FromStr,
        N: Clone + Eq + Hash + std::fmt::Debug + std::str::FromStr,
        E: Clone + std::fmt::Debug + Default + std::str::FromStr,
        <W as std::str::FromStr>::Err: std::fmt::Debug,
        <N as std::str::FromStr>::Err: std::fmt::Debug,
        <E as std::str::FromStr>::Err: std::fmt::Debug;
}

impl<W, N, E> CsvIO<W, N, E> for Graph<W, N, E>
where
    W: Copy + PartialEq,
    N: Clone + Eq + Hash + std::fmt::Debug,
    E: Clone + std::fmt::Debug + Default,
{
    fn save_to_csv(&self, nodes_file: &str, edges_file: &str) -> io::Result<()>
    where
        W: std::fmt::Display,
        N: std::fmt::Display,
        E: std::fmt::Display,
    {
        // Save nodes: create the nodes CSV file
        let mut nodes_writer = File::create(nodes_file)?;
        let mut node_attrs: Vec<String> = self
            .nodes
            .iter()
            .flat_map(|(_, node)| node.attributes.keys())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .cloned()
            .collect();
        node_attrs.sort(); // Sort for consistent column order
        writeln!(nodes_writer, "node_id,{}", node_attrs.join(","))?;

        // Write each node's ID and attributes
        for (id, node) in self.nodes.iter() {
            let attrs: Vec<String> = node_attrs
                .iter()
                .map(|key| {
                    node.attributes
                        .get(key)
                        .map_or("".to_string(), |v| v.to_string())
                })
                .collect();
            writeln!(nodes_writer, "{},{}", id, attrs.join(","))?;
        }

        // Save edges: create the edges CSV file
        let mut edges_writer = File::create(edges_file)?;
        let mut edge_attrs: Vec<String> = self
            .edges
            .iter()
            .flat_map(|(_, edge)| edge.attributes.keys())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .cloned()
            .collect();
        edge_attrs.sort(); // Sort for consistent column order
        writeln!(
            edges_writer,
            "from,to,weight,{}",
            edge_attrs.join(",")
        )?;

        // Write each edge's details and attributes
        for (_, edge) in self.edges.iter() {
            let attrs: Vec<String> = edge_attrs
                .iter()
                .map(|key| {
                    edge.attributes
                        .get(key)
                        .map_or("".to_string(), |v| v.to_string())
                })
                .collect();
            writeln!(
                edges_writer,
                "{},{},{},{}",
                edge.from,
                edge.to,
                edge.weight,
                attrs.join(",")
            )?;
        }

        Ok(())
    }

    fn load_from_csv(nodes_file: &str, edges_file: &str, directed: bool) -> io::Result<Self>
    where
        W: Default + std::str::FromStr,
        N: std::str::FromStr,
        E: std::str::FromStr,
        <W as std::str::FromStr>::Err: std::fmt::Debug,
        <N as std::str::FromStr>::Err: std::fmt::Debug,
        <E as std::str::FromStr>::Err: std::fmt::Debug,
    {
        // Initialize a new graph with the specified directionality
        let mut graph = Graph::new(directed);

        // Load nodes: read the nodes CSV file
        let nodes_reader = BufReader::new(File::open(nodes_file)?);
        let mut lines = nodes_reader.lines();
        let header = lines.next().ok_or(io::Error::new(
            io::ErrorKind::InvalidData,
            "Empty nodes file",
        ))??;
        let attr_keys: Vec<&str> = header.split(',').skip(1).collect();

        // Parse each node line
        for line in lines {
            let line = line?;
            let parts: Vec<&str> = line.split(',').collect();
            let _node_id: usize = parts[0].parse().unwrap_or_default();
            let data = parts[0].parse().map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Node data parse error: {:?}", e),
                )
            })?;
            let node = graph.nodes.insert(crate::graph::node::Node::new(data));

            // Load node attributes
            for (key, value) in attr_keys.iter().zip(parts.iter().skip(1)) {
                if !value.is_empty() {
                    graph.set_node_attribute(node, key.to_string(), value.to_string());
                }
            }
        }

        // Load edges: read the edges CSV file
        let edges_reader = BufReader::new(File::open(edges_file)?);
        let mut lines = edges_reader.lines();
        let header = lines.next().ok_or(io::Error::new(
            io::ErrorKind::InvalidData,
            "Empty edges file",
        ))??;
        let attr_keys: Vec<&str> = header.split(',').skip(3).collect();

        // Parse each edge line
        for line in lines {
            let line = line?;
            let parts: Vec<&str> = line.split(',').collect();
            let from: usize = parts[0].parse().unwrap_or_default();
            let to: usize = parts[1].parse().unwrap_or_default();
            let weight = parts[2].parse().map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Weight parse error: {:?}", e),
                )
            })?;
            let data = "".parse().map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Edge data parse error: {:?}", e),
                )
            })?;

            graph.add_edge(from, to, weight, data).unwrap_or_default();

            // Load edge attributes
            for (key, value) in attr_keys.iter().zip(parts.iter().skip(3)) {
                if !value.is_empty() {
                    graph.set_edge_attribute(from, to, key.to_string(), value.to_string());
                }
            }
        }

        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests CSV input/output functionality.
    #[test]
    fn test_csv_io() {
        let mut graph = Graph::<u32, String, String>::new(false);
        let n1 = graph.add_node("A".to_string());
        let n2 = graph.add_node("B".to_string());
        graph.add_edge(n1, n2, 1, "edge".to_string()).unwrap();
        graph.set_node_attribute(n1, "color".to_string(), "red".to_string());
        graph.set_edge_attribute(n1, n2, "type".to_string(), "road".to_string());

        graph.save_to_csv("nodes.csv", "edges.csv").unwrap();
        let loaded_graph =
            Graph::<u32, String, String>::load_from_csv("nodes.csv", "edges.csv", false).unwrap();

        assert_eq!(graph.nodes.len(), loaded_graph.nodes.len());
        assert_eq!(graph.edges.len(), loaded_graph.edges.len());
    }
}