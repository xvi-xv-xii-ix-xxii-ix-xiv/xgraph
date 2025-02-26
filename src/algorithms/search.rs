//! Graph search algorithms and operations
//!
//! Provides functionality for:
//! - Path finding (DFS and BFS)
//! - Cycle detection
//! - Node existence checks
//!
//! # Examples
//!
//! Basic usage:
//! ```rust
//! use xgraph::graph::graph::Graph;
//! use xgraph::algorithms::search::Search;
//!
//! let mut graph = Graph::<u32, &str, &str>::new(true);
//! let a = graph.add_node("A");
//! let b = graph.add_node("B");
//! graph.add_edge(a, b, 1, "link").unwrap();
//!
//! assert!(graph.has_path(a, b));
//! assert_eq!(graph.bfs_path(a, b), Some(vec![a, b]));
//! ```

use crate::graph::graph::Graph;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;

/// Trait for graph search operations
///
/// Provides fundamental algorithms for graph traversal and analysis
pub trait Search<W, N, E> {
    /// Checks if a path exists between two nodes using DFS
    ///
    /// # Arguments
    /// * `start` - Starting node ID
    /// * `target` - Target node ID
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::search::Search;
    ///
    /// let mut graph = Graph::<u32, u32, ()>::new(true);
    /// let a = graph.add_node(0);
    /// let b = graph.add_node(1);
    /// let c = graph.add_node(2);
    ///
    /// graph.add_edge(a, b, 1, ()).unwrap();
    /// graph.add_edge(b, c, 1, ()).unwrap();
    ///
    /// assert!(graph.has_path(a, c));
    /// assert!(!graph.has_path(c, a)); // Directed graph would fail this
    /// ```
    fn has_path(&self, start: usize, target: usize) -> bool;

    /// Finds shortest path between nodes using BFS
    ///
    /// # Arguments
    /// * `start` - Starting node ID
    /// * `target` - Target node ID
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::search::Search;
    ///
    /// let mut graph = Graph::new(true);
    /// let nodes = (0..4).map(|i| graph.add_node(i)).collect::<Vec<_>>();
    ///
    /// graph.add_edge(nodes[0], nodes[1], 1, ()).unwrap();
    /// graph.add_edge(nodes[1], nodes[2], 1, ()).unwrap();
    /// graph.add_edge(nodes[0], nodes[3], 1, ()).unwrap();
    ///
    /// assert_eq!(
    ///     graph.bfs_path(nodes[0], nodes[2]),
    ///     Some(vec![nodes[0], nodes[1], nodes[2]])
    /// );
    /// ```
    fn bfs_path(&self, start: usize, target: usize) -> Option<Vec<usize>>;

    /// Recursive DFS implementation for path checking
    ///
    /// # Arguments
    /// * `current` - Current node in recursion
    /// * `target` - Target node to find
    /// * `visited` - Mutable reference to visited nodes set
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::search::Search;
    /// use std::collections::HashSet;
    ///
    /// let mut graph = Graph::new(false);
    /// let a = graph.add_node("A");
    /// let b = graph.add_node("B");
    /// let mut visited = HashSet::new();
    ///
    /// graph.add_edge(a, b, 1, ()).unwrap();
    /// assert!(graph.dfs(a, b, &mut visited));
    /// ```
    fn dfs(&self, current: usize, target: usize, visited: &mut HashSet<usize>) -> bool;

    /// Checks if node exists in the graph
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::search::Search;
    ///
    /// let graph: Graph<i32, (), ()> = Graph::new(false);
    /// assert!(!graph.has_node(0));
    /// ```
    fn has_node(&self, node: usize) -> bool;

    /// Detects cycles in the graph
    ///
    /// Uses different algorithms for directed vs undirected graphs:
    /// - Directed: DFS with recursion stack
    /// - Undirected: DFS with parent tracking
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::search::Search;
    ///
    /// // Directed cycle
    /// let mut directed = Graph::new(true);
    /// let nodes = (0..3).map(|i| directed.add_node(i)).collect::<Vec<_>>();
    /// directed.add_edge(nodes[0], nodes[1], 1, ()).unwrap();
    /// directed.add_edge(nodes[1], nodes[2], 1, ()).unwrap();
    /// directed.add_edge(nodes[2], nodes[0], 1, ()).unwrap();
    /// assert!(directed.has_cycle());
    ///
    /// // Undirected cycle
    /// let mut undirected = Graph::new(false);
    /// let nodes = (0..3).map(|i| undirected.add_node(i)).collect::<Vec<_>>();
    /// undirected.add_edge(nodes[0], nodes[1], 1, ()).unwrap();
    /// undirected.add_edge(nodes[1], nodes[2], 1, ()).unwrap();
    /// undirected.add_edge(nodes[2], nodes[0], 1, ()).unwrap();
    /// assert!(undirected.has_cycle());
    /// ```
    fn has_cycle(&self) -> bool;

    /// Helper for directed cycle detection
    ///
    /// # Arguments
    /// * `node` - Current node
    /// * `visited` - Global visited nodes
    /// * `recursion_stack` - Current recursion path
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::search::Search;
    /// use std::collections::HashSet;
    ///
    /// let mut graph = Graph::new(true);
    /// let a = graph.add_node(0);
    /// let b = graph.add_node(1);
    /// graph.add_edge(a, b, 1, ()).unwrap();
    /// graph.add_edge(b, a, 1, ()).unwrap();
    ///
    /// let mut visited = HashSet::new();
    /// let mut stack = HashSet::new();
    /// assert!(graph.has_cycle_directed(a, &mut visited, &mut stack));
    /// ```
    fn has_cycle_directed(
        &self,
        node: usize,
        visited: &mut HashSet<usize>,
        recursion_stack: &mut HashSet<usize>,
    ) -> bool;

    /// Helper for undirected cycle detection
    ///
    /// # Arguments
    /// * `node` - Current node
    /// * `parent` - Parent node in DFS tree
    /// * `visited` - Global visited nodes
    ///
    /// # Examples
    /// ```rust
    /// use xgraph::graph::graph::Graph;
    /// use xgraph::algorithms::search::Search;
    /// use std::collections::HashSet;
    ///
    /// let mut graph = Graph::new(false);
    /// let a = graph.add_node(0);
    /// let b = graph.add_node(1);
    /// let c = graph.add_node(2);
    /// graph.add_edge(a, b, 1, ()).unwrap();
    /// graph.add_edge(b, c, 1, ()).unwrap();
    /// graph.add_edge(c, a, 1, ()).unwrap();
    ///
    /// let mut visited = HashSet::new();
    /// assert!(graph.has_cycle_undirected(a, None, &mut visited));
    /// ```
    fn has_cycle_undirected(
        &self,
        node: usize,
        parent: Option<usize>,
        visited: &mut HashSet<usize>,
    ) -> bool;
}

impl<W, N, E> Search<W, N, E> for Graph<W, N, E>
where
    W: Copy + Default + PartialEq,
    N: Clone + Eq + Hash + Debug,
    E: Clone + Default + Debug,
{
    /// Checks if there is a path from the start node to the target node using DFS.
    fn has_path(&self, start: usize, target: usize) -> bool {
        let mut visited = HashSet::new();
        self.dfs(start, target, &mut visited)
    }

    /// Finds a path from the start node to the target node using BFS.
    fn bfs_path(&self, start: usize, target: usize) -> Option<Vec<usize>> {
        if !self.has_node(start) || !self.has_node(target) {
            return None;
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent = HashMap::new();

        queue.push_back(start);
        visited.insert(start);

        while let Some(current) = queue.pop_front() {
            if current == target {
                // Reconstruct the path from start to target
                let mut path = vec![current];
                let mut node = current;
                while let Some(&p) = parent.get(&node) {
                    path.push(p);
                    node = p;
                }
                path.reverse();
                return Some(path);
            }

            for &(neighbor, _) in &self.nodes[current].neighbors {
                if !visited.contains(&neighbor) && self.has_node(neighbor) {
                    visited.insert(neighbor);
                    parent.insert(neighbor, current);
                    queue.push_back(neighbor);
                }
            }
        }
        None
    }

    /// Performs Depth First Search (DFS) to check if there is a path from `current` to `target`.
    fn dfs(&self, current: usize, target: usize, visited: &mut HashSet<usize>) -> bool {
        if current == target {
            return true;
        }

        if !self.has_node(current) || !self.has_node(target) {
            return false;
        }

        visited.insert(current);

        for &(neighbor, _) in &self.nodes[current].neighbors {
            if !visited.contains(&neighbor)
                && self.has_node(neighbor)
                && self.dfs(neighbor, target, visited)
            {
                return true;
            }
        }
        false
    }

    /// Checks if the graph contains a given node.
    fn has_node(&self, node: usize) -> bool {
        self.nodes.get(node).is_some()
    }

    /// Checks if the graph contains any cycles.
    fn has_cycle(&self) -> bool {
        if self.directed {
            let mut visited = HashSet::new();
            let mut recursion_stack = HashSet::new();

            for (node_id, _) in self.nodes.iter() {
<<<<<<< HEAD
                if !visited.contains(&node_id) && self.has_cycle_directed(node_id, &mut visited, &mut recursion_stack) {
=======
                if !visited.contains(&node_id)
                    && self.has_cycle_directed(node_id, &mut visited, &mut recursion_stack)
                {
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)
                    return true;
                }
            }
            false
        } else {
            let mut visited = HashSet::new();

<<<<<<< HEAD



            for (node_id, _) in self.nodes.iter() {
                if !visited.contains(&node_id) && self.has_cycle_undirected(node_id, None, &mut visited) {
=======
            for (node_id, _) in self.nodes.iter() {
                if !visited.contains(&node_id)
                    && self.has_cycle_undirected(node_id, None, &mut visited)
                {
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)
                    return true;
                }
            }

            false
        }
    }

    /// Helper function for cycle detection in directed graphs using DFS.
    fn has_cycle_directed(
        &self,
        node: usize,
        visited: &mut HashSet<usize>,
        recursion_stack: &mut HashSet<usize>,
    ) -> bool {
        if recursion_stack.contains(&node) {
            return true;
        }

        if visited.contains(&node) {
            return false;
        }

        visited.insert(node);
        recursion_stack.insert(node);

        if let Some(neighbors) = self.nodes.get(node).map(|n| &n.neighbors) {
            for &(neighbor, _) in neighbors {
                if self.has_cycle_directed(neighbor, visited, recursion_stack) {
                    return true;
                }
            }
        }

        recursion_stack.remove(&node);
        false
    }

    /// Helper function for cycle detection in undirected graphs using DFS.
    fn has_cycle_undirected(
        &self,
        node: usize,
        parent: Option<usize>,
        visited: &mut HashSet<usize>,
    ) -> bool {
        if visited.contains(&node) {
            return true;
        }

        visited.insert(node);

        if let Some(neighbors) = self.nodes.get(node).map(|n| &n.neighbors) {
            for &(neighbor, _) in neighbors {
                if Some(neighbor) == parent {
                    continue;
                }

                if self.has_cycle_undirected(neighbor, Some(node), visited) {
                    return true;
                }
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_path() {
        let mut graph = Graph::<u32, (), ()>::new(false);
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        graph.add_edge(n0, n1, 1, ()).unwrap();

        assert!(graph.has_path(n0, n1));
        assert!(graph.has_path(n1, n0));
    }

    #[test]
    fn test_bfs_path() {
        let mut graph = Graph::<u32, (), ()>::new(true);
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        graph.add_edge(n0, n1, 1, ()).unwrap();
        graph.add_edge(n1, n2, 1, ()).unwrap();

        assert_eq!(graph.bfs_path(n0, n2), Some(vec![n0, n1, n2]));
    }

    #[test]
    fn test_dfs() {
        let mut graph = Graph::<u32, (), ()>::new(false);
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        graph.add_edge(n0, n1, 1, ()).unwrap();
        graph.add_edge(n1, n2, 1, ()).unwrap();

        let mut visited = HashSet::new();
        assert!(graph.dfs(n0, n2, &mut visited));
    }

    #[test]
    fn test_invalid_nodes() {
        let graph = Graph::<u32, (), ()>::new(false);
        assert!(!graph.has_node(0));
        assert!(!graph.has_path(0, 1));
        assert_eq!(graph.bfs_path(0, 1), None);
    }

    #[test]
    fn test_cycle_detection_directed() {
        let mut graph = Graph::<u32, (), ()>::new(true);
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        graph.add_edge(n0, n1, 1, ()).unwrap();
        graph.add_edge(n1, n2, 1, ()).unwrap();
        graph.add_edge(n2, n0, 1, ()).unwrap();

        assert!(graph.has_cycle());
    }

    #[test]
    fn test_cycle_detection_undirected() {
        let mut graph = Graph::<u32, (), ()>::new(false);
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        graph.add_edge(n0, n1, 1, ()).unwrap();
        graph.add_edge(n1, n2, 1, ()).unwrap();
        graph.add_edge(n2, n0, 1, ()).unwrap();

        assert!(graph.has_cycle());
    }

    #[test]
    fn test_no_cycle_directed() {
        let mut graph = Graph::<u32, (), ()>::new(true);
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        graph.add_edge(n0, n1, 1, ()).unwrap();
        graph.add_edge(n1, n2, 1, ()).unwrap();

        assert!(!graph.has_cycle());
    }

    #[test]
    fn test_no_cycle_undirected() {
        let mut graph = Graph::<u32, (), ()>::new(false);
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        graph.add_edge(n0, n1, 1, ()).unwrap();
        graph.add_edge(n1, n2, 1, ()).unwrap();

        assert!(!graph.has_cycle());
    }
}
