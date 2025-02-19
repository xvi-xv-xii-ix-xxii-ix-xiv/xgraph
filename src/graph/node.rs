// node.rs
use std::collections::HashMap;

type NodeId = usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node<W, N> {
    pub data: N,
    pub neighbors: Vec<(NodeId, W)>,
    pub attributes: HashMap<String, String>,
}

impl<W, N: Default> Default for Node<W, N> {
    fn default() -> Self {
        Self {
            data: N::default(),
            neighbors: Vec::new(),
            attributes: HashMap::new(),
        }
    }
}

impl<W, N> Node<W, N> {
    pub fn new(data: N) -> Self {
        Node {
            data,
            neighbors: Vec::new(),
            attributes: HashMap::new(),
        }
    }
}
