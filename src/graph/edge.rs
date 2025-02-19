// edge.rs
use std::collections::HashMap;

type NodeId = usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edge<W, E> {
    pub from: NodeId,
    pub to: NodeId,
    pub weight: W,
    pub data: E,
    pub attributes: HashMap<String, String>,
}

impl<W: Default, E: Default> Default for Edge<W, E> {
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
