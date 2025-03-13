//! # xgraph
//!
//! A comprehensive graph theory library providing data structures and algorithms
//! for working with directed and undirected graphs.

#[cfg(feature = "graph")]
pub mod graph;
#[cfg(feature = "hgraph")]
pub mod hgraph;

pub mod prelude;

#[cfg(feature = "graph")]
pub use graph::{
    bridges, centrality, connectivity, leiden_clustering, search, shortest_path, wiedemann_ford,
    Edge, Graph, Node,
};

#[cfg(feature = "hgraph")]
pub use hgraph::{
    h_bridges, h_centrality, h_connectivity, h_leiden_clustering, h_search, h_shortest_path,
    h_wiedemann_ford, HEdge, HNode, HeterogeneousGraph,
};
