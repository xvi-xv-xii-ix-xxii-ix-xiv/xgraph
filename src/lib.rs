//! # xgraph
//!
//! A comprehensive graph theory library providing data structures and algorithms
//! for working with directed and undirected graphs.

pub mod algorithms;
pub mod graph;
pub mod utils;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::algorithms::*;
    pub use crate::graph::graph::Graph;
}
