use criterion::{criterion_group, criterion_main, Criterion};
use xgraph::hgraph::prelude::*;

fn benchmark_graph_creation(c: &mut Criterion) {
    c.bench_function("create graph with 1000 nodes", |b| {
        let mut graph = Graph::<u32, usize, ()>::new(false);
        b.iter(|| {
            for i in 0..1000 {
                graph.add_node(i);
            }
        });
    });

    c.bench_function("create graph with 1000 nodes and edges", |b| {
        let mut graph = Graph::<u32, usize, ()>::new(false);
        b.iter(|| {
            for i in 0..1000 {
                graph.add_node(i);
                if i > 0 {
                    let _ = graph.add_edge(i - 1, i, 1, ());
                }
            }
        });
    });
}

criterion_group!(graph_benchmarks, benchmark_graph_creation);
criterion_main!(graph_benchmarks);
