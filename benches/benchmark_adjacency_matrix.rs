use criterion::{black_box, criterion_group, criterion_main, Criterion};
use xgraph::hgraph::prelude::*;

fn benchmark_adjacency_matrix(c: &mut Criterion) {
    // Создание графа с 1000 узлами и рёбрами
    let mut graph = Graph::new(false);
    for i in 0..1000 {
        let _ = graph.add_node(i);
        if i > 0 {
            let _ = graph.add_edge(i - 1, i, 1, ());
        }
    }

    c.bench_function("adjacency matrix creation 1000 nodes", |b| {
        b.iter(|| {
            black_box(graph.to_adjacency_matrix()); // Используем black_box для предотвращения оптимизаций
        });
    });

    // Создание графа с 10,000 узлами и рёбрами
    let mut graph_large = Graph::new(false);
    for i in 0..10_000 {
        graph_large.add_node(i);
        if i > 0 {
            let _ = graph_large.add_edge(i - 1, i, 1, ());
        }
    }

    c.bench_function("adjacency matrix creation 10,000 nodes", |b| {
        b.iter(|| {
            black_box(graph_large.to_adjacency_matrix()); // Используем black_box
        });
    });
}

criterion_group!(adjacency_matrix_benchmarks, benchmark_adjacency_matrix);
criterion_main!(adjacency_matrix_benchmarks);
