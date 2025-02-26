use criterion::{black_box, criterion_group, criterion_main, Criterion};
use xgraph::algorithms::shortest_path::ShortestPath;
use xgraph::prelude::*;

fn benchmark_dijkstra(c: &mut Criterion) {
    // Создание графа с 1000 узлами и рёбрами
    let mut graph = Graph::new(false);
    for i in 0..1000 {
        graph.add_node(i);
        if i > 0 {
            let _ = graph.add_edge(i - 1, i, 1, ());
        }
    }

    c.bench_function("dijkstra 1000 nodes", |b| {
        b.iter(|| {
            black_box(graph.dijkstra(0)); // Используем black_box для предотвращения оптимизаций
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

    c.bench_function("dijkstra 10,000 nodes", |b| {
        b.iter(|| {
            black_box(graph_large.dijkstra(0)); // Используем black_box
        });
    });
}

criterion_group!(benchmarks, benchmark_dijkstra);
criterion_main!(benchmarks);
