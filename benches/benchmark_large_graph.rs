use criterion::{black_box, criterion_group, criterion_main, Criterion};
<<<<<<< HEAD
use xgraph::graph::graph::Graph;
use xgraph::prelude::shortest_path::ShortestPath;
=======
use xgraph::algorithms::shortest_path::ShortestPath;
use xgraph::prelude::*;
>>>>>>> 093494d (refactor(leiden): enhance algorithm structure and documentation)

fn benchmark_large_graph(c: &mut Criterion) {
    // Создание большого графа
    let mut graph = Graph::new(false);
    for i in 0..10_000 {
        graph.add_node(i);
        if i > 0 {
            let _ = graph.add_edge(i - 1, i, 1, ());
        }
    }

    c.bench_function("large graph creation 10,000 nodes", |b| {
        b.iter(|| {
            // Просто создаем граф, но без изменений
            black_box(&graph);
        });
    });

    c.bench_function("large graph dijkstra 10,000 nodes", |b| {
        b.iter(|| {
            // Вызываем алгоритм Dijkstra для поиска кратчайшего пути
            black_box(graph.dijkstra(0));
        });
    });
}

criterion_group!(large_graph_benchmarks, benchmark_large_graph);
criterion_main!(large_graph_benchmarks);
