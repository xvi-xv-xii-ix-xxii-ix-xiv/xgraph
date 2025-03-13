import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

def load_graph(nodes_file, edges_file):
    # Загружаем данные
    nodes_df = pd.read_csv(nodes_file)
    edges_df = pd.read_csv(edges_file)

    # Приводим node_id, from и to к int
    nodes_df["node_id"] = nodes_df["node_id"].astype(int)
    edges_df["from"] = edges_df["from"].astype(int)
    edges_df["to"] = edges_df["to"].astype(int)

    # Обрабатываем NaN в узлах
    nodes_df[["data", "region", "type"]] = nodes_df[["data", "region", "type"]].fillna("unknown").astype(str)
    nodes_df["population"] = nodes_df["population"].astype(str)

    # Обрабатываем NaN в рёбрах
    edges_df["weight"] = edges_df["weight"].astype(float)
    edges_df[["data", "priority"]] = edges_df[["data", "priority"]].fillna("unknown").astype(str)

    # Создаем мультиграф
    G = nx.MultiGraph()

    # Добавляем узлы
    for _, row in nodes_df.iterrows():
        G.add_node(row["node_id"], **row.to_dict())

    # Добавляем рёбра
    for _, row in edges_df.iterrows():
        edge_attributes = {
            "weight": row["weight"],
            "route": row["data"],
            "priority": row["priority"],
        }
        G.add_edge(row["from"], row["to"], **edge_attributes)

    return G

def draw_graph(G):
    plt.figure(figsize=(10, 8))  # Уменьшаем размер фигуры
    pos = nx.spring_layout(G, seed=42, k=0.3)

    # Определяем цвета узлов
    node_color_map = {
        "port": "#1f77b4",
        "trade_hub": "#ff7f0e",
        "settlement": "#2ca02c",
        "fortress": "#d62728",
        "religious_center": "#9467bd",
        "unknown": "#7f7f7f",
    }
    node_colors = [
        node_color_map.get(G.nodes[n].get("type", "unknown"), "#7f7f7f")
        for n in G.nodes
    ]

    # Рисуем узлы
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, edgecolors="black", node_size=100, alpha=0.9
    )

    # Подписываем узлы
    labels = {
        n: f"{n}: {G.nodes[n]['data']}\n({G.nodes[n].get('type', 'unknown')}, Pop: {G.nodes[n].get('population', 'N/A')})"
        for n in G.nodes
    }
    label_pos = {k: (v[0], v[1] + 0.05) for k, v in pos.items()}  # Смещение подписей вверх
    nx.draw_networkx_labels(G, label_pos, labels, font_size=8, font_weight="bold")  # Уменьшаем размер шрифта

    # Определяем цвета и стили рёбер
    edge_color_map = {
        "sea": "blue",
        "river": "green",
        "trade": "orange",
        "mountain": "brown",
        "unknown": "gray",
    }
    edge_style_map = {
        "sea": "solid",
        "river": "dashed",
        "trade": "dotted",
        "mountain": "dashdot",
        "unknown": "solid",
    }

    # Рисуем рёбра с изгибом
    for u, v, key, data in G.edges(keys=True, data=True):
        route_type = data.get("route", "unknown")
        edge_color = edge_color_map.get(route_type, "gray")
        edge_style = edge_style_map.get(route_type, "solid")

        # Изгиб для рёбер между одними и теми же узлами
        num_edges = len(G[u][v])
        if num_edges > 1:
            rad = 0.1 * key  # Изгиб зависит от ключа ребра
        else:
            rad = 0.0

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            edge_color=edge_color,
            style=edge_style,
            width=2,
            alpha=0.7,
            arrows=True,  # Включаем стрелки
            connectionstyle=f"arc3,rad={rad}",  # Изгиб рёбер
        )

    # Добавляем легенду для типов рёбер
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=edge_color_map["sea"], lw=2, label="Sea"),
        Line2D([0], [0], color=edge_color_map["river"], lw=2, linestyle="--", label="River"),
        Line2D([0], [0], color=edge_color_map["trade"], lw=2, linestyle=":", label="Trade"),
        Line2D([0], [0], color=edge_color_map["mountain"], lw=2, linestyle="-.", label="Mountain"),
        Line2D([0], [0], color=edge_color_map["unknown"], lw=2, label="Unknown"),
    ]
    plt.legend(handles=legend_elements, loc="upper left", fontsize=8)

    plt.title("Heterogeneous MultiGraph Visualization", fontsize=12, fontweight="bold")
    plt.tight_layout()  # Улучшаем отступы
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    graph = load_graph("viking_nodes.csv", "viking_edges.csv")
    draw_graph(graph)