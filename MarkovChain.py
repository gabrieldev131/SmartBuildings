# MarkovChain.py
import os
import warnings
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

warnings.filterwarnings("ignore")


def _filter_passage_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "event_type" in df.columns:
        passages = df[df["event_type"] == "CAMERA_PASSAGE"].copy()
        if not passages.empty:
            return passages
    return df


def _count_transitions_from_df(df: pd.DataFrame) -> dict:
    df = df.sort_values(by=["global_id", "timestamp_in"])
    df["next_camera"] = df.groupby("global_id")["camera_id"].shift(-1).fillna("Saida")
    transitions = df.groupby(["camera_id", "next_camera"]).size().reset_index(name="count")
    trans_dict: dict = {}
    for _, row in transitions.iterrows():
        trans_dict.setdefault(row["camera_id"], {})[row["next_camera"]] = row["count"]
    return trans_dict


def extract_all_transitions(csv_path: str) -> dict:
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Ficheiro {csv_path} não encontrado. Abortando.")
        return {}
    df = _filter_passage_dataframe(df)
    if df.empty or len(df["camera_id"].unique()) <= 1:
        print("Usando cenário fictício para demonstração de múltiplas câmaras...")
        return gerar_dados_ficticios()
    return _count_transitions_from_df(df)


def run_mcmc_for_camera(origem: str, destinos_dict: dict) -> dict:
    destinos = list(destinos_dict.keys())
    contagens = np.array(list(destinos_dict.values()))
    if len(destinos) == 1:
        return {destinos[0]: 1.0}
    print(f"  -> Inferindo (MCMC) para o nó: {origem}...")
    with pm.Model():
        probs = pm.Dirichlet("prob_transicao", a=np.ones(len(contagens)))
        pm.Multinomial("obs", n=contagens.sum(), p=probs, observed=contagens)
        trace = pm.sample(draws=1000, tune=500, chains=2, return_inferencedata=True, progressbar=False)
    mean_probs = trace.posterior["prob_transicao"].mean(dim=["chain", "draw"]).values
    return dict(zip(destinos, mean_probs))


def export_matrix_to_csv(prob_matrix: dict, output_filename: str = "matriz_probabilidades.csv") -> pd.DataFrame:
    origens = sorted(prob_matrix.keys())
    destinos = sorted({d for trans in prob_matrix.values() for d in trans.keys()})
    df_prob = pd.DataFrame(0.0, index=origens, columns=destinos)
    for origem, transicoes in prob_matrix.items():
        for destino, prob in transicoes.items():
            df_prob.loc[origem, destino] = prob
    df_prob.to_csv(output_filename, float_format="%.4f")
    print(f"\n[SUCESSO] Matriz exportada para: {output_filename}")
    return df_prob


def _build_flow_graph(prob_matrix: dict) -> tuple[nx.DiGraph, list[str]]:
    graph = nx.DiGraph()
    for origem, trans in prob_matrix.items():
        for destino, prob in trans.items():
            if prob > 0.02:
                graph.add_edge(origem, destino, weight=prob)
    colors = ["#ff9999" if "saida" in str(n).lower() else ("#99ff99" if "entrada" in str(n).lower() else "#99ccff") for n in graph.nodes()]
    return graph, colors


def _draw_flow_graph(ax, prob_matrix: dict) -> None:
    graph, colors = _build_flow_graph(prob_matrix)
    pos = nx.spring_layout(graph, seed=42, k=2.5)
    nx.draw_networkx_nodes(graph, pos, node_size=3500, node_color=colors, edgecolors="black", linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold", ax=ax)
    weights = [graph[u][v]["weight"] * 6 + 1 for u, v in graph.edges()]
    nx.draw_networkx_edges(graph, pos, edgelist=graph.edges(), width=weights, ax=ax, arrowstyle="-|>", arrowsize=30, edge_color="#666666", connectionstyle="arc3,rad=0.2", min_source_margin=22, min_target_margin=22)
    ax.set_title("Topologia do Fluxo (Direção e Volume)", fontsize=14, fontweight="bold", pad=15)
    ax.axis("off")
    ax.legend(handles=[mpatches.Patch(color="#99ff99", label="Entrada"), mpatches.Patch(color="#99ccff", label="Câmaras"), mpatches.Patch(color="#ff9999", label="Saída")], loc="upper left")


def _draw_transition_heatmap(ax, df_prob: pd.DataFrame) -> None:
    sns.heatmap(df_prob, annot=True, fmt=".1%", cmap="Blues", cbar_kws={"label": "Probabilidade de Transição"}, linewidths=1, linecolor="white", ax=ax)
    ax.set_title("Matriz de Probabilidades (Markov Chain)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("ESTÁ AQUI (Origem)", fontsize=11, fontweight="bold")
    ax.set_xlabel("VAI PARA (Destino)", fontsize=11, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)


def plot_dashboard(prob_matrix: dict, df_prob: pd.DataFrame, save_img_path: str = "dashboard_previsao.png") -> None:
    fig, (ax_grafo, ax_heatmap) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.canvas.manager.set_window_title("Análise Preditiva de Fluxo - Markov Chain")
    _draw_flow_graph(ax_grafo, prob_matrix)
    _draw_transition_heatmap(ax_heatmap, df_prob)
    plt.tight_layout()
    plt.savefig(save_img_path, dpi=300, bbox_inches="tight")
    print(f"[SUCESSO] Imagem do Dashboard salva em: {os.path.abspath(save_img_path)}")
    plt.show()


def gerar_dados_ficticios() -> dict:
    return {
        "Cam_Entrada": {"Cam_Corredor1": 200, "Cam_Corredor2": 80},
        "Cam_Corredor1": {"Cam_Refeitorio": 100, "Cam_Recepcao": 60, "Saida_Norte": 40},
        "Cam_Corredor2": {"Cam_Recepcao": 70, "Saida_Sul": 10},
        "Cam_Refeitorio": {"Cam_Corredor1": 30, "Saida_Norte": 70},
        "Cam_Recepcao": {"Saida_Principal": 130},
    }


def main() -> None:
    observacoes = extract_all_transitions("tracking_data_final.csv")
    if not observacoes:
        return
    matriz = {origem: run_mcmc_for_camera(origem, dests) for origem, dests in observacoes.items()}
    df_prob = export_matrix_to_csv(matriz, "matriz_probabilidades.csv")
    plot_dashboard(matriz, df_prob, "dashboard_previsao.png")


if __name__ == "__main__":
    main()
