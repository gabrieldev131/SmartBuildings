import pymc as pm
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

# Ignorar avisos do PyMC para manter o terminal limpo
warnings.filterwarnings("ignore")

def extract_all_transitions(csv_path: str):
    """
    Lê o CSV de rastreamento e cria um dicionário de transições globais.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Ficheiro {csv_path} não encontrado. Abortando.")
        return {}

    # Se o CSV for o de teste que só tem 1 câmera, usamos os dados fictícios
    if len(df['camera_id'].unique()) <= 1:
        print("Aviso: Apenas uma câmara encontrada no CSV. Gerando cenário fictício complexo...")
        return gerar_dados_ficticios()

    df = df.sort_values(by=['global_id', 'timestamp_in'])
    df['next_camera'] = df.groupby('global_id')['camera_id'].shift(-1)
    df['next_camera'] = df['next_camera'].fillna('Saida')

    transitions = df.groupby(['camera_id', 'next_camera']).size().reset_index(name='count')
    
    trans_dict = {}
    for _, row in transitions.iterrows():
        origem = row['camera_id']
        destino = row['next_camera']
        trans_dict.setdefault(origem, {})[destino] = row['count']
        
    return trans_dict

def run_mcmc_for_camera(origem: str, destinos_dict: dict):
    """Roda o modelo MCMC e infere probabilidades a posteriori."""
    destinos = list(destinos_dict.keys())
    contagens = np.array(list(destinos_dict.values()))
    
    if len(destinos) == 1:
        return {destinos[0]: 1.0}
        
    print(f"  -> Inferindo (MCMC) para o nó: {origem}...")
    
    with pm.Model() as modelo_transicao:
        probabilidades = pm.Dirichlet('prob_transicao', a=np.ones(len(contagens)))
        obs = pm.Multinomial('obs', n=contagens.sum(), p=probabilidades, observed=contagens)
        trace = pm.sample(draws=1000, tune=500, chains=2, return_inferencedata=True, progressbar=False)
        
    mean_probs = trace.posterior['prob_transicao'].mean(dim=["chain", "draw"]).values
    return dict(zip(destinos, mean_probs))

def export_matrix_to_csv(prob_matrix: dict, output_filename: str = 'matriz_probabilidades.csv'):
    """
    Converte o dicionário de probabilidades em um DataFrame Pandas
    e salva em um arquivo CSV.
    Retorna o DataFrame para ser usado na visualização.
    """
    # Extrair todos os nós únicos para criar as linhas e colunas
    origens = sorted(list(prob_matrix.keys()))
    destinos_set = set()
    for transicoes in prob_matrix.values():
        destinos_set.update(transicoes.keys())
    destinos = sorted(list(destinos_set))

    # Criar um DataFrame vazio (preenchido com 0)
    df_prob = pd.DataFrame(0.0, index=origens, columns=destinos)
    
    # Preencher a tabela com os valores
    for origem, transicoes in prob_matrix.items():
        for destino, prob in transicoes.items():
            df_prob.loc[origem, destino] = prob

    # Salvar em CSV formatando para 4 casas decimais
    df_prob.to_csv(output_filename, float_format='%.4f')
    print(f"\n[SUCESSO] Matriz exportada para o ficheiro: {output_filename}")
    
    return df_prob

def plot_dashboard(prob_matrix: dict, df_prob: pd.DataFrame, save_img_path: str = 'dashboard_previsao.png'):
    """
    Renderiza o Grafo e o Heatmap, e salva a imagem em disco.
    """
    print("Renderizando o Dashboard visual...")
    
    fig, (ax_grafo, ax_heatmap) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1.2, 1]})
    fig.canvas.manager.set_window_title('Análise Preditiva de Fluxo - Markov Chain')

    # ==========================================
    # 1. CONSTRUÇÃO DO GRAFO LIMPO (ESQUERDA)
    # ==========================================
    G = nx.DiGraph()
    for origem, transicoes in prob_matrix.items():
        for destino, prob in transicoes.items():
            if prob > 0.02: # Filtrar ruído
                G.add_edge(origem, destino, weight=prob)
                
    pos = nx.spring_layout(G, seed=42, k=2.5)
    
    node_colors = []
    for node in G.nodes():
        if 'saida' in str(node).lower(): node_colors.append('#ff9999')
        elif 'entrada' in str(node).lower(): node_colors.append('#99ff99')
        else: node_colors.append('#99ccff')

    nx.draw_networkx_nodes(G, pos, node_size=3500, node_color=node_colors, edgecolors='black', linewidths=1.5, ax=ax_grafo)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax_grafo)
    
    edges = G.edges()
    weights = [G[u][v]['weight'] * 6 + 1 for u, v in edges]
    
    nx.draw_networkx_edges(
        G, pos, edgelist=edges, width=weights, ax=ax_grafo,
        arrowstyle='-|>', arrowsize=30, edge_color='#666666', 
        connectionstyle='arc3,rad=0.2', min_source_margin=22, min_target_margin=22
    )
    
    ax_grafo.set_title("Topologia do Fluxo (Direção e Volume)", fontsize=14, fontweight='bold', pad=15)
    ax_grafo.axis("off")
    
    legend_elements = [
        mpatches.Patch(color='#99ff99', label='Entrada'),
        mpatches.Patch(color='#99ccff', label='Câmaras'),
        mpatches.Patch(color='#ff9999', label='Saída')
    ]
    ax_grafo.legend(handles=legend_elements, loc='upper left')

    # ==========================================
    # 2. CONSTRUÇÃO DA MATRIZ DE TRANSIÇÃO (DIREITA)
    # ==========================================
    sns.heatmap(
        df_prob, 
        annot=True, fmt=".1%", cmap="Blues", 
        cbar_kws={'label': 'Probabilidade de Transição'},
        linewidths=1, linecolor='white', ax=ax_heatmap
    )
    
    ax_heatmap.set_title("Matriz de Probabilidades (Markov Chain)", fontsize=14, fontweight='bold', pad=15)
    ax_heatmap.set_ylabel("ESTÁ AQUI (Origem)", fontsize=11, fontweight='bold')
    ax_heatmap.set_xlabel("VAI PARA (Destino)", fontsize=11, fontweight='bold')
    
    plt.setp(ax_heatmap.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax_heatmap.get_yticklabels(), rotation=0)

    plt.tight_layout()
    
    # Salvar a imagem antes de mostrar na tela
    plt.savefig(save_img_path, dpi=300, bbox_inches='tight')
    print(f"[SUCESSO] Imagem do Dashboard salva em: {os.path.abspath(save_img_path)}")
    
    # Exibir a janela interativa
    plt.show()

def gerar_dados_ficticios():
    return {
        'Cam_Entrada': {'Cam_Corredor1': 200, 'Cam_Corredor2': 80},
        'Cam_Corredor1': {'Cam_Refeitorio': 100, 'Cam_Recepcao': 60, 'Saida_Norte': 40},
        'Cam_Corredor2': {'Cam_Recepcao': 70, 'Saida_Sul': 10},
        'Cam_Refeitorio': {'Cam_Corredor1': 30, 'Saida_Norte': 70}, 
        'Cam_Recepcao': {'Saida_Principal': 130}
    }

def main():
    csv_path = 'tracking_data_final.csv'
    
    print("1. Processando logs de rastreamento...")
    observacoes = extract_all_transitions(csv_path)
    if not observacoes: return
        
    print("\n2. Processando Markov Chain Monte Carlo (MCMC)...")
    matriz_probabilidades = {}
    for origem, destinos_obs in observacoes.items():
        probs_inferidas = run_mcmc_for_camera(origem, destinos_obs)
        matriz_probabilidades[origem] = probs_inferidas
            
    print("\n3. Gerando ficheiros de saída...")
    # Exporta para CSV e guarda o DataFrame gerado
    df_prob = export_matrix_to_csv(matriz_probabilidades, 'matriz_probabilidades.csv')
    
    # Gera a visualização e salva a imagem
    plot_dashboard(matriz_probabilidades, df_prob, 'dashboard_previsao.png')

if __name__ == "__main__":
    main()