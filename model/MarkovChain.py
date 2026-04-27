import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

def extract_transitions_from_csv(csv_path: str, source_camera: str):
    """
    Lê o CSV gerado pelo sistema de visão e calcula a matriz de transição
    observada a partir de uma câmara específica.
    """
    try:
        # Carrega os dados exportados pelo sistema
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Ficheiro {csv_path} não encontrado. Usando dados fictícios.")
        return np.array([500, 300, 200]), ["A -> B", "A -> C", "A -> Saída"]

    # Ordena cronologicamente por ID e tempo de entrada
    df = df.sort_values(by=['global_id', 'timestamp_in'])

    # Cria uma coluna com a 'próxima câmara' que a pessoa visitou
    df['next_camera'] = df.groupby('global_id')['camera_id'].shift(-1)
    # Se a próxima câmara é nula, significa que a pessoa saiu do sistema (Saída)
    df['next_camera'] = df['next_camera'].fillna('Saida')

    # Filtra apenas os eventos em que a pessoa estava na câmara de origem
    from_source = df[df['camera_id'] == source_camera]
    
    # Conta para onde as pessoas foram
    transition_counts = from_source['next_camera'].value_counts()
    
    rotas = []
    valores = []
    
    for next_cam, count in transition_counts.items():
        rotas.append(f"{source_camera} -> {next_cam}")
        valores.append(count)

    # Evita erros se o sistema rodou muito pouco tempo
    if not valores:
        print(f"Sem dados suficientes para a câmara {source_camera}.")
        return np.array([1, 1, 1]), ["Sem Rota A", "Sem Rota B", "Sem Rota C"]

    return np.array(valores), rotas

def main():
    # 1. OS SEUS DADOS (Agora extraídos automaticamente do CSV real)
    # Substitua "Cam_A" pelo ID real da sua câmara configurada no Kafka
    camara_origem = "Cam_A" 
    
    transicoes_observadas, rotas = extract_transitions_from_csv("dados_rastreamento.csv", camara_origem)
    
    print(f"Transições observadas na {camara_origem}: {dict(zip(rotas, transicoes_observadas))}")
    print("Construindo o modelo de transição MCMC...")
    
    with pm.Model() as modelo_transicao:
        
        # A PRIORI (Prior): Distribuição de Dirichlet (1 para cada rota possível)
        probabilidades = pm.Dirichlet('probabilidades_transicao', a=np.ones(len(transicoes_observadas)))
        
        # VEROSSIMILHANÇA (Likelihood): Distribuição Multinomial
        observacoes = pm.Multinomial(
            'obs', 
            n=transicoes_observadas.sum(), 
            p=probabilidades, 
            observed=transicoes_observadas
        )
        
        # AMOSTRAGEM MCMC
        trace = pm.sample(draws=2000, tune=1000, chains=4, return_inferencedata=True)

    # 2. VISUALIZAÇÃO
    print("Gerando o grafo de densidade das probabilidades de transição...")
    az.style.use("arviz-darkgrid")

    axes = az.plot_posterior(
        trace, 
        var_names=['probabilidades_transicao'], 
        hdi_prob=0.95,
        figsize=(12, 4), 
        color='darkorange'
    )

    # Se retornar apenas 1 eixo (apenas 1 rota possível), transforma num array para o loop não falhar
    if len(rotas) == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.set_title(f"Probabilidade: {rotas[i]}")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support() 
    main()