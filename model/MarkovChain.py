import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

def main():
    # 1. OS SEUS DADOS (Extraídos do GlobalIdentityManager)
    transicoes_observadas_A = np.array([500, 300, 200])

    print("Construindo o modelo de transição MCMC para a Câmera A...")
    with pm.Model() as modelo_transicao:
        
        # A PRIORI (Prior): Distribuição de Dirichlet
        probabilidades = pm.Dirichlet('probabilidades_transicao', a=np.array([1, 1, 1]))
        
        # VEROSSIMILHANÇA (Likelihood): Distribuição Multinomial
        observacoes = pm.Multinomial(
            'obs', 
            n=transicoes_observadas_A.sum(), 
            p=probabilidades, 
            observed=transicoes_observadas_A
        )
        
        # AMOSTRAGEM MCMC
        # Ao rodar no Windows, o PyMC agora sabe que deve despachar as 
        # chains (cadeias) em segurança.
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

    rotas = ["A -> B", "A -> C", "A -> Saída"]
    for i, ax in enumerate(axes):
        ax.set_title(f"Probabilidade: {rotas[i]}")

    plt.tight_layout()
    plt.show()

# O "Guardião" necessário para multiprocessamento no Windows
if __name__ == '__main__':
    # Opcional, mas recomendado no Windows para congelar o executável corretamente
    from multiprocessing import freeze_support
    freeze_support() 
    
    main()