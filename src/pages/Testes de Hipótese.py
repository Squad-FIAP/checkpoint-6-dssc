import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Testes de Hipótese", layout="wide")
st.title("🔍 Testes de Hipótese - Análise Salarial em AI/ML/DS")

# Carregar o dataset
df = pd.read_csv("src/dataset.csv")

# Introdução
st.markdown("""
Os testes de hipótese são ferramentas estatísticas que nos ajudam a tomar decisões com base em dados.

Neste estudo, vamos aplicar o **teste T de duas amostras** para comparar os salários de diferentes níveis de senioridade (Pleno e Sênior).    
- **Hipótese nula (H₀)**: Não há diferença significativa entre os salários dos diferentes níveis de senioridade.
- **Hipótese alternativa (H₁)**: Há uma diferença significativa entre os salários dos diferentes níveis de senioridade.
""")

# Filtrar os dados para Pleno e Sênior
salarios_pleno = df[df['experience_level'] == 'MI']['salary_in_usd'].dropna()
salarios_senior = df[df['experience_level'] == 'SE']['salary_in_usd'].dropna()

# Exibir as médias salariais
media_pleno = np.mean(salarios_pleno)
media_senior = np.mean(salarios_senior)

st.markdown(f"""
### Salários Médios
- **Pleno**: ${media_pleno:,.2f}
- **Sênior**: ${media_senior:,.2f}
""")

# Teste T de duas amostras
t_stat, p_val = stats.ttest_ind(salarios_pleno, salarios_senior, equal_var=False)

# Exibir os resultados do teste
st.markdown(f"""
### Resultados do Teste T
- Estatística t: {t_stat:.2f}
- p-valor: {p_val:.4f}
""")

# Nível de significância
alpha = st.slider("Escolha o nível de significância (%)", min_value=1, max_value=10, value=5) / 100

if p_val < alpha:
    st.success(f"Rejeitamos H₀ ao nível de significância de {alpha*100:.0f}%. Existe diferença significativa entre os salários de Pleno e Sênior.")
else:
    st.info(f"Não rejeitamos H₀ ao nível de significância de {alpha*100:.0f}%. Não foi encontrada diferença significativa entre os salários de Pleno e Sênior.")

st.markdown("""
    Outra hipótese é em relação a proporção de profissionais Sênior que trabalham remoto. Ela é maior que 50%?
        - H₀: p = 0,5
        - H₁: p > 0,5
""")

# Filtrar os dados para Sêniores que trabalham remoto
remote_seniors = len(df[(df['experience_level'] == 'SE') & (df['remote_ratio'] == 100)])
total_seniors = len(df[df['experience_level'] == 'SE'])

# Calcular a proporção
proporcao_remote = remote_seniors / total_seniors if total_seniors > 0 else 0
st.markdown(f"""
### Proporção de Sêniores que trabalham remoto
- Total de Sêniores: {total_seniors}
- Total de Sêniores Remotos: {remote_seniors}
- Proporção: {proporcao_remote:.2%}
""")

# Teste de Proporção
z = (proporcao_remote - 0.5) / np.sqrt((0.5 * (1 - 0.5)) / total_seniors)
p_val_proporcao = 1 - stats.norm.cdf(z)
st.markdown(f"""
### Resultados do Teste de Proporção
- Estatística z: {z:.2f}
- p-valor: {p_val_proporcao:.4f}
""")

# Exibir o resultado do teste de proporção
if p_val_proporcao < alpha:
    st.success(f"Rejeitamos H₀ ao nível de significância de {alpha*100:.0f}%. A proporção de Sêniores que trabalham remoto é maior que 50%.")
else:
    st.info(f"Não rejeitamos H₀ ao nível de significância de {alpha*100:.0f}%. A proporção de Sêniores que trabalham remoto não é maior que 50%.")




