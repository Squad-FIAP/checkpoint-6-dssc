import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats

st.set_page_config(page_title="Intervalo de Confiança", layout="wide")

df = pd.read_csv("src/dataset.csv")

media_salarial = df['salary_in_usd'].mean()
desvio_padrao = df['salary_in_usd'].std()

st.markdown(
    r"""
    ## Intervalos de Confiança
    O intervalo de confiança é uma ferramenta estatística que fornece uma faixa de valores dentro da qual
    podemos esperar que um parâmetro populacional (como a média) esteja localizado, com um certo nível de confiança.
    """
)

st.markdown(
    f"""
    - Média Salarial Anual: **{media_salarial:.2f}** USD/ano  
    - Média Salarial Mensal: **{media_salarial/12:.2f}** USD/mês
    
    > _Calculada a partir de **{len(df)} observações** com **{df['salary_in_usd'].isnull().sum()} valores ausentes** na coluna `salary_in_usd`._
    """
) 

st.markdown("### Distribuição Normal")

# 🎯 Histograma geral
mu = media_salarial
sigma = desvio_padrao

fig, ax = plt.subplots(figsize=(15, 10))
sns.histplot(data=df['salary_in_usd']/1000, bins=80, kde=True, stat='probability', ax=ax)

ax.set_xlabel('Salário Anual (K USD)')
ax.set_ylabel('Probabilidade')
ax.set_title('Distribuição Salarial de Profissionais em AI/ML/DS')
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}K'))

st.pyplot(fig)

# 🔍 Intervalo de Confiança para Júnior
st.markdown("### Intervalo de Confiança para Profissionais Júnior")

salarios = df[df["experience_level"] == "EN"]['salary_in_usd'].dropna()/1000

conf = st.slider("Escolha o nível de confiança (%)", min_value=80, max_value=99, value=95)
alpha = 1 - (conf / 100)

n = len(salarios)
media = np.mean(salarios)
desvio_amostral = np.std(salarios, ddof=1)
graus_de_liberdade = n - 1
t_critico = stats.t.ppf(1 - alpha/2, df=graus_de_liberdade)
margem_erro = t_critico * (desvio_amostral / np.sqrt(n))
lim_inf = media - margem_erro
lim_sup = media + margem_erro

st.markdown(
    r"""
    #### Fórmulas usadas:
    - Média amostral: $\bar{x} = \frac{\sum x_i}{n}$
    - Erro padrão: $SE = \frac{s}{\sqrt{n}}$
    - Intervalo de confiança (com distribuição t):  
      $\bar{x} \pm t_{\alpha/2, \, df} \cdot \frac{s}{\sqrt{n}}$
    """
)

fig, ax = plt.subplots(figsize=(15, 10))
sns.histplot(salarios, bins=40, kde=True, stat="probability", ax=ax)
ax.axvspan(lim_inf, lim_sup, color='skyblue', alpha=0.15)
ax.axvline(media, color='blue', linestyle='--', label=f'Média: ${media:,.1f}K')
ax.axvline(lim_inf, color='skyblue', linestyle=':', linewidth=2, label='Limite Inferior')
ax.axvline(lim_sup, color='skyblue', linestyle=':', linewidth=2, label='Limite Superior')

ax.set_xlabel("Salário Anual (K USD)")
ax.set_ylabel("Probabilidade")
ax.set_title(f"Distribuição Salarial com Intervalo de Confiança ({conf}%) - Junior")
ax.legend()

st.pyplot(fig)

st.markdown(
    f"""
    Portanto, podemos afirmar que a média salarial anual dos profissionais de senioridade Junior está no intervalo:  
    **({lim_inf:.1f}K, {lim_sup:.1f}K)** com **{conf}% de certeza**.
    """
)

# 📊 Comparação Pleno vs Sênior
st.markdown("### Comparando Intervalos de Confiança: Pleno vs Sênior")

conf_2 = st.slider("Escolha o nível de confiança (%)", min_value=80, max_value=99, value=95, key="conf_2")
alpha_2 = 1 - (conf_2 / 100)

# Separando os dados
salarios_pleno = df[df['experience_level'] == 'MI']['salary_in_usd'].dropna()/1000
salarios_senior = df[df['experience_level'] == 'SE']['salary_in_usd'].dropna()/1000

# Pleno
n_pleno = len(salarios_pleno)
media_pleno = np.mean(salarios_pleno)
std_pleno = np.std(salarios_pleno, ddof=1)
t_pleno = stats.t.ppf(1 - alpha_2/2, df=n_pleno-1)
erro_pleno = t_pleno * (std_pleno / np.sqrt(n_pleno))
lim_inf_pleno = media_pleno - erro_pleno
lim_sup_pleno = media_pleno + erro_pleno

# Sênior
n_senior = len(salarios_senior)
media_senior = np.mean(salarios_senior)
std_senior = np.std(salarios_senior, ddof=1)
t_senior = stats.t.ppf(1 - alpha_2/2, df=n_senior-1)
erro_senior = t_senior * (std_senior / np.sqrt(n_senior))
lim_inf_senior = media_senior - erro_senior
lim_sup_senior = media_senior + erro_senior

st.markdown(
    r"""
    #### Fórmulas usadas:
    Repetimos o mesmo processo anterior para os dois grupos:  

    $$
    \bar{x}_{pleno} \pm t \cdot \frac{s}{\sqrt{n}} \\
    \bar{x}_{senior} \pm t \cdot \frac{s}{\sqrt{n}}
    $$
    """
)

# Plot conjunto
fig2, ax2 = plt.subplots(figsize=(15, 10))

sns.histplot(salarios_pleno, bins=40, kde=True, stat="probability", ax=ax2, color='orange', label='Pleno')
sns.histplot(salarios_senior, bins=40, kde=True, stat="probability", ax=ax2, color='blue', label='Sênior')

# Pleno - intervalo
ax2.axvspan(lim_inf_pleno, lim_sup_pleno, color='orange', alpha=0.15)
ax2.axvline(media_pleno, color='orange', linestyle='--', label=f'Média Pleno: ${media_pleno:,.1f}K')
ax2.axvline(lim_inf_pleno, color='orange', linestyle=':', linewidth=2, label='Limite Inferior Pleno')
ax2.axvline(lim_sup_pleno, color='orange', linestyle=':', linewidth=2, label='Limite Superior Pleno')

# Sênior - intervalo
ax2.axvspan(lim_inf_senior, lim_sup_senior, color='blue', alpha=0.15)
ax2.axvline(media_senior, color='blue', linestyle='--', label=f'Média Sênior: ${media_senior:,.1f}K')
ax2.axvline(lim_inf_senior, color='blue', linestyle=':', linewidth=2, label='Limite Inferior Sênior')
ax2.axvline(lim_sup_senior, color='blue', linestyle=':', linewidth=2, label='Limite Superior Sênior')

ax2.set_xlabel("Salário Anual (K USD)")
ax2.set_ylabel("Probabilidade")
ax2.set_title(f"Comparação de Intervalos de Confiança ({conf_2}%) - Pleno vs Sênior")
ax2.legend()

st.pyplot(fig2)

st.markdown(
    f"""
    Portanto, podemos afirmar que a média salarial anual dos profissionais de senioridade Pleno está no intervalo:
    **({lim_inf_pleno:.1f}K, {lim_sup_pleno:.1f}K)** com **{conf_2}% de certeza**.
    E a média salarial anual dos profissionais de senioridade Sênior está no intervalo:
    **({lim_inf_senior:.1f}K, {lim_sup_senior:.1f}K)** com **{conf_2}% de certeza**.

    As diferenças entre os intervalos são:
    - Pleno: **{lim_inf_pleno:.1f}K** - **{lim_sup_pleno:.1f}K** = **{lim_sup_pleno-lim_inf_pleno:.1f}K**
    - Sênior: **{lim_inf_senior:.1f}K** - **{lim_sup_senior:.1f}K** = **{lim_sup_senior-lim_inf_senior:.1f}K**



    #### Conclusão
    Podemos observar que os intervalos de confiança para os profissionais Pleno e Sênior não apresentam uma mudança significativa.
    No entanto, a média salarial dos profissionais Sêniores é significativamente maior do que a dos Plenos.
    Isso sugere que a experiência e senioridade têm um impacto positivo nos salários, mas não necessariamente uma diferença estatística significativa entre os dois grupos.    
    """
)

st.markdown("### Comparando Intervalos de Confiança: Remoto vs Presencial vs Híbrido")

conf_3 = st.slider("Escolha o nível de confiança (%)", min_value=80, max_value=99, value=95, key="conf_3")
alpha_3 = 1 - (conf_3 / 100)

# Separando os dados
salarios_remoto = df[df['remote_ratio'] == 100]['salary_in_usd'].dropna()/1000
salarios_presencial = df[df['remote_ratio'] == 0]['salary_in_usd'].dropna()/1000
salarios_hibrido = df[df['remote_ratio'] == 50]['salary_in_usd'].dropna()/1000

# Remoto
n_remoto = len(salarios_remoto)
media_remoto = np.mean(salarios_remoto)
std_remoto = np.std(salarios_remoto, ddof=1)
t_remoto = stats.t.ppf(1 - alpha_3/2, df=n_remoto-1)
erro_remoto = t_remoto * (std_remoto / np.sqrt(n_remoto))
lim_inf_remoto = media_remoto - erro_remoto
lim_sup_remoto = media_remoto + erro_remoto

# Presencial
n_presencial = len(salarios_presencial)
media_presencial = np.mean(salarios_presencial)
std_presencial = np.std(salarios_presencial, ddof=1)
t_presencial = stats.t.ppf(1 - alpha_3/2, df=n_presencial-1)
erro_presencial = t_presencial * (std_presencial / np.sqrt(n_presencial))
lim_inf_presencial = media_presencial - erro_presencial
lim_sup_presencial = media_presencial + erro_presencial

# Híbrido
n_hibrido = len(salarios_hibrido)
media_hibrido = np.mean(salarios_hibrido)
std_hibrido = np.std(salarios_hibrido, ddof=1)
t_hibrido = stats.t.ppf(1 - alpha_3/2, df=n_hibrido-1)
erro_hibrido = t_hibrido * (std_hibrido / np.sqrt(n_hibrido))
lim_inf_hibrido = media_hibrido - erro_hibrido
lim_sup_hibrido = media_hibrido + erro_hibrido

# Plot conjunto para Remoto, Presencial e Híbrido
fig3, ax3 = plt.subplots(figsize=(15, 10))

# Plot histograms for each category
sns.histplot(salarios_remoto, bins=40, kde=True, stat="probability", ax=ax3, color='green', label='Remoto')
sns.histplot(salarios_presencial, bins=40, kde=True, stat="probability", ax=ax3, color='red', label='Presencial')
sns.histplot(salarios_hibrido, bins=40, kde=True, stat="probability", ax=ax3, color='purple', label='Híbrido')

# Remoto - intervalo
ax3.axvspan(lim_inf_remoto, lim_sup_remoto, color='green', alpha=0.15)
ax3.axvline(media_remoto, color='green', linestyle='--', label=f'Média Remoto: ${media_remoto:,.1f}K')
ax3.axvline(lim_inf_remoto, color='green', linestyle=':', linewidth=2, label='Limite Inferior Remoto')
ax3.axvline(lim_sup_remoto, color='green', linestyle=':', linewidth=2, label='Limite Superior Remoto')

# Presencial - intervalo
ax3.axvspan(lim_inf_presencial, lim_sup_presencial, color='red', alpha=0.15)
ax3.axvline(media_presencial, color='red', linestyle='--', label=f'Média Presencial: ${media_presencial:,.1f}K')
ax3.axvline(lim_inf_presencial, color='red', linestyle=':', linewidth=2, label='Limite Inferior Presencial')
ax3.axvline(lim_sup_presencial, color='red', linestyle=':', linewidth=2, label='Limite Superior Presencial')

# Híbrido - intervalo
ax3.axvspan(lim_inf_hibrido, lim_sup_hibrido, color='purple', alpha=0.15)
ax3.axvline(media_hibrido, color='purple', linestyle='--', label=f'Média Híbrido: ${media_hibrido:,.1f}K')
ax3.axvline(lim_inf_hibrido, color='purple', linestyle=':', linewidth=2, label='Limite Inferior Híbrido')
ax3.axvline(lim_sup_hibrido, color='purple', linestyle=':', linewidth=2, label='Limite Superior Híbrido')

# Configure plot labels and title
ax3.set_xlabel("Salário Anual (K USD)")
ax3.set_ylabel("Probabilidade")
ax3.set_title(f"Comparação de Intervalos de Confiança ({conf_3}%) - Remoto vs Presencial vs Híbrido")
ax3.legend()

# Display the plot
st.pyplot(fig3)

st.markdown(
    f"""
    Os intervalos de confiança com {conf_3}% de confiança, para os profissionais Remoto, Presencial e Híbrido são:
    - Remoto: **({lim_inf_remoto:.1f}K, {lim_sup_remoto:.1f}K)** = **{lim_sup_remoto-lim_inf_remoto:.1f}K**
    - Presencial: **({lim_inf_presencial:.1f}K, {lim_sup_presencial:.1f}K)** = **{lim_sup_presencial-lim_inf_presencial:.1f}K**
    - Híbrido: **({lim_inf_hibrido:.1f}K, {lim_sup_hibrido:.1f}K)** = **{lim_sup_hibrido-lim_inf_hibrido:.1f}K**

    #### Conclusão
    Podemos observar que os intervalos de confiança para os profissionais em regime Remoto e Presencial são próximos, entretanto o intervalo de confiança para os profissionais Híbridos é significativamente maior.
    Isso sugere que o regime de trabalho Híbrido pode estar associado a uma maior variação salarial, possivelmente devido a fatores como localização geográfica e flexibilidade de trabalho.
    Além disso, a média salarial dos profissionais Remoto é insignificativamente maior do que a dos profissionais Presenciais, o que pode indicar uma equivalência salarial entre os dois regimes.
    """
)