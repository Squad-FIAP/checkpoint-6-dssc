import streamlit as st
import pandas as pd

# Configuração da Página
st.set_page_config(page_title="Checkpoint 6 - 2ESPR", layout="wide")

# Barra lateral com informações
st.sidebar.markdown("""🧑‍💻 Desenvolvido por:
- Gabriel Mediotti - [Github](https://github.com/mediotti)
- Jó Sales - [Github](https://github.com/Josales9)
- Miguel Garcez de Carvalho - [Github](https://github.com/MiguelGarcez)
- Vinicius Souza e Silva - [Github](https://github.com/Vinissil)
""")

# Introdução
st.markdown("## Checkpoint 6 - 2ESPR")
st.markdown("### Introdução ao Problema")
st.write(
    """
    O mercado de Inteligência Artificial (AI), Machine Learning (ML) e Data Science (DS) está crescendo rapidamente,
    e entender as tendências salariais é crucial tanto para profissionais quanto para empresas.
    Este dashboard busca analisar a distribuição de salários desses profissionais ao longo dos anos,
    levando em conta fatores como experiência, tipo de contrato, localização e modelo de trabalho (remoto ou presencial).
    """
)

# Apresentação do Dataset
st.markdown("## Apresentação do Dataset")

st.write(
    """
    O conjunto de dados utilizado contém informações sobre salários de profissionais de **AI, Machine Learning e Data Science** 
    entre **2020 e 2025**. Ele é atualizado semanalmente e apresenta salários convertidos para **USD**, 
    ajustados com a taxa de câmbio média do respectivo ano.

    O dataset inclui as seguintes variáveis:
    - **Ano do pagamento** (`work_year`)
    - **Nível de experiência** (`experience_level`): Junior (EN), Pleno (MI), Sênior (SE), Executivo (EX)
    - **Tipo de emprego** (`employment_type`): Tempo integral (FT), Parcial (PT), Contrato (CT), Freelancer (FL)
    - **Título do cargo** (`job_title`)
    - **Salário bruto** (`salary`)
    - **Moeda original do salário** (`salary_currency`)
    - **Salário convertido para USD** (`salary_in_usd`)
    - **País de residência do funcionário** (`employee_residence`)
    - **Proporção de trabalho remoto** (`remote_ratio`): Presencial (0), Híbrido (50), 100% Remoto (100)
    - **Localização da empresa** (`company_location`)
    - **Tamanho da empresa** (`company_size`): Pequena (S), Média (M), Grande (L)

    📌 **Fonte do dataset:** [Kaggle - The Global AI/ML/Data Science Salary for 2025](https://www.kaggle.com/datasets/samithsachidanandan/the-global-ai-ml-data-science-salary-for-2025)
    """
)


st.markdown("#### Exemplo de Dados")
try:
    df = pd.read_csv("src/dataset.csv")
    st.dataframe(df.head())
except:
    st.warning("⚠️ Dataset não encontrado. Certifique-se de fazer o upload do arquivo.")

# Hipóteses e Perguntas
st.markdown("### Hipóteses e Perguntas Investigativas")
st.write(
    """
    Algumas perguntas que iremos explorar utilizando Intervalos de Confiança e Testes de Hipótese:
    
    - **Intervalos de Confiança:**  
      - Qual é o intervalo de confiança para o salário médio dos profissionais de AI/ML/DS, para uma determinada senioridade?
      - Há uma diferença significativa no intervalo de confiança do salário médio entre diferentes níveis de experiência?  
      - O intervalo de confiança para o salário médio de profissionais remotos é maior ou menor do que para os presenciais?  

    - **Testes de Hipótese:**  
      - O salário médio de profissionais com nível "Senior" é significativamente maior do que o de "Mid-level"?  
      - Existe uma diferença estatisticamente significativa entre os salários médios de empresas pequenas, médias e grandes?  
      - Profissionais que trabalham remotamente ganham salários significativamente diferentes dos que trabalham presencialmente?  
    """
)

# Estrutura Inicial do Dashboard
st.markdown("### Estrutura Inicial do Dashboard")
st.write(
    """
    O dashboard será dividido em:
    1. **Exploração dos Dados** – Estatísticas descritivas, gráficos de distribuição salarial.
    2. **Intervalos de Confiança** – Cálculo e visualização de intervalos para diferentes grupos.
    3. **Testes de Hipótese** – Análises estatísticas para validar hipóteses levantadas.
    4. **Conclusões e Insights** – Resumo dos achados mais importantes.
    """
)

# Finalização
st.markdown("🚀 Vamos explorar os dados e gerar insights valiosos!")

