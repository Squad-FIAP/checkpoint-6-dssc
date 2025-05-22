import streamlit as st
import pandas as pd

df = pd.read_csv("src/dataset.csv")

st.markdown(f"""
# Análise de Correlação e Regressão Linear

## 1. O que é Correlação?
**Correlação** é uma medida estatística que indica o grau de associação entre duas variáveis quantitativas. Ela mostra se, e como, as variáveis estão relacionadas. A correlação pode ser:
- **Positiva**: Quando uma variável aumenta, a outra tende a aumentar.
- **Negativa**: Quando uma variável aumenta, a outra tende a diminuir.
- **Nula**: Não há relação linear entre as variáveis.

O coeficiente de correlação mais comum é o **Coeficiente de Pearson**, que varia entre -1 e 1:
- Próximo de 1: Forte correlação positiva.
- Próximo de -1: Forte correlação negativa.
- Próximo de 0: Correlação fraca ou inexistente.

## 2. O que é Regressão Linear?
A **regressão linear** é uma técnica estatística que busca modelar a relação entre uma variável dependente (por exemplo: `salary_in_usd`) e uma ou mais variáveis independentes (por exemplo: `salary`). Ela permite prever valores da variável dependente a partir dos valores da(s) variável(eis) independente(s).

A equação básica da regressão linear simples estimada nos resultados é:
```python
salary_in_usd = a + b * salary
```
Onde:
- **a**: intercepto da reta (`a`), ou seja, valor esperado de `salary_in_usd` quando `salary = 0`.
- **b**: coeficiente angular (`b`), ou seja, o quanto `salary_in_usd` varia, em média, a cada unidade de aumento em `salary`.

            

### Observações Importantes:
- **Correlação não implica causalidade!** Uma alta correlação não garante que um fator causa o outro.
- Outras variáveis (experiência, país, cargo) podem influenciar a relação. Para análises mais robustas, use regressão múltipla incluindo mais variáveis.
- **Verifique a multicolinearidade** entre as variáveis independentes, pois isso pode afetar a interpretação dos coeficientes.
- **Considere a normalidade dos resíduos** para validar os pressupostos da regressão linear.
- **Ajuste o modelo** conforme necessário, testando diferentes variáveis independentes e interações para melhorar a precisão das previsões.
- **Explore diferentes visualizações** para entender melhor a relação entre as variáveis.
- **Considere a validação cruzada** para avaliar a performance do modelo em diferentes subconjuntos de dados.
""")

# Análise de correlação e regressão entre salary e salary_in_usd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.markdown("---")
st.markdown("## Análise prática: salary vs salary_in_usd")

# Remover dados nulos para evitar erros
df_limpo = df.dropna(subset=["salary", "salary_in_usd"])

# Calcular correlação de Pearson
correlacao = df_limpo["salary"].corr(df_limpo["salary_in_usd"])
st.write(f"**Coeficiente de correlação de Pearson (salary x salary_in_usd):** `{correlacao:.2f}`")

# Ajustar regressão linear
X = df_limpo[["salary"]]
y = df_limpo["salary_in_usd"]
modelo = LinearRegression()
modelo.fit(X, y)
a = modelo.intercept_
b = modelo.coef_[0]
r2 = modelo.score(X, y)
st.write(f"**Equação da reta ajustada:** salary_in_usd = {a:.2f} + {b:.2f} * salary")
st.write(f"**R² do modelo:** `{r2:.2f}`")

# Gráfico de dispersão com reta de regressão (tamanho reduzido)
fig, ax = plt.subplots(figsize=(4, 3))
ax.scatter(df_limpo["salary"], df_limpo["salary_in_usd"], alpha=0.5, label="Dados")
x_vals = np.linspace(df_limpo["salary"].min(), df_limpo["salary"].max(), 100)
y_vals = modelo.predict(x_vals.reshape(-1, 1))
ax.plot(x_vals, y_vals, color="red", label="Regressão Linear")
ax.set_xlabel("salary")
ax.set_ylabel("salary_in_usd")
ax.set_title("Dispersão e Regressão Linear")
ax.legend()
st.pyplot(fig)

# Gráfico de resíduos (tamanho reduzido)
residuos = y - modelo.predict(X)
fig2, ax2 = plt.subplots(figsize=(4, 3))
ax2.scatter(df_limpo["salary"], residuos, color="purple", alpha=0.5)
ax2.axhline(0, color='gray', linestyle='--')
ax2.set_xlabel("salary")
ax2.set_ylabel("Resíduo")
ax2.set_title("Gráfico de Resíduos")
st.pyplot(fig2)

# Conclusão
st.markdown(f"""
## Conclusão

Os resultados mostram um coeficiente de correlação de Pearson de **0.34** e um R² do modelo de apenas **0.12** para a regressão linear entre `salary` e `salary_in_usd`. Esses números indicam uma **correlação positiva, porém fraca** entre as duas variáveis.

Isso significa que, apesar de `salary_in_usd` ser derivado de `salary` (por conversão de moeda), outros fatores — como diferentes moedas, taxas de câmbio variadas e possíveis inconsistências ou outliers nos dados — contribuem para uma relação linear fraca. Isso é evidenciado pelos gráficos:

- **O gráfico de dispersão** mostra grande dispersão dos pontos, com alguns valores extremos de `salary` que não se traduzem proporcionalmente em `salary_in_usd`.
- **O gráfico de resíduos** apresenta padrão não aleatório, sugerindo que o modelo linear não explica adequadamente a variação em `salary_in_usd` com base apenas em `salary`.

Outro ponto importante a ser considerado é que, **para profissionais que atuam nos EUA, os valores de `salary` e `salary_in_usd` são iguais**, pois não há necessidade de conversão cambial. Já para profissionais de outros países, a conversão para dólar americano pode introduzir variações e distorções, impactando a força da correlação observada.

Portanto, **não é possível afirmar que praticamente toda a variação em `salary_in_usd` pode ser explicada por `salary`**. Na prática, a conversão de salários para dólar americano não preserva uma relação linear forte, possivelmente devido à diversidade de moedas, volatilidade cambial e possíveis erros ou outliers no dataset.  
Para uma modelagem mais precisa, seria necessário tratar os outliers, analisar as moedas e considerar outros fatores que impactam a conversão salarial.
""")