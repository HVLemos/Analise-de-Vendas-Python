import os
print(os.getcwd())


import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.express as px



df = pd.read_csv(r'/home/victor/LojaComercial/Vendas.csv', sep=';', encoding='latin1')

print(df.head())
print(df.info())
print(df.describe())

# df = pd.read_csv(r'/home/victor/LojaComercial/Produtos.csv', sep=';', encoding='latin1')


# print(df.head())
# print(df.info())
# print(df.describe())

# df = pd.read_csv(r'/home/victor/LojaComercial/Vendedores.csv', sep=';', encoding='latin1')

# print(df.head())
# print(df.info())
# print(df.describe())

num_cols = [
    'Quantidade', 'Valor_da_Comissao', 'Comissão',
    'Custo_Unitário', 'Custo_Total', 'Custo_(Venda)', 'Custo_Total_(Venda)'
] 

for c in num_cols:
    if c in df.columns:
        df[c] = (
            df[c].astype(str)
                 .str.replace(r'[^\d,.\-]', '', regex=True)   
                 .str.replace('.', '', regex=False)          
                 .str.replace(',', '.', regex=False)
        )          
        df[c] = pd.to_numeric(df[c], errors='coerce') 



plt.boxplot(df['Custo_Total']) 
plt.title('Distribuição de Custo Total')
plt.ylabel('Valor')
plt.show()


plt.scatter(df['Quantidade'], df['Valor_da_Comissao']) 
plt.title('Relação entre Quantidade e Valor da Comissão')
plt.xlabel('Quantidade')
plt.ylabel('Valor da Comissão')
plt.show()


df.to_csv('Vendas_Limpo.csv', index=False) 


X = df[num_cols].values

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

fig = px.scatter(
    df, 
    x='Quantidade', 
    y='Valor_da_Comissao', 
    color='Cluster', 
    hover_data=['Nome_Produto', 'Regiao']
)
fig.update_layout(title='Clusters de Vendas - K-Means (2D Interativo)')
fig.show()

perfil_clusters = df.groupby('Cluster')[num_cols].mean()
print(perfil_clusters)




# --------------------------------------------------------------
# LIMPEZA DE DADOS
# --------------------------------------------------------------
# Aqui eu importei meus arquivos CSV com pandas.
# Tive que colocar "sep=';'" porque o separador era ponto e vírgula
# e "encoding='latin1'" pra corrigir os acentos e caracteres especiais.
#
# Depois eu dei um df.head() e df.info() pra ver se as colunas estavam certinhas.
# A maioria das colunas numéricas veio como texto (por causa de vírgula, R$ etc),
# então eu converti tudo pra float pra poder fazer conta e gráficos.
#
# Também tratei os valores faltando (NaN), ou removendo, ou preenchendo com média.
# No fim, fiquei com uma base limpa, organizada e pronta pra análise.


# --------------------------------------------------------------
# VISUALIZAÇÃO DOS DADOS
# --------------------------------------------------------------
# Usei o Matplotlib pra ver como os dados estavam distribuídos.
# Fiz boxplot pra achar valores fora do padrão (outliers)
# e scatter plot pra ver se tem relação entre variáveis, tipo
# quantidade x valor da comissão.
#
# Depois parti pro Plotly, que é mais bonito e interativo.
# Dá pra dar zoom, girar o gráfico e ver os valores ao passar o mouse.
# Assim fica bem mais fácil entender o comportamento dos dados.


# --------------------------------------------------------------
#  AGRUPAMENTO (K-MEANS)
# --------------------------------------------------------------
# Aqui comecei a parte de machine learning.
# Usei o K-Means pra dividir as vendas em grupos (clusters)
# baseados em características parecidas, tipo quantidade, custo e comissão.
#
# O K-Means escolhe automaticamente pontos parecidos e forma grupos.
# Depois adicionei uma coluna "Cluster" pra saber onde cada linha ficou.
#
# Fiz um gráfico 3D com Plotly mostrando os clusters com cores diferentes.
# Aí dá pra ver bem quais grupos têm valores mais altos, médios ou baixos.
#
# Pra fechar, calculei a média dos números dentro de cada grupo
# e interpretei o "perfil" de cada cluster (ex: clientes grandes, médios e pequenos).
