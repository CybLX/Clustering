#!/usr/bin/env python
# coding: utf-8

# # Sumário
# 
# 1. [Introdução](#Seção-1:-Introdução)
# 2. [Feature Engineering](#Seção-2:-Feature-Engineering)  
#    2.1 [Statistics](#Seção-21:-Statistics)  
#    2.2 [Principal component analysis (PCA)](#Seção-22:-Principal-component-analysis-(PCA))
# 3. [Algoritmos](#Seção-3:-Algoritmos)  
#    3.1 [K-Means](#Seção-31:-K-Means)  
#    3.2 [DBSCAN](#Seção-32:-DBSCAN)  
#    3.3 [K - NearestNeighbors](#Seção-33:-K---NearestNeighbors)  
#    3.4 [Modelos de Misturas Gaussianas (GMM)](#Seção-34:-Modelos-de-Misturas-Gaussianas-(GMM))  
#    3.5 [Agglomerative Clustering](#Seção-35:-Agglomerative-Clustering)
# 4. [Conclusão](#Seção-4:-Conclusão)

# # Seção 1: Introdução
# 
# A respectiva resenha tem como motivação um conjunto de dados da Copa do Mundo da FIFA de 1930 a 2018 para modelagem. Para caracterizá-las, foram informados 86 seleções diferentes e diversos dados sobre elas. Iremos avaliar os problemas do aprendizado de máquina não supervisionado, trabalhando com conjuntos de dados sem utilizar rótulos ou qualquer tipo de informação sobre as instâncias manipuladas (Zhi e Goldberg, 2009). As principais tarefas aqui apresentadas nesta seção serão: a redução de dimensionalidade e a engenharia de dados.
# 
# 
# ## Seção 1.1: Análise de agrupamento (Clusterização)
# 
# A análise de agrupamento é uma técnica multivariada que, por meio de métodos numéricos, agrupa variáveis em k grupos distintos, onde cada grupo contém n objetos descritos por m características. O objetivo é segmentar os k grupos, com 𝑘 ≪ 𝑛, de forma que os objetos dentro de um mesmo grupo sejam semelhantes entre si e diferentes dos objetos de outros grupos. A medida de dissimilaridade utilizada considera as m características disponíveis no conjunto de dados original para determinar se os objetos são similares ou não (Jain e Dubes, 1988; Tan et al., 2006).
# 
# A escolha dessa medida de dissimilaridade é crucial para a definição adequada dos grupos e envolve a consideração de diversos fatores, como atributos quantitativos, qualitativos nominais e ordinais, ou uma combinação desses tipos. Para atributos quantitativos, muitas medidas de dissimilaridade são baseadas na distância de Minkowski, que serve como base para diversos algoritmos de agrupamento.
# 
# 
# ## Seção 1.2: Distância de Minkowski:
#  Considerando dois objetos $X_i$ e $X_j$, essas distancias sao definidads como:
# $
# \begin{equation}
# d(x_i, x_j) = \left( \sum_{l=1}^{p} |x_{il} - x_{jl}|^p \right)^{\frac{1}{p}}
# \end{equation}
# $
# 
# A escolha do valor de p define variações para essa medida. Dentre elas, os três casos mais
# conhecidos são [Faceli et al. 2011]:
# 
# - Distância de Manhattan ($P=1$)
# - Distância Euclidiana ($P=2$)
# - Distância de Chebyshev ($P=\inf$)
# 
# Importante salientar que não há uma definição universal para a constituição de um grupo (Everitt et al. 2001, Faceli et al. 2011). Dessa forma, cada problema depende do algoritmo e da aplicação estudada. Distintamente de aplicações de Classificação, a Clusterização é uma técnica na qual nenhuma suposição é feita a respeito dos grupos, porém, formalmente devem satisfazer as propriedades de uma partição rígida. Sendo elas:
# 
# Formalmente, dado um conjunto $X = {x_1 , · · · , x_n}$, uma partição rígida consiste em
# uma coleção de subconjuntos $C = {C_1, · · · ,C_k}$, satisfazendo as seguintes propriedades
# [Xu e Wunsch 2005]:
# - $C_1 \cup C_2 \cup \cdots \cup C_k = X$;
# - $C_i \neq \emptyset, \, \forall i \in [1, k]$; e
# - $C_i \cap C_j = \emptyset, \, \forall i, j \in [1, k] \text{ e } i \neq j$.
# 

# In[1]:


from sklearn.metrics import silhouette_score,davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import scale, minmax_scale, MinMaxScaler, LabelEncoder
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as shc
from itertools import combinations
import matplotlib.pyplot as plt
from kneed import KneeLocator
import seaborn as sns
import pandas as pd
import numpy as np
import glob
#https://kneed.readthedocs.io/en/stable/api.html#kneelocator


# In[ ]:


# Download and unzip files
get_ipython().system('wget https://github.com/cmcouto-silva/datasets/raw/main/datasets/fifa-football-world-cup-dataset.zip')
get_ipython().system('unzip fifa-football-world-cup-dataset.zip')


# In[2]:


# Listando arquivos
files = glob.glob('fifa-football-world-cup-dataset/FIFA - *')


# In[3]:


def read_csv(file):
  df = pd.read_csv(file)
  df['Year'] = int(file.split(' ')[-1].split('.')[0])
  return df


_ = [read_csv(file) for file in files]
df_teams = pd.concat(_)
df_teams.head()


# # Seção 2: Feature Engineering
# ## Seção 2.1: Statistics

# In[4]:


# feature engineering

df_teams = (
  df_teams
 .assign(**{
     'Win %': lambda x: x['Win'] / x['Games Played'],
     'Draw %': lambda x: x['Draw'] / x['Games Played'],
     'Loss %': lambda x: x['Loss'] / x['Games Played'],
     'Avg Goals For': lambda x: x['Goals For'] / x['Games Played'],
     'Avg Goals Against': lambda x: x['Goals Against'] / x['Games Played'],
 })
)

df_teams['Rank'] = df_teams.groupby('Year')['Position'].transform(lambda x: 1 - minmax_scale(x))
df_teams['Goal Difference'] = df_teams['Goal Difference'].apply(lambda x: str(x).replace('−', '-').strip()).astype(int)
df_teams.reset_index(inplace = True, drop=True)

# Saving teams code
teams_code = LabelEncoder()
teams_code.fit(df_teams['Team'])
df_teams['Team'] = teams_code.transform(df_teams['Team'])
df_teams = df_teams.astype(float)


# In[5]:


df_teams.info()


# In[6]:


# Agrupando pelos times

df_teams_stats = df_teams.groupby(['Team']).agg(
  n_cups = ('Team', 'count'),
  avg_wins = ('Win %', 'mean'),
  avg_draws = ('Draw %', 'mean'),
  avg_losses = ('Loss %', 'mean'),
  avg_goals_for = ('Avg Goals For', 'mean'),
  avg_goals_against = ('Avg Goals Against', 'mean'),
  avg_rank = ('Rank', 'mean')
)


# In[7]:


df_teams_stats.head()


# In[8]:


# Verificando outliers
df_teams_stats.apply(scale).boxplot()
plt.xticks(rotation=60, ha='right')
plt.show()

# Filtrando outliers por 3 std
for col in df_teams_stats.columns:
  avg,std = df_teams_stats[col].agg(['mean','std'])
  df_teams_stats[col] = df_teams_stats[col].clip(lower=avg-3*std, upper=avg+3*std)


# In[9]:


df_teams_stats.head()


# In[10]:


# Criando um Check Point
teams = df_teams_stats.copy()


# Assim, vamos interpretar nossos dados utilizando o Pandas e aplicar uma normalização dos dados para que seu menor valor fique -1 e maior +1.

# In[11]:


# Normalizando valores para para uma mesma escala (menor valor será -1 e maior 1)
teams = teams.astype(float)
scaler = MinMaxScaler(feature_range=(-1,1))
teams[:] = scaler.fit_transform(teams)



# "Outra técnica […] é calcular a média estatística e o desvio padrão dos valores dos atributos, subtrair a média de cada valor e dividir o resultado pelo desvio padrão. Esse processo é chamado de padronização de uma variável estatística e resulta em um conjunto de valores cuja média é zero e o desvio padrão é um."
# — Página 61, Mineração de Dados: Ferramentas e Técnicas Práticas de Aprendizado de Máquina , 2016.
# 

# In[12]:


#Vamos montar tambem algumas funcoes para visualizacao dos dados e da matriz de correlacao:

def figures(sensores,title, color = ['r', 'g', 'm']):
    coluns = sensores.columns.drop('labels')
    comb_sens = combinations(coluns, 2)
    for comb in comb_sens:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        x = sensores[f'{comb[0]}']
        y = sensores[f'{comb[1]}']
        plt.title(f'{title}')
        sns.scatterplot(data=sensores, x=comb[0],y=comb[1],hue='labels')
        ax.set_xlabel(f'Cluster {comb[0]}')
        ax.set_ylabel(f'Cluster {comb[1]}')
    return plt.show()

def matrix_corr(db, title):
    matriz_Ms = pd.DataFrame(db).corr()

    plt.figure(figsize=(10, 16))
    plt.title(f'{title}', fontsize=16)

    sns.heatmap(matriz_Ms,
                annot=True,           
                cmap='coolwarm',       
                xticklabels=matriz_Ms.columns,
                yticklabels=matriz_Ms.columns)

    plt.tight_layout()  
    plt.show()
    plt.close()

def paises_por_cluster(df):
    table_data = {}
    for label in df['labels'].unique():
        index_list = df[df['labels'] == label]['index'].tolist()
        table_data[label] = index_list
    
    table_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in table_data.items()]))
    
    display(table_df)

def boxplot_paises(df):
    required_columns = ['labels', 'n_cups', 'avg_wins', 'avg_draws', 
                        'avg_losses', 'avg_goals_for', 
                        'avg_goals_against', 'avg_rank', 'index']
    
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Colunas not found {required_columns}")

    sns.set(style="darkgrid", palette="deep")
    plt.style.use('dark_background')

    variables = ['n_cups', 'avg_wins', 'avg_draws', 
                 'avg_losses', 'avg_goals_for', 
                 'avg_goals_against', 'avg_rank']
    
    plt.figure(figsize=(20, 12))
    
    for i, var in enumerate(variables, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(x='labels', y=var, data=df, hue='labels', palette='Set2', legend=False)
        plt.title(f'Boxplot de {var} por Label', fontsize=16, color='white')
        plt.xlabel('Label', fontsize=14, color='white')
        plt.ylabel(var, fontsize=14, color='white')
        plt.xticks(fontsize=12, color='white')
        plt.yticks(fontsize=12, color='white')
        plt.grid(color='grey')
    plt.tight_layout(pad=3.0)
    plt.show()



# A matriz de correlação abaixo quantifica em uma relação linear das colunas entre elas mesmas. O cálculo da correlação é feito usando o coeficiente de Pearson, que varia de -1 a 1. Um valor de 1 indica uma correlação positiva perfeita, -1 uma correlação negativa perfeita, e 0 nenhuma correlação. O resultado gera um matriz de correlação útil para compararmos as características entre si.

# In[13]:


matrix_corr(teams, 'Fifa World Cup Correlation')


# ## Seção 2.2: Principal component analysis (PCA)
# ### Redução da dimensionalidade
# 
# A Análise de Componentes Principais (PCA) é uma técnica eficaz para identificar padrões e diferenças em um conjunto de dados. Ao identificar essas características, conseguimos comprimir os dados em um espaço menor, sem perder uma parte significativa de sua variância. Essencialmente, o PCA projeta os dados em um subespaço formado por autovetores (eixos ortogonais), o que resulta em uma redução linear da dimensionalidade, usando a decomposição de valor singular (SVD). Para isso, utilizamos a implementação LAPACK do SVD completo ou o SVD truncado randomizado, conforme o método de Halko et al. (2009).
# 
# Podemos aplicar o PCA ao conjunto de dados para reduzir o número de variáveis, agrupando-as em componentes mais amplas e significativas. Esses componentes serão ortogonais entre si, preservando até 95% da variância original. Essa técnica é especialmente útil para grandes conjuntos de dados contínuos e multivariados, onde não há uma relação linear evidente entre as variáveis.
# 
# No código a seguir, realizamos testes para reduzir os componentes a até 7 dimensões, analisando a soma acumulada de variância para cada componente. Em seguida, determinamos o número ideal de componentes que melhor preserva 95% da variância e aplicamos a redução para esse número.

# In[14]:


evr = []
for i in range(1,7):
    principal=PCA(n_components=i)
    data = principal.fit(teams)
    evr.append(principal.explained_variance_ratio_.sum()) # Soma da variacia total acumulada para o respectivo n_components

plt.plot(range(1, 7), evr, marker = 'o', linestyle = '--')
plt.title('The Explained Variance Ratio')
plt.xlabel('Number of Dimmensions')
plt.ylabel('Variance Ratio')
plt.show()


# Segue o novo conjunto de vetores para os dados originais:
# Preservando 95% da variância dos dados

# In[15]:


arr = np.abs(np.array(evr)- 0.95) #  Preservar 95% da variancia dos dados
indice = np.where(arr == np.amin(arr)) # Identifica a quantidade n_components
principal=PCA(n_components = indice[0][0])# Reduz a dimenssao dos sensores
vec_ = pd.DataFrame(principal.fit_transform(teams), columns = ['P{}'.format(col) for col in range(0,indice[0][0])]) # Novos vetores
vec_.head()


# Com a redução dos sensores, é possível determinar a qual cluster cada ponto pertence, classificando-os e relacionando-os a cada variável independente. As pontuações recém-obtidas do PCA serão incorporadas aos algoritmos K-Means, K-Nearest Neighbors, DBScan, Agglomerative Clustering e Gaussian mixture models para realizar as segmentações e classificações.

# # Seção 3: Algoritmos
# ## Seção 3.1: K-Means
# 
# O algoritmo K-means agrupa os dados tentando separá-los em n grupos de variância semelhante, minimizando o critério de inércia ou soma dos quadrados (WCSS) para cada solução. Ele divide o conjunto de N amostras em K clusters disjuntos, sendo cada um representado pela média das amostras dentro do cluster, conhecidos como centróides. Esse método é conhecido como Elbow, onde decidimos quantos clusters manter com base na forma da curva.
# 
# A inércia não é uma métrica normalizada; sabemos apenas que valores menores, próximos de zero, são ideais. Em espaços de alta dimensão, as distâncias euclidianas podem se tornar infladas (a chamada maldição da dimensionalidade). Por isso, aplicamos a redução de dimensionalidade utilizando as pontuações do PCA.
# 
# #### Como ele funciona:
# 
# O K-means é frequentemente chamado de algoritmo de Lloyd e opera em três etapas principais.
# 
# Na primeira etapa, são escolhidos os centróides iniciais, selecionando amostras do próprio conjunto de dados. Após essa inicialização, o algoritmo alterna entre as duas etapas seguintes: a atribuição de cada amostra ao centróide mais próximo.
# 
# A segunda etapa, novos centróides são calculados tomando a média de todas as amostras associadas ao centróide anterior. A diferença entre o centróide antigo e o novo é então calculada, e o processo se repete até que essa diferença seja menor que um valor pré-definido, ou seja, até que os centróides não mudem significativamente.
# 
# Após coletar os valores de inércia para o primeiro cluster até o sétimo, calculamos os coeficientes de Silhueta e outras métricas.
# 
# ### Seção 3.1.1: Métricas
# 
# #### silhouette score
# O Coeficiente de Silhueta é calculado com base na distância média dentro do próprio cluster e na distância média até o cluster mais próximo, do qual a amostra não faz parte. Esse coeficiente mede o quão bem uma amostra está associada ao seu próprio cluster em comparação com outros clusters. A fórmula utilizada é:
# $
# \begin{equation}
# \frac{(b - a)}{\max(a, b)}
# \end{equation}
# $
# 
# Onde "a" é a distância média intra-cluster e "b" é a distância até o cluster mais próximo. O coeficiente de silhueta só é definido para:
# 
# 
# $
# \begin{equation}
# 2 \leq n_{\text{labels}} \leq n_{\text{samples}} - 1
# \end{equation}
# $
# 
# 
# O melhor valor é 1, enquanto o pior é -1. Valores próximos de 0 indicam que os clusters estão sobrepostos, e valores negativos sugerem que a amostra foi atribuída ao cluster errado, pois pertence a um cluster mais semelhante.
# 
# https://scikit-learn.org/dev/modules/clustering.html#silhouette-coefficient
# 
# #### Davies Bouldin Score
# 
# O índice de Davies-Bouldin (DB) é uma métrica que avalia a qualidade dos clusters gerados, sendo baseado na comparação entre a distância intra-cluster e a distância entre os centróides dos clusters. Quanto menor o valor do índice, melhor a separação entre os clusters. 
# 
# Uma das vantagens do índice DB é sua simplicidade de cálculo em comparação com o coeficiente de silhueta, pois utiliza apenas as distâncias entre os pontos do dataset. No entanto, ele tende a ter valores mais altos para clusters convexos, sendo menos eficaz para clusters de densidade, como os gerados pelo DBSCAN.
# 
# A formulação matemática do índice DB é dada por:
# 
# $
# \begin{equation}
# DB = \frac{1}{k} \sum_{i=1}^{k} \max_{i \neq j} \left( \frac{S_i + S_j}{d_{ij}} \right)
# \end{equation}
# $
# 
# Onde:
# - $(S_{i})$ é a distância média entre os pontos do cluster $( C_{i} )$ e seu centróide (diâmetro do cluster).
# - $(d_{ij})$ é a distância entre os centróides dos clusters $(C_i)$ e $(C_j)$.
# 
# https://scikit-learn.org/stable/modules/clustering.html#davies-bouldin-index
# 
# #### Calinski Harabasz Score
# 
# O índice de Calinski-Harabasz, também conhecido como Critério de Razão de Variância, avalia a qualidade dos clusters considerando a dispersão entre e dentro dos clusters. Um valor maior indica clusters bem definidos. 
# 
# A fórmula para o índice é dada como a razão entre a dispersão entre clusters e a dispersão dentro dos clusters:
# 
# $
# \begin{equation}
# s = \frac{tr(B_k)}{tr(W_k)} \times \frac{(n_E - k)}{(k - 1)}
# \end{equation}
# $
# 
# Onde:
# 
# $
# \begin{equation}
# W_k = \sum_{q=1}^{k} \sum_{x \in C_q} (x - c_q)(x - c_q)^T
# \end{equation}
# $
# 
# $
# \begin{equation}
# B_k = \sum_{q=1}^{k} n_q (c_q - c_E)(c_q - c_E)^T
# \end{equation}
# $
# 
# Aqui, $(c_{q})$ é o centro do cluster $(q)$, $(c_{E})$ é o centro do conjunto de dados $(E)$, e $(n_{q})$ é o número de pontos no cluster $(q)$.
# 
# https://scikit-learn.org/stable/modules/clustering.html#calinski-harabasz-index

# In[16]:


# Aplicando o algoritmo KMeans para diferentes números de clusters
metrics = silhouette_score, davies_bouldin_score, calinski_harabasz_score
metrics_results = []
wcss = []
for i in range(3, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 200, n_init = 25, tol = 0.001)#, random_state = 50)
    kmeans.fit(vec_)

    results = {'k':i}
    results['wcss'] = kmeans.inertia_
    wcss.append(kmeans.inertia_)
    cluster_labels = kmeans.labels_
    for metric in metrics:
        results[metric.__name__] = metric(vec_, cluster_labels)
    
    metrics_results.append(results)


plt.plot(range(3, 10), wcss, marker = 'o', linestyle = '--')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

pd.DataFrame(metrics_results).set_index('k').style.background_gradient(cmap='Blues')


# In[17]:


(
  pd.DataFrame(metrics_results)
  .set_index('k').style.background_gradient(cmap='Oranges', subset='wcss')
  .highlight_max(subset=['silhouette_score','calinski_harabasz_score'])
  .highlight_min(subset='davies_bouldin_score')
)


# Notamos que o número ideal de clusters, com os melhores scores, varia entre 3 e 4 ao utilizar o algoritmo KMeans. Assim, podemos classificar tanto os vetores PCA quanto os times originais em 3 ou 4 clusters.

# In[18]:


kmeans = KMeans(n_clusters=3, init = 'k-means++', max_iter=500)
kmeans.fit_predict(teams)

# Classificando os vetores PCA
vec_['labels'] = kmeans.labels_
vec_.head()


# In[19]:


# Preparando os dados originais para a classificação
df_teams_stats.index = teams_code.inverse_transform(df_teams_stats.index.astype(int))
df_teams_stats.reset_index(inplace = True, drop = False)


# In[20]:


# Classica os paises em grupos pelo KMeans
df_teams_stats['labels'] = kmeans.labels_
grouped = df_teams_stats.groupby('labels')['index'].apply(list).reset_index()
grouped['count'] = grouped['index'].apply(len)


plt.figure(figsize=(10, 6))
plt.bar(grouped['labels'], grouped['count'], color='black')
plt.title('Número de Países por Cluster')
plt.xlabel('Labels')
plt.ylabel('Número de Países')
plt.xticks(grouped['labels'])
plt.grid(axis='y')
plt.show()


# Observe a correlacao dos vetores PCA com as suas classificacoes

# In[21]:


matrix_corr(vec_, title = 'Cluester KMeans')


# Também podemos observar a correlação dos dados originais com os clusters

# In[22]:


#_ =  pd.concat([teams, vec_], axis = 1)

teams['labels'] = kmeans.labels_
matrix_corr(teams, title = 'Sensores originais')


# Podemos também visualizar a lista de países por cluster e comparar as características de cada um deles.

# In[23]:


boxplot_paises(df_teams_stats)
paises_por_cluster(df_teams_stats)


# #### COMPLEMENTO CÁLCULO EUCLIDIANO DA DISTANCIA ENTRE OS PONTOS.
# 

# In[24]:


# aproximação direta
import math

def calculate_kn_distance(X,k):

    kn_distance = []
    for i in range(len(X)):
        eucl_dist = []
        for j in range(len(X)):
            eucl_dist.append(
                math.sqrt(
                    ((X[i,0] - X[j,0]) ** 2) +
                    ((X[i,1] - X[j,1]) ** 2)))

        eucl_dist.sort()
        kn_distance.append(eucl_dist[k])

    plt.figure(figsize=(8, 5))  
    plt.hist(kn_distance, bins=30)
    plt.ylabel('n')
    plt.xlabel('Epsilon distance')
    plt.title('Histogram of Epsilon Distance')
    plt.tight_layout()
    plt.show()  
    return kn_distance

eps_dist = calculate_kn_distance(vec_.values, 3)


# Podemos também identificar o ponto de joelho, que é definido como o ponto de máxima curvatura em um sistema. Reconhecer essa localização pode ser útil em diversas situações, mas, no contexto de aprendizado de máquina, é particularmente valioso para auxiliar na escolha de um valor apropriado de $k$ na técnica de agrupamento K-means.
# 
# Essa abordagem permite otimizar o número de clusters, aumentando a eficiência e a interpretabilidade dos resultados da análise dos dados. Para isso, utilizaremos um repositório específico para localizar esse ponto.
# 
# Uma vez instanciada, a classe `KneeLocator` tenta encontrar o ponto de curvatura máxima em uma linha. O joelho é acessível através do atributo .knee .

# In[25]:


kneedle = KneeLocator(range(1,len(eps_dist) +1),  #x values
                        eps_dist, # y values
                        S=1.0, #parameter suggested from paper
                        curve="convex", #parameter from figure
                        direction="decreasing") #parameter from figure
print(f"Knee at y = {kneedle.knee_y}")


plt.figure(figsize=(8, 5)) 
kneedle.plot_knee_normalized()
plt.title('Kneedle - Normalized Plot')
plt.tight_layout() 
plt.show()


# ## Seção 3.2: DBSCAN 
# ### Algoritmo de Agrupamento Baseado em Densidade
# 
# O DBScan é um algoritmo de agrupamento que se baseia na densidade dos dados. Para identificar regiões densas no espaço, o DBScan utiliza dois parâmetros principais: MinPts (número mínimo de vizinhos) e Eps (distância mínima entre pontos de clusters distintos). Esses parâmetros são fundamentais para reconhecer os pontos de maior densidade dentro do conjunto de dados.
# 
# Diferentemente de outros algoritmos, o DBScan não fornece um valor ótimo para a distância entre amostras. Para isso, utilizamos o algoritmo NearestNeighbors para determinar esses parâmetros. O DBScan, que significa "Agrupamento Espacial de Aplicações com Ruído Baseado em Densidade", expande clusters a partir de pontos que atendem aos critérios de densidade estabelecidos.
# 
# A ideia central do DBScan é que, para cada cluster, há uma quantidade mínima de pontos dentro de um determinado raio. A densidade de um ponto vizinho p em relação a q é medida pela k-distância, que define o limite que separa o ruído (à esquerda do limiar k) e o cluster (à direita do limiar k). Portanto, o DBScan utiliza valores globais para Eps e MinPts, aplicando os mesmos valores a todos os clusters. Esses parâmetros de densidade do cluster mais "fino" são bons candidatos para especificar os limites abaixo dos quais a densidade não é considerada ruído.
# 
# Para identificar o ruído em nossos dados, podemos calcular diretamente os dois parâmetros do algoritmo: MinPts e Eps. O objetivo é encontrar o maior MinPts e o menor Eps. Para isso, calculamos a distância média intra-cluster e a distância média do cluster mais próximo para cada amostra, utilizando o Coeficiente de Silhueta. Este coeficiente mede a distância entre uma amostra e o cluster mais próximo do qual essa amostra não faz parte.

# In[26]:


# busca por Forca Bruta: Alto nivel de processamento, muito custoso.
min_samples = range(3,50)
eps = np.arange(0.2,5., 0.01)

output = []
for ms in min_samples:
    for ep in eps:
        labels = DBSCAN(min_samples=ms, eps = ep).fit(vec_).labels_
        try:
            score = silhouette_score(vec_, labels)
            output.append((ms, ep, score))
        except:
            pass


min_samples, eps, score = sorted(output, key=lambda x:x[-1])[-1] # ordena do menor para o maior score, selecionando-o
print(f"Best silhouette_score: {score}")
print(f"min_samples: {min_samples}")
print(f"eps: {eps}")
print("*"*20)



# Com isso, estabelecemos a quantidade mínima de pontos necessária para a menor distância entre uma amostra e o cluster do qual ela não pertence. Assim, podemos rotular nossos pontos de acordo com a presença de ruído e o cluster ao qual estão associados.

# In[27]:


labels = DBSCAN(min_samples=18, eps = 1.0600000000000007).fit(vec_).labels_

print(f"Number of clusters totais: {len(set(labels))}")
print(f"Number of clusters, ignoring noise: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"Number of outliers/noise: {list(labels).count(-1)}")
print(f"Silhouette_score: {silhouette_score(teams, labels)}")

vec_['labels'] = labels
figures(vec_, title = 'DBScan Clusters')


# Note que algumas comparações apresentam clusters bem definidos, enquanto outras parecem estar sobrepostos. Podemos então classificar os dados originais.

# In[28]:


# Classica os paises em grupos pelo KMeans
df_teams_stats['labels'] = labels
grouped = df_teams_stats.groupby('labels')['index'].apply(list).reset_index()
grouped['count'] = grouped['index'].apply(len)

# Plotar o gráfico
plt.figure(figsize=(10, 6))
plt.bar(grouped['labels'], grouped['count'], color='skyblue')
plt.title('Número de Países por Cluster')
plt.xlabel('Labels')
plt.ylabel('Número de Países')
plt.xticks(grouped['labels'])
plt.grid(axis='y')

plt.show()


# In[29]:


boxplot_paises(df_teams_stats)
paises_por_cluster(df_teams_stats)


# Podemos também adotar uma abordagem mais heurística utilizando o algoritmo K-Nearest Neighbors para determinar a melhor distância entre vizinhos (eps) com base no ponto de joelho. Esse método reduz o tempo e o custo de processamento necessários para definir os parâmetros do DBSCAN.

# ## Seção 3.3: K - NearestNeighbors
# 
# 
# https://scikit-learn.org/stable/modules/neighbors.html
# 
# O algoritmo K-Nearest Neighbors (KNN) é uma técnica fundamental em aprendizado de máquina, sua ideia central é prever o rótulo de um ponto de dados novo com base na proximidade a um número predefinido de amostras de treinamento. Para isso, o algoritmo encontra os k vizinhos mais próximos e, a partir desses vizinhos, determina a classe a ser atribuída. A distância entre os pontos pode ser calculada utilizando diferentes métricas, sendo a distância Euclidiana uma das mais comuns.
# 
# Matematicamente, a previsão para um novo ponto 𝑥 é feita pela função:
# 
# $
# \begin{equation}
# s = \frac{tr(B_k)}{tr(W_k)} \times \frac{(n_E - k)}{(k - 1)}
# \end{equation}
# $
# 
# onde $𝑦_𝑖$ são os rótulos dos k vizinhos mais próximos de 𝑥. Essa abordagem é considerada um método não paramétrico, pois não assume uma forma funcional específica para a distribuição dos dados, simplesmente "lembra" de todas as amostras de treinamento.
# 
# ### Implementação e Escolha de Parâmetros
# 
# A implementação do KNN em scikit-learn utiliza três algoritmos principais: BallTree, KDTree e um algoritmo de força bruta. Quando o valor padrão `auto` é utilizado, o algoritmo tenta identificar a melhor abordagem com base nos dados de treinamento. A árvore de bola (Ball Tree) é preferível para problemas em alta dimensionalidade, pois particiona os dados em hiperesferas aninhadas, permitindo uma busca mais eficiente. A eficiência na busca por vizinhos é aprimorada pela utilização da desigualdade do triângulo, que reduz o número de candidatos para a busca.
# 
# A escolha do valor ótimo para k deve ser feita com cautela, pois valores altos podem suavizar os limites de decisão, enquanto valores baixos podem ser sensíveis ao ruído. Por outro lado, a distância $\epsilon$ (em algoritmos como DBSCAN) é encontrada no ponto de curvatura máxima da relação entre o número de vizinhos e a distância, podendo ser visualizada graficamente. 
# 
# Em dados não uniformemente amostrados, o classificador `RadiusNeighborsClassifier` pode ser mais eficaz, pois permite que a quantidade de vizinhos utilizados na classificação varie com a densidade local dos dados. A utilização de pesos para os vizinhos, definida pelo parâmetro `weights`, também melhora a precisão das classificações, onde vizinhos mais próximos têm maior influência na decisão final.
# 
# Segue abaixo um exemplo com o nosso conjunto de dados:

# In[30]:


output = []
nn = range(3, 25)

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)

# Loop para diferentes valores de k vizinhos
for k in nn:
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm = 'brute', n_jobs = -1).fit(vec_)
    distances, indices = nbrs.kneighbors(vec_)
    distances = np.sort(distances, axis=0)  # Ordena as distâncias de cada ponto
    distances = sorted(distances[:, k-1], reverse=True)  # Pega a k-ésima distância mais próxima e ordena
    ax.plot(list(range(1, len(distances) + 1)), distances)
    output.append(distances)

plt.title('Distância dos Pontos')
ax.set_xlabel('Quantidade de Pontos')
ax.set_ylabel('Distância Euclidiana')
ax.grid()
plt.show()
#Lets plot the distances of each point in ascending order of the distance, elbow point will give us the samller range for optimal eps value.


# In[31]:


#https://kneed.readthedocs.io/en/stable/api.html#kneelocator
neighbors = []
for d in output:
    kneedle = KneeLocator(range(1,len(d) +1),  #x values
                          d, # y values
                          S=1.0, #parameter suggested from paper
                          curve="convex", #parameter from figure
                          direction="decreasing") #parameter from figure
    neighbors.append(kneedle.knee_y)

mean = sum(neighbors)/len(neighbors)
arr = np.abs(np.array(neighbors)- mean)
indice = np.where(arr == np.amin(arr))

print(f'The average maximum distance between two neighboring samples is: {mean}')
print(f'The number of samples in a neighborhood: {indice[0]}')


# In[32]:


kneedle = KneeLocator(range(1,len(output[9]) +1),  #x values
                    output[9], # y values
                    S=1.0, #parameter suggested from paper
                    curve="convex", #parameter from figure
                    direction="decreasing") #parameter from figure
plt.figure(figsize=(8, 5)) 
kneedle.plot_knee_normalized()
plt.title('Kneedle - Normalized Plot')
plt.tight_layout() 
plt.show()


# Com os parâmetros `min_samples` e `eps` definidos pelo algoritmo NearestNeighbors, podemos aplicar o DBSCAN para classificar os clusters, identificando regiões densas e expandindo os grupos a partir dos pontos centrais, ou "núcleos".
# 
# Essa abordagem não apenas permite a identificação dos clusters existentes, mas também distingue os pontos classificados como ruído.

# In[33]:


labels = DBSCAN(min_samples=9, eps = 1.1722356349857794).fit(teams).labels_

print(f"Number of clusters totais: {len(set(labels))}")
print(f"Number of clusters, ignoring noise: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"Number of outliers/noise: {list(labels).count(-1)}")


# Podemos mais uma vez efetuar a classificação nos dados originais.
# 

# In[34]:


# Classica os paises em grupos pelo NearestNeighbors e DBScan
df_teams_stats['labels'] = labels
grouped = df_teams_stats.groupby('labels')['index'].apply(list).reset_index()
grouped['count'] = grouped['index'].apply(len)

# Plotar o gráfico
plt.figure(figsize=(10, 6))
plt.bar(grouped['labels'], grouped['count'], color='skyblue')
plt.title('Número de Países por Cluster')
plt.xlabel('Labels')
plt.ylabel('Número de Países')
plt.xticks(grouped['labels'])
plt.grid(axis='y')

plt.show()


# In[35]:


vec_['labels'] = labels
figures(vec_, title = 'DBScan Clusters')


# In[36]:


boxplot_paises(df_teams_stats)
paises_por_cluster(df_teams_stats)


# ## Seção 3.4: Modelos de Misturas Gaussianas (GMM)
# 
# Um Modelo de Misturas Gaussianas (GMM) é um modelo estocástico que representa dados como uma combinação de múltiplas distribuições gaussianas. No contexto do reconhecimento de locutor, cada classe pode ser interpretada como uma unidade acústica ou um locutor, onde a informação temporal não é considerada. O objetivo principal é calcular a probabilidade a posteriori de uma observação $o$ dada uma classe $C_i$, utilizando a fórmula de Bayes:
# 
# $
# \begin{equation}
# P(C_i | o) = \frac{P(o | C_i) \cdot P(C_i)}{P(o)}
# \end{equation}
# $
# 
# onde $P(o | C_{i})$ é a probabilidade condicional de observar $o$ dado que ele pertence à classe $C_{i}$ e $P(C_{i})$ é a probabilidade a priori da classe $C_{i}$.
# 
# Para um conjunto de $T$ vetores acústicos observados, onde cada vetor é considerado um evento independente, podemos maximizar a probabilidade a posteriori ao utilizar a seguinte regra de decisão:
# 
# $
# \begin{equation}
# \alpha_{k} = \arg\max_{1 \leq i \leq M} \prod_{t = 1}^T  P(o_t | C_i) P(C_i)
# \end{equation}
# $
# 
# Neste caso, o termo $P(o)$ do denominador da equação anterior pode ser omitido, pois ele é constante para todas as classes testadas. A modelagem das funções densidade de probabilidade (fdps) dos vetores acústicos é crucial para o reconhecimento de locutor, sendo realizada através de algoritmos de re-estimação de parâmetros, como o algoritmo de Baum-Welch, também conhecido como algoritmo Forward-Backward.
# 
# Os modelos GMM são formalizados pela equação:
# 
# $
# \begin{equation}
# p(o_t | C_j) = p(o_t | \lambda_j) = \sum_{i=1}^{I} c_i \cdot N(o_t; \mu_i, \Sigma_i)
# \end{equation}
# $
# 
# onde $I$ é o número de componentes gaussianas na mistura, $c_i$ representa o peso de cada gaussiana, e $N(o_t; \mu_i, \Sigma_i)$ denota a função de densidade de probabilidade da gaussiana multivariada com média $\mu_i$ e matriz de covariância $\Sigma_i$. 
# 
# Assim, a probabilidade condicional $p(o_t | C_j)$ é calculada, sendo utilizada no processo de decisão anteriormente mencionado, onde a probabilidade a priori reflete a frequência de ocorrência de cada classe $C_i$ na base de dados.

# In[37]:


# GMM aplicada ao nosso caso
results = []
k_range = range(3,50)
covariance_types = ['full', 'tied', 'diag', 'spherical']

for n_components in k_range:
  for covariance_type in covariance_types:
    mclust = GaussianMixture(n_components=n_components, warm_start=True, covariance_type=covariance_type)
    mclust.fit(vec_)
    results.append({
      'bic': mclust.bic(vec_),
      'n_components': n_components,
      'covariance_type': covariance_type,
    })

results = pd.DataFrame(results)

sns.lineplot(data=results, x='n_components', y='bic', hue='covariance_type', marker='o', alpha=.8)
plt.title('BIC Scores by Number of Components and Covariance Type')
plt.xlabel('Number of Components')
plt.ylabel('BIC Score')
plt.legend(title='Covariance Type')
plt.grid()
plt.show()


# Com o número de componentes $n_{components}$ definido e o Critério de Informação Bayesiana (BIC) calculado, podemos identificar o melhor ajuste para o modelo de Mistura Bayesiana (BMM). O BIC é uma métrica que avalia a qualidade de um modelo estatístico, levando em consideração sua complexidade e evitando o overfitting. Sua fórmula é expressa como:
# 
# $
# \begin{equation}
# BIC = -2 * \log(L) + k * \log(n)
# \end{equation}
# $
# 
# Onde $L$ representa a verossimilhança do modelo, 𝑘 é o número de parâmetros e n é o número de observações. 
# 
# Um valor menor de BIC indica um modelo preferível, sugerindo uma melhor adequação aos dados sem introduzir complexidade desnecessária. Ao comparar diferentes modelos ou números de componentes em uma mistura gaussiana, o BIC serve como uma ferramenta fundamental para determinar a escolha mais apropriada para os dados, possibilitando uma avaliação quantitativa do ajuste desse tipo de modelo.

# In[38]:


results.sort_values('bic').head()


# Com $n_{components}$ e o tipo de covariância definidos, podemos classificar nossos dados originais utilizando os clusters calculados.

# In[39]:


model = GaussianMixture(n_components=4, covariance_type='diag')
model.fit(teams)
labels = model.predict(teams)
vec_['labels'] = labels
df_teams_stats['labels'] = labels


# In[40]:


# A quantidade de times por clusters
grouped = df_teams_stats.groupby('labels')['index'].apply(list).reset_index()
grouped['count'] = grouped['index'].apply(len)

# Plotar o gráfico
plt.figure(figsize=(10, 6))
plt.bar(grouped['labels'], grouped['count'], color='skyblue')
plt.title('Número de Países por Cluster')
plt.xlabel('Labels')
plt.ylabel('Número de Países')
plt.xticks(grouped['labels'])
plt.grid(axis='y')

plt.show()


# In[41]:


boxplot_paises(df_teams_stats)
paises_por_cluster(df_teams_stats)


# ## Seção 3.5: Agglomerative Clustering
# 
# O agrupamento hierárquico é uma família de algoritmos que constrói clusters aninhados por meio de fusões ou divisões sucessivas. Essa hierarquia de clusters é representada como uma árvore, ou dendrograma, onde a raiz da árvore representa o único cluster que agrupa todas as amostras e as folhas correspondem aos clusters que contêm apenas uma amostra. 
# 
# O algoritmo de AgglomerativeClustering adota uma abordagem de baixo para cima: cada observação começa em seu próprio cluster e os clusters são sucessivamente mesclados. O critério de ligação determina a métrica utilizada na estratégia de mesclagem, podendo ser:
# 
# - **Ward**: minimiza a soma das diferenças quadradas dentro de todos os clusters, sendo similar à função objetivo do k-means, mas aplicada de forma aglomerativa. Para dois clusters $C_i$ e $C_j$, a distância é dada por:
# 
# $ 
# \begin{equation}
# d(C_i, C_j) = \sum_{x \in C_i} \sum_{y \in C_j} || x - y ||^2 
# \end{equation}
# $
# 
# - **Ligação Máxima (Complete Linkage)**: minimiza a distância máxima entre as observações de pares de clusters.
# 
# $
# \begin{equation}
# d(C_i, C_j) = \max_{x_a \in C_i, x_b \in C_j} || x_a - x_b || 
# \end{equation}
# $
# 
# - **Ligação Média (Average Linkage)**: minimiza a média das distâncias entre todas as observações de pares de clusters.
# 
# $ \begin{equation}
# d(C_i, C_j) = \min_{x_a \in C_i, x_b \in C_j} || x_a - x_b || 
# \end{equation}
# $
# 
# - **Ligação Única (Single Linkage)**: minimiza a distância entre as observações mais próximas de pares de clusters.
# 
# $
# \begin{equation}
# d(C_i, C_j) = \frac{1}{|C_i| \cdot |C_j|} \sum_{x_a \in C_i} \sum_{x_b \in C_j} || x_a - x_b || 
# \end{equation}
# $

# In[42]:


plt.figure(figsize=(10, 7))
plt.title("Dendrograma Ward")
dendrogram = shc.dendrogram(shc.linkage(teams, method='ward', metric='euclidean'))
plt.axhline(y=6, color='r', linestyle='--')
plt.grid(False)
plt.show()


# In[43]:


plt.figure(figsize=(10, 7))
plt.title("Dendrograma Complete Linkage")
dendrogram = shc.dendrogram(shc.linkage(teams, method='complete', metric='euclidean'))
plt.axhline(y=3, color='r', linestyle='--')
plt.grid(False)
plt.show()


# In[44]:


plt.figure(figsize=(10, 7))
plt.title("Dendrograma Average Linkage")
dendrogram = shc.dendrogram(shc.linkage(teams, method='average', metric='euclidean'))
plt.axhline(y=2, color='r', linestyle='--')
plt.grid(False)
plt.show()


# In[45]:


plt.figure(figsize=(10, 7))
plt.title("Dendrograma Single Linkage")
dendrogram = shc.dendrogram(shc.linkage(teams, method='single', metric='euclidean'))
plt.axhline(y=1.20, color='r', linestyle='--')
plt.grid(False)
plt.show()


# Uma das principais vantagens do agrupamento hierárquico é a visualização dos resultados por meio de dendrogramas, que mostram a hierarquia das partições obtidas. No dendrograma, cada fusão de clusters é representada por um link em forma de U, onde a altura do link indica a distância entre os clusters fundidos. Cortes horizontais no dendrograma permitem a identificação de diferentes partições, tornando a interpretação do agrupamento mais intuitiva.

# In[46]:


# Selecionando o corte de 3 clusters e classificando os dados originais
agc = AgglomerativeClustering(n_clusters=3)
labels = agc.fit_predict(teams)


# In[47]:


vec_['labels'] =labels
figures(vec_, title ='AgglomerativeClustering')


# In[48]:


# Classica os paises em grupos pelo AgglomerativeClustering
df_teams_stats['labels'] = labels
grouped = df_teams_stats.groupby('labels')['index'].apply(list).reset_index()
grouped['count'] = grouped['index'].apply(len)

# Plotar o gráfico
plt.figure(figsize=(10, 6))
plt.bar(grouped['labels'], grouped['count'], color='skyblue')
plt.title('Número de Países por Cluster')
plt.xlabel('Labels')
plt.ylabel('Número de Países')
plt.xticks(grouped['labels'])
plt.grid(axis='y')

plt.show()


# In[49]:


boxplot_paises(df_teams_stats)
paises_por_cluster(df_teams_stats)


# # Seção 4: Conclusão
# 
# Neste trabalho, exploramos diversas técnicas de redução de dimensionalidade, começando com a Análise de Componentes Principais (PCA) para minimizar o custo operacional dos algoritmos em cenários com grandes dimensões. Em seguida, aplicamos o algoritmo K-means, que permitiu classificar os dados em três clusters, priorizando a métrica silhouette_score, apesar de outras métricas, como Davies-Bouldin e Calinski-Harabasz, sugerirem a formação de cinco clusters. Sinta-se livre para testar a formação de cinco clusters se desejar.
# 
# Após a aplicação do K-means, investigamos o número mínimo de vizinhos e a distância mínima entre os clusters utilizando o DBScan. A abordagem de força bruta resultou em três clusters com parâmetros de min_samples e eps de 18 e 1.06, respectivamente. O método baseado em NearestNeighbors também concluiu a formação de três clusters, mas com min_samples de 9 e eps de 1.1722, evidenciando a robustez das análises. Adicionalmente, aplicamos o algoritmo Agglomerative Cluster, que também resultou em três clusters. Essa consistência entre os diferentes métodos de agrupamento ressalta a eficácia de nossa abordagem na identificação de padrões nos dados. O uso do agrupamento hierárquico permitiu observar a hierarquia dos clusters, enriquecendo a análise com diversos metodos de mesclagem.
# 
# Por fim, a análise dos gráficos de boxplot revelou uma discrepância significativa entre os clusters no que diz respeito ao número de copas ganhas e suas classificações. Identificamos uma clara correlação entre gols marcados e copas vencidas: o cluster 2 inclui times que sofreram muitos gols e conquistaram poucas copas, enquanto o cluster 0 abrange os times mais vitoriosos, funcionando como um intermediário entre os campeões e aqueles com desempenho inferior. Essa revisão sobre aprendizado não supervisionado, englobando métodos de clusterização como K-means, DBScan e Agglomerative Cluster, proporcionou insights valiosos sobre a estrutura dos dados e suas inter-relações, aprofundando nosso entendimento sobre o desempenho dos times na Copa do Mundo.

## Contato

#**Nome:** Lucas Oliveira Alves
#**Email:** [alves_lucasoliveira@usp.br](mailto:alves_lucasoliveira@usp.br)
#**LinkedIn:** [linkedin.com/in/cyblx](https://www.linkedin.com/in/cyblx/)
#**GitHub:** [github.com/cyblx](https://github.com/cyblx)
