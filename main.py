from turtledemo.round_dance import stop

import kagglehub
import pandas as pd
import os

# baixa o dataset
path = kagglehub.dataset_download("shashwatwork/consume-complaints-dataset-fo-nlp")
print("Arquivos disponíveis:", os.listdir(path))
# carrega o arquivo
df = pd.read_csv(path + "/complaints_processed.csv")

# primeiras informações
print("Tamanho:", df.shape)
print("\nColunas:", df.columns.tolist())
print("\nPrimeiras linhas:")
print(df.head())

#mostrar produtos e reclamações
print("Produtos e quantidade de reclamações:")
print(df['product'].value_counts())

#exemplo de reclamação completa
print ("\nExemplo de reclamação:")
print(df['narrative'][0])

#checar se existe texto vazio
print("\nValores vazios por coluna:")
print(df.isnull().sum())

#remover a coluna inutil
df = df.drop(columns=['Unnamed: 0'])
#remover linhas com texto vazio
df = df.dropna(subset=['narrative'])
#verificação
print("Tamanho após limpeza:", df.shape)
print("\nValores vazios:")
print(df.isnull().sum())

from sklearn.feature_extraction.text import TfidfVectorizer

#cria o vetorizador
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
#aplica no texto das reclamações
matriz = tfidf.fit_transform(df['narrative'])
print("Tamanho da matriz TF-IDF:", matriz.shape)
print("\nAlgumas palavras capturadas:")
print(tfidf.get_feature_names_out()[:20])

#nome das 1000 palavras
palavras = tfidf.get_feature_names_out()
#media de tf-idf por categoria
df_tfidf = pd.DataFrame(matriz.toarray(), columns=palavras)
df_tfidf['product'] = df['product'].values

print("Top 10 palavras por categoria de reclamação: \n")
for categoria in df['product'].unique():
    media =df_tfidf[df_tfidf['product'] == categoria].drop(columns='product').mean()
    top10 = media.sort_values(ascending = False).head(10)
    print(f"---{categoria}---")
    print(top10.to_string())
    print()
