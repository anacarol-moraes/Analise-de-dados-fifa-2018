import pandas as pd
import matplotlib.pyplot as plt
#carregar dataset
df = pd.read_csv(r"C:\Users\anaca\Downloads\archive\players_18.csv")
#explorando os dados
print(df.head())
print(df.columns)
# Selecionando as colunas relevantes para análise
df_players = df[['short_name','player_positions','overall','potential','age']]
# mostrando os 10 jogadores com maior potencial
print(df_players.sort_values('potential', ascending = False).head(10))
top10 = df_players.sort_values(by=['potential','overall'], ascending=[False,False]).head(10)
# Extraindo a posição principal de cada jogador
df_players ['main_position'] = df_players['player_positions'].str.split(',').str.get(0)
#mostrando a media de overall das posições principais de cada jogador
print(df_players.groupby('main_position')['overall'].mean())
# Calculando a diferença entre potencial e overall
df_players ['diferenca']= df_players ['potential'] -df_players ['overall']
#exibe os dados dos jogadores com mais diferença de overall
print(df_players[['short_name', 'age', 'overall', 'potential', 'diferenca']].sort_values('diferenca', ascending=False).head(10))
plt.barh(top10['short_name'], top10['overall'])
plt.title("Top 10 jogadores de 2018")
plt.xlabel("Eixo X (Overall)")
plt.ylabel('Eixo Y (nome)')
plt.show()