# James Keenan
# INST414
# Medium Post 4

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

chess_df = pd.read_csv('games.csv',usecols=['id', 'opening_eco', 'opening_name', 'victory_status', 'winner', 'turns'])
chess_df['index'] = pd.factorize(chess_df['id'])[0]
chess_df.set_index('index')
chess_df.drop('id', axis=1)

kmeans = KMeans(n_clusters=8)
chess_df_encoded = pd.get_dummies(chess_df)
cluster_labels = kmeans.fit_predict(chess_df_encoded)

chess_df['cluster'] = cluster_labels

# print(chess_df[:5])

grouped = chess_df.groupby('cluster')
print(list(grouped)[:5])

print("Number of chess games per cluster:")
print(chess_df['cluster'].value_counts())
for cluster, group in grouped:
    print(f"\nCluster {cluster}:\n")
    sample_games = group.sample(n=5).index
    for game in sample_games:
        print(game[str('index'),'opening_name', 'victory_status', 'winner', str('turns')])