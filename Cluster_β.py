# ライブラリのインポート
import pandas as pd
import matplotlib.pyplot as plt
# fcluster を追加
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# --- 1. データの準備 ---

# データの読み込み
# 'player'列をインデックス（行のラベル）として指定
try:
    # PCAのスコアデータを読み込みます
    player_scores = pd.read_csv("pca_scores_dataΩ.csv", index_col='player')
    print("PCAスコアデータの読み込みに成功しました。")
except FileNotFoundError:
    print("エラー: 'pca_scores_dataγ.csv' が見つかりません。")
    exit()

# --- 2. 階層型クラスター分析の実行 ---

# linkage関数を使い、選手間の距離を計算し、クラスターを形成
# method='ward' は、クラスター内のばらつきを最小化する方法で、一般的に良い結果が得られます
linked = linkage(player_scores, method='ward', metric='euclidean')

# --- 3. 結果の可視化（デンドログラム） ---

# グラフの描画領域を定義
plt.figure(figsize=(16, 12))
plt.title('Player Clustering Dendrogram (based on PCA Scores)', size=15)
plt.xlabel('Player', size=12)
plt.ylabel('Distance (Dissimilarity)', size=12)

# デンドログラムを描画
dendrogram(linked,
           orientation='top',
           labels=player_scores.index,  # 横軸に選手名を表示
           leaf_rotation=90,            # ラベルを90度回転させて見やすくする
           leaf_font_size=10)           # フォントサイズを調整

# 見やすくするためにグリッド線を追加
plt.grid(axis='y', linestyle='--')
plt.tight_layout() # レイアウトを自動調整
plt.show()


# --- ▼▼▼ 追加した機能 ▼▼▼ ---

# 4. 各選手のクラスタリング結果を取得

# デンドログラムを見て、クラスター数を決定（ここでは3つに分ける）
num_clusters = 3
clusters = fcluster(linked, num_clusters, criterion='maxclust')

# 結果を選手名と結合して表示
# 'player_scores.index' を使うことで、正しく選手名を取得します
result_df = pd.DataFrame({'player': player_scores.index, 'cluster_id': clusters})
print(f"\n--- {num_clusters}個のグループに分類した結果 ---")

# クラスターごとに選手を一覧表示
for i in range(1, num_clusters + 1):
    cluster_members = result_df[result_df['cluster_id'] == i]['player'].values
    print(f"\n[クラスター {i}]")
    print(", ".join(cluster_members))

# --- ▲▲▲ 追加ここまで ▲▲▲ ---