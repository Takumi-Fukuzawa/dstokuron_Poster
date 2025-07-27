# ライブラリのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# データの読み込み
df_workers = pd.read_csv("mleague_data_β.csv", index_col=0)

# 変数の標準化
df_workers_std = df_workers.apply(lambda x: (x-x.mean())/x.std(), axis=0)

# 主成分分析の実行
pca = PCA()
pca.fit(df_workers_std)

# データを主成分に変換
pca_row = pca.transform(df_workers_std)

# 寄与率を求める
pca_col = ["PC{}".format(x + 1) for x in range(len(df_workers_std.columns))]
df_con_ratio = pd.DataFrame([pca.explained_variance_ratio_], columns = pca_col)
print(df_con_ratio.head())

# 累積寄与率を図示する
cum_con_ratio = np.hstack([0, pca.explained_variance_ratio_]).cumsum()
plt.plot(cum_con_ratio, 'D-')
plt.xticks(range(19))
plt.yticks(np.arange(0,1.05,0.05))
plt.grid()
plt.show()

# 主成分負荷量を求める
df_pca = pd.DataFrame(pca_row, columns = pca_col)
df_pca_vec = pd.DataFrame(pca.components_, columns=df_workers.columns, index=["PC{}".format(x + 1) for x in range(len(df_pca.columns))])
print(df_pca_vec)

# 主成分負荷量を図示する
plt.figure(figsize=(6, 6))
for x, y, name in zip(pca.components_[0], pca.components_[1], df_workers.columns[0:]):
    plt.text(x, y, name)
plt.scatter(pca.components_[0], pca.components_[1])
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# 主成分得点を求める
plt.figure(figsize=(6, 6))
plt.scatter(pca_row[:, 0], pca_row[:, 1], alpha=0.8)
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
x = pca_row[:, 0]
y = pca_row[:, 1]
annotations = df_workers_std.index
for i, label in enumerate(annotations):
    plt.annotate(label, (x[i], y[i]))
plt.show()

