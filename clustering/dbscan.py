from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import cluster

np.random.seed(1)
noisy_moons = datasets.make_moons(n_samples=1500, noise=.05)

X_train2, T_train2 = noisy_moons

# Scaling data
scaler2 = StandardScaler()
X_train_s2 = scaler2.fit_transform(X_train2)

model2 = cluster.DBSCAN(eps=0.3, min_samples=10)
Y_train2_d = model2.fit_predict(X_train_s2)

model2 = cluster.KMeans(n_clusters=2, random_state=1)
Y_train2_k = model2.fit_predict(X_train_s2)

model2 = cluster.AgglomerativeClustering(n_clusters=2)
Y_train2_a = model2.fit_predict(X_train_s2)

plt.rcParams['figure.figsize'] = [30, 8]
fig = plt.figure()
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

xs = X_train2[:,0]
ys = X_train2[:,1]
ax1.set_title("DBSCAN")
scatter = ax1.scatter(xs, ys, c=Y_train2_d, cmap=plt.get_cmap('rainbow', 5))
legend = ax1.legend(*scatter.legend_elements(), loc='upper right', title='Clusters')
ax2.set_title("K-means Clustering")
scatter = ax2.scatter(xs, ys, c=Y_train2_k, cmap=plt.get_cmap('rainbow', 5))
legend = ax2.legend(*scatter.legend_elements(), loc='upper right', title='Labels')
ax3.set_title("Agglomerative Clustering")
scatter = ax3.scatter(xs, ys, c=Y_train2_a, cmap=plt.get_cmap('rainbow', 5))
legend = ax3.legend(*scatter.legend_elements(), loc='upper right', title='Labels')
plt.savefig('dbscan, agglomerative, kmeans.png')
plt.show()