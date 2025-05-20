import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import cluster
from sklearn.preprocessing import StandardScaler

X, T= fetch_openml('mnist_784', version=1, return_X_y=True) #X는 784차원의 픽셀 데이터, T는 0~9사이 숫자 

for i in range(5): #총 5개의 클래스에 대해 
    temp_arr=X[T=='{}'.format(i)] #i번째 클래스에 해당하는 데이터만 필터링 
    temp_arr_T=T[T=='{}'.format(i)]
    try:
        X_part=np.vstack((X_part, temp_arr))
        T_part=np.concatenate((T_part, temp_arr_T))
    except:
        X_part=temp_arr
        T_part=temp_arr_T

np.random.seed(0)

X_train, X_test, T_train, T_test=train_test_split(X_part, T_part, train_size=5000, test_size=1000, shuffle=True)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.manifold import TSNE

model=TSNE(learning_rate=300, random_state=1)
TSNE_X=model.fit_transform(X_train) #784차원을 2차원으로 

# Image plotting function
def img_plt(X_train, Y_train, n):
  X_train_2d = X_train.reshape(X_train.shape[0], 28, 28)
  fig, axes = plt.subplots(n, 10, figsize=(7.5,7.5))
  for j in range(n):
    for i in range(10):
        ax = axes[j, i]
        try:
          ax.imshow(X_train_2d[Y_train==j][i], cmap='gray_r')
          ax.axis('off')
          if i == 0:
            ax.set_title('Cluster: {}'.format(j))
        except:
          ax.axis('off')

  plt.tight_layout()
  plt.savefig('clustering_result_pca.png')
  plt.show()


# Visualization with TSNE
def vec_vis(x, y, T, n):
  plt.rcParams['figure.figsize'] = [20, 8]
  color_num = n
  fig = plt.figure()
  ax1 = fig.add_subplot(1, 2, 1)
  ax2 = fig.add_subplot(1, 2, 2)

  xs = x[:,0]
  ys = x[:,1]
  ax1.set_title("t-SNE Visualization with Clustering")
  scatter = ax1.scatter(xs, ys, c=y, cmap=plt.get_cmap('rainbow', color_num))
  legend = ax1.legend(*scatter.legend_elements(), loc='upper right', title='Clusters')
  ax2.set_title("t-SNE Visualization with True label")
  scatter = ax2.scatter(xs, ys, c=list(map(int, T)), cmap=plt.get_cmap('rainbow', 5))
  legend = ax2.legend(*scatter.legend_elements(), loc='upper right', title='Labels')
  plt.savefig('clustering_result_pca.png')
  plt.show()

n = 5 # number of clusters


model = cluster.KMeans(n_clusters=n, random_state=1) # 'Kmeans' from 'sklearn.cluster', 'n_cluster' default=2
Y_train = model.fit_predict(X_train) # their clustering result using 'X_train_s' will be assigned to 'Y_train'

img_plt(X_train, Y_train, n)

vec_vis(TSNE_X, Y_train, T_train, n)