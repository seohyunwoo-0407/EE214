import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from tsne_cluster import vec_vis, img_plt


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

from sklearn import decomposition

pca = decomposition.PCA(n_components=30).fit(X_train)
reduced_X = pca.transform(X_train)

color_num = 5

plt.rcParams['figure.figsize'] = [10, 8]
xs = reduced_X[:,0]
ys = reduced_X[:,1]

scatter = plt.scatter(xs, ys, c=list(map(int, T_train)), cmap=plt.get_cmap('rainbow', color_num))
legend = plt.legend(*scatter.legend_elements(), loc='upper right', title='Labels')

plt.savefig('pca_result.png')
plt.show()

from sklearn.manifold import TSNE

model = TSNE(learning_rate=300, random_state=1)
TSNE_X_red = model.fit_transform(reduced_X)

n=5

model = cluster.KMeans(n_clusters=n, random_state=1)
Y_train = model.fit_predict(reduced_X)

img_plt(X_train, Y_train, n)
vec_vis(TSNE_X_red, Y_train, T_train, n)