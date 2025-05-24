import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from ucimlrepo import fetch_ucirepo

# 데이터 로딩 및 전처리
derm = fetch_ucirepo(name="Dermatology")
X_raw = derm.data.features.to_numpy()
y_true = derm.data.targets.to_numpy().ravel()

# NaN 제거
nan_mask = ~np.isnan(X_raw).any(axis=1)
X_raw = X_raw[nan_mask]
y_true = y_true[nan_mask]

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# PCA 변환
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 여기서 사용할 데이터 선택 (X_scaled, X_pca 중 하나)
X_std = X_pca  # PCA 데이터 사용

# 수정된 K_means_clustering 함수
def K_means_clustering(n, x_train):
    """K-means 클러스터링 수행"""
    model = KMeans(n_clusters=n, random_state=42, n_init=10)
    Y_train = model.fit_predict(x_train)
    inertia = model.inertia_  # inertia 추가
    return Y_train, inertia  # 두 값 모두 반환

# 시각화 함수 (루프 전에 정의)
def vec_vis_cluster(x, y_clusters, n_clusters):
    plt.rcParams['figure.figsize'] = [10, 8]
    
    xs = x[:,0]
    ys = x[:,1]
    
    plt.figure()
    plt.title(f"{n_clusters}-means Clustering Results Visualization")
    scatter = plt.scatter(xs, ys, c=y_clusters, cmap=plt.get_cmap('tab10', n_clusters), alpha=0.7)
    legend = plt.legend(*scatter.legend_elements(), loc='upper right', title='Clusters')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 클러스터링 수행
sil_score_dictionary = dict()
inertia_list = []

print("Performing clustering analysis...")
for n in range(2, 16):
    print(f"Testing n_clusters = {n}...")
    
    # 수정된 함수 호출 (두 값 반환)
    Y_cluster, inertia = K_means_clustering(n, X_std)
    sil_score = silhouette_score(X_std, Y_cluster)
    
    # 결과 저장
    sil_score_dictionary[n] = sil_score
    inertia_list.append(inertia)
    
    print(f"  Silhouette Score: {sil_score:.4f}")
    print(f"  Inertia: {inertia:.2f}")
    
    # 시각화 (선택적으로 몇 개만)
    if n in [2, 4, 6, 8]:  # 일부만 시각화
        vec_vis_cluster(X_std, Y_cluster, n)

# 결과 요약
print("\n=== 결과 요약 ===")
print("클러스터 수별 Silhouette Score:")
for n, score in sil_score_dictionary.items():
    print(f"  n={n}: {score:.4f}")

print(f"\nInertia 리스트: {inertia_list}")

# 최적 클러스터 수 찾기
best_n = max(sil_score_dictionary, key=sil_score_dictionary.get)
print(f"\n최고 Silhouette Score: n={best_n}, score={sil_score_dictionary[best_n]:.4f}") 