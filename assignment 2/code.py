import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from ucimlrepo import fetch_ucirepo
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터 로딩
print("=== Assignment 2 Part 2: Clustering Analysis ===")
print("\n1. Loading Dermatology Dataset...")

# Dermatology 데이터셋 로딩
derm = fetch_ucirepo(name="Dermatology")
X_raw = derm.data.features.to_numpy()
y_true = derm.data.targets.to_numpy().ravel()

print(f"Original data shape: {X_raw.shape}")
print(f"Original unique classes: {np.unique(y_true)}")
print(f"Original class distribution:")
unique, counts = np.unique(y_true, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count} samples")

# NaN 값을 가진 행 제거
print(f"\nChecking for NaN values...")
nan_mask = ~np.isnan(X_raw).any(axis=1)
print(f"Rows with NaN: {(~nan_mask).sum()}")
print(f"Rows without NaN: {nan_mask.sum()}")

X_raw = X_raw[nan_mask]
y_true = y_true[nan_mask]
print(f"Data shape after removing NaN: {X_raw.shape}")

print(f"Classes after removing NaN: {np.unique(y_true)}")
print(f"Class distribution after removing NaN:")
unique, counts = np.unique(y_true, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count} samples")

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
print(f"Data standardized.")

print("\n2. Loading Unknown Dataset...")
# 미지의 데이터셋 로딩 (나중에 제공될 예정)
# unknown_data = pd.read_csv('unknown_dataset.csv')
print("Unknown dataset will be loaded later...")

# 3. PCA 차원 축소 (95% 분산 보존)
print("\n3. Applying PCA for dimensionality reduction...")
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA reduced data shape: {X_pca.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

# PCA 분산 분석
print(f"\n3.2. PCA Variance Analysis...")
print("Explained variance ratio for each PC:")
for i, ratio in enumerate(pca.explained_variance_ratio_[:5]):  # 상위 5개 PC만 출력
    print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")

# 첫 번째 주성분에서 가장 기여도가 높은 특성
print(f"\nMost important features in PC1:")
pc1_components = pca.components_[0]
pc1_abs = np.abs(pc1_components)
top_5_features = np.argsort(pc1_abs)[::-1][:5]

for i, feature_idx in enumerate(top_5_features):
    contribution = pc1_components[feature_idx]
    print(f"  {i+1}. Feature {feature_idx}: {contribution:.4f} (abs: {pc1_abs[feature_idx]:.4f})")

# 모든 주성분을 고려한 특성 중요도
feature_importance = np.sum(np.abs(pca.components_), axis=0)
top_features_overall = np.argsort(feature_importance)[::-1][:5]

print(f"\nOverall feature importance (across all PCs):")
for i, feature_idx in enumerate(top_features_overall):
    print(f"  {i+1}. Feature {feature_idx}: {feature_importance[feature_idx]:.4f}")

# PCA 시각화
print("\n3.1. Visualizing PCA Results...")
plt.rcParams['figure.figsize'] = [10, 8]
xs = X_pca[:,0]
ys = X_pca[:,1]

# Label encoding for visualization
label_encoder = LabelEncoder()
numerical_labels = label_encoder.fit_transform(y_true)

print(f"\nLabel encoding results:")
print(f"Original labels: {np.unique(y_true)}")
print(f"Encoded labels: {np.unique(numerical_labels)}")
print(f"Label mapping:")
for original, encoded in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"  Original {original} → Encoded {encoded}")

scatter = plt.scatter(xs, ys, c=numerical_labels, cmap='tab10', alpha=0.7)
plt.xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.3f})')
plt.ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.3f})')
plt.title('PCA Visualization of Dermatology Dataset')

legend1 = plt.legend(*scatter.legend_elements(), loc="upper right", title="True Labels")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. AutoEncoder 구현
print("\n4. Implementing AutoEncoder...")

class SimpleAutoEncoder:
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        # 중간 차원 계산
        hidden_dim = (input_dim + encoding_dim) // 2
        
        # Encoder: input_dim → hidden_dim → encoding_dim
        self.encoder = MLPRegressor(
            hidden_layer_sizes=(hidden_dim,), 
            activation='tanh',
            solver='adam',
            max_iter=2000,
            random_state=42,
            alpha=0.01
        )
        
        # Decoder: encoding_dim → hidden_dim → input_dim  
        self.decoder = MLPRegressor(
            hidden_layer_sizes=(hidden_dim,),
            activation='tanh', 
            solver='adam',
            max_iter=2000,
            random_state=42,
            alpha=0.01
        )
        
    def fit(self, X):
        # 더 나은 AutoEncoder 구현을 위해 PCA 기반 접근법 사용
        pca_encoder = PCA(n_components=self.encoding_dim, random_state=42)
        encoded = pca_encoder.fit_transform(X)
        
        # Decoder 학습: encoded → original
        self.decoder.fit(encoded, X)
        self.pca_encoder = pca_encoder
        
        # 재구성 오차 계산
        reconstructed = self.decoder.predict(encoded)
        mse = np.mean((X - reconstructed) ** 2)
        print(f"AutoEncoder reconstruction MSE: {mse:.6f}")
        
        return self
        
    def encode(self, X):
        return self.pca_encoder.transform(X)
        
    def decode(self, encoded):
        return self.decoder.predict(encoded)
    
    def reconstruct(self, X):
        encoded = self.encode(X)
        return self.decode(encoded)

# AutoEncoder 학습
autoencoder = SimpleAutoEncoder(input_dim=X_scaled.shape[1], encoding_dim=10)
autoencoder.fit(X_scaled)
X_encoded = autoencoder.encode(X_scaled)
print(f"AutoEncoder latent features shape: {X_encoded.shape}")

# 5. 클러스터링 함수 정의
def perform_clustering(X, data_name, n_clusters_range=range(2, 11)):
    """여러 클러스터링 기법을 적용하고 평가"""
    results = {}
    
    print(f"\n=== Clustering Analysis on {data_name} ===")
    
    # K-Means 클러스터링
    print("Performing K-Means clustering...")
    kmeans_results = {'silhouette': [], 'davies_bouldin': [], 'inertia': []}
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        sil_score = silhouette_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        
        kmeans_results['silhouette'].append(sil_score)
        kmeans_results['davies_bouldin'].append(db_score)
        kmeans_results['inertia'].append(kmeans.inertia_)
        
        print(f"  K={n_clusters}: Silhouette={sil_score:.3f}, Davies-Bouldin={db_score:.3f}")
    
    results['kmeans'] = kmeans_results
    
    # Agglomerative 클러스터링
    print("Performing Agglomerative clustering...")
    agg_results = {'silhouette': [], 'davies_bouldin': []}
    
    for n_clusters in n_clusters_range:
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agg.fit_predict(X)
        
        sil_score = silhouette_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        
        agg_results['silhouette'].append(sil_score)
        agg_results['davies_bouldin'].append(db_score)
        
        print(f"  K={n_clusters}: Silhouette={sil_score:.3f}, Davies-Bouldin={db_score:.3f}")
    
    results['agglomerative'] = agg_results
    
    # DBSCAN 클러스터링 (eps 값 조정)
    print("Performing DBSCAN clustering...")
    eps_values = np.arange(0.3, 2.0, 0.2)
    dbscan_results = {'eps': [], 'n_clusters': [], 'silhouette': [], 'davies_bouldin': []}
    
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        if n_clusters > 1 and n_noise < len(labels) * 0.5:  # Valid clustering
            sil_score = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            
            dbscan_results['eps'].append(eps)
            dbscan_results['n_clusters'].append(n_clusters)
            dbscan_results['silhouette'].append(sil_score)
            dbscan_results['davies_bouldin'].append(db_score)
            
            print(f"  eps={eps:.1f}: Clusters={n_clusters}, Noise={n_noise}, Silhouette={sil_score:.3f}, Davies-Bouldin={db_score:.3f}")
    
    results['dbscan'] = dbscan_results
    
    return results, n_clusters_range

# 6. 원본 특성에 대한 클러스터링
print("\n6. Clustering on Original Features...")
original_results, k_range = perform_clustering(X_scaled, "Original Features")

# 7. PCA 특성에 대한 클러스터링  
print("\n7. Clustering on PCA Features...")
pca_results, _ = perform_clustering(X_pca, "PCA Features")

# 8. AutoEncoder 잠재 특성에 대한 클러스터링
print("\n8. Clustering on AutoEncoder Latent Features...")
ae_results, _ = perform_clustering(X_encoded, "AutoEncoder Features")

print("\n=== Clustering Analysis Complete ===")
print("Results stored for visualization and comparison...")

# 9. 추가 시각화 함수들
def vec_vis(x, y_clusters, n_clusters):
    """클러스터링 결과만 시각화"""
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

def vec_vis_comparison(x, y_clusters, y_true, n_clusters):
    """클러스터링 결과와 실제 레이블을 비교 시각화"""
    plt.rcParams['figure.figsize'] = [20, 8]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    xs = x[:,0]
    ys = x[:,1]
    
    # 클러스터링 결과 시각화
    ax1.set_title("Visualization with Clustering Results")
    scatter1 = ax1.scatter(xs, ys, c=y_clusters, cmap=plt.get_cmap('tab10', n_clusters), alpha=0.7)
    legend1 = ax1.legend(*scatter1.legend_elements(), loc='upper right', title='Predicted Clusters')
    ax1.grid(True, alpha=0.3)
    
    # 실제 레이블 시각화
    ax2.set_title("Visualization with True Labels")
    n_true_classes = len(np.unique(y_true))
    scatter2 = ax2.scatter(xs, ys, c=y_true, cmap=plt.get_cmap('tab10', n_true_classes), alpha=0.7)
    legend2 = ax2.legend(*scatter2.legend_elements(), loc='upper right', title='True Labels')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def K_means_clustering(n, x_train):
    """K-means 클러스터링 수행"""
    model = KMeans(n_clusters=n, random_state=42, n_init=10)
    Y_train = model.fit_predict(x_train)
    return Y_train

# 10. PCA 데이터에 대한 K-means 클러스터링 수행
print("\n10. Performing K-means clustering on PCA data...")
n_clusters = 4  # 클러스터 수 설정
Y_clusters = K_means_clustering(n_clusters, X_pca)

# 클러스터링 평가
sil_score = silhouette_score(X_pca, Y_clusters)
db_score = davies_bouldin_score(X_pca, Y_clusters)

print(f"K-means clustering with {n_clusters} clusters:")
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Score: {db_score:.4f}")

# 결과를 dictionary에 저장
clustering_results = {
    'pca_kmeans_4': {
        'silhouette': sil_score,
        'davies_bouldin': db_score,
        'n_clusters': n_clusters,
        'labels': Y_clusters
    }
}

print(f"\nResults stored in dictionary:")
print(f"Silhouette from dict: {clustering_results['pca_kmeans_4']['silhouette']:.4f}")

# 시각화 (올바른 파라미터 순서로)
print("\n10.1. Visualizing clustering results vs true labels...")
vec_vis(X_pca, Y_clusters, n_clusters)