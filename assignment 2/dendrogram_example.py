import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from ucimlrepo import fetch_ucirepo

# scipy의 hierarchy 모듈 (dendrogram 생성용)
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

# 데이터 로딩 및 전처리
print("데이터 로딩 중...")
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

# PCA 변환 (시각화를 위해 차원 축소)
pca = PCA(n_components=10, random_state=42)  # 10개 성분으로 축소
X_pca = pca.fit_transform(X_scaled)

# 샘플 수가 너무 많으면 일부만 사용 (dendrogram이 너무 복잡해짐)
n_samples = min(100, len(X_pca))  # 최대 100개 샘플만 사용
indices = np.random.choice(len(X_pca), n_samples, replace=False)
X_sample = X_pca[indices]
y_sample = y_true[indices]

print(f"Dendrogram 생성용 샘플 수: {n_samples}")

# 1. 기본 Dendrogram (Ward linkage)
print("\n1. Ward Linkage Dendrogram 생성...")

plt.figure(figsize=(15, 8))
plt.subplot(2, 2, 1)

# Ward linkage 계산
ward_linkage = linkage(X_sample, method='ward')

# Dendrogram 그리기
dendrogram(ward_linkage, 
          truncate_mode='level', 
          p=5,  # 상위 5 레벨만 표시
          orientation='top',
          leaf_rotation=90)
plt.title('Ward Linkage Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')

# 2. Complete Linkage Dendrogram
plt.subplot(2, 2, 2)
complete_linkage = linkage(X_sample, method='complete')
dendrogram(complete_linkage, 
          truncate_mode='level', 
          p=5,
          orientation='top',
          leaf_rotation=90)
plt.title('Complete Linkage Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')

# 3. Average Linkage Dendrogram
plt.subplot(2, 2, 3)
average_linkage = linkage(X_sample, method='average')
dendrogram(average_linkage, 
          truncate_mode='level', 
          p=5,
          orientation='top',
          leaf_rotation=90)
plt.title('Average Linkage Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')

# 4. Single Linkage Dendrogram
plt.subplot(2, 2, 4)
single_linkage = linkage(X_sample, method='single')
dendrogram(single_linkage, 
          truncate_mode='level', 
          p=5,
          orientation='top',
          leaf_rotation=90)
plt.title('Single Linkage Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')

plt.tight_layout()
plt.savefig('dendrograms_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 자세한 Ward Dendrogram (색상 포함)
print("\n2. 상세한 Ward Dendrogram (색상 포함)...")

plt.figure(figsize=(20, 10))

# 색상이 포함된 dendrogram
dendrogram_plot = dendrogram(ward_linkage,
                           leaf_rotation=90,
                           leaf_font_size=8,
                           color_threshold=0.7*max(ward_linkage[:,2]))  # 색상 임계값 설정

plt.title('Detailed Ward Linkage Dendrogram with Colors', fontsize=16)
plt.xlabel('Sample Index', fontsize=14)
plt.ylabel('Distance', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('detailed_dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Dendrogram에서 클러스터 추출
print("\n3. Dendrogram에서 클러스터 추출...")

# 특정 거리에서 클러스터 추출
n_clusters = 4
clusters = fcluster(ward_linkage, n_clusters, criterion='maxclust')

print(f"추출된 클러스터 수: {len(np.unique(clusters))}")
print(f"각 클러스터의 샘플 수:")
unique, counts = np.unique(clusters, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count} samples")

# 7. PCA 공간에서 클러스터 시각화
print("\n4. PCA 공간에서 클러스터 결과 시각화...")

plt.figure(figsize=(12, 5))

# 계층적 클러스터링 결과
plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X_sample[:, 0], X_sample[:, 1], 
                      c=clusters, cmap='tab10', alpha=0.7)
plt.title('Hierarchical Clustering Results')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(*scatter1.legend_elements(), title='Clusters')
plt.grid(True, alpha=0.3)

# 실제 레이블
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X_sample[:, 0], X_sample[:, 1], 
                      c=y_sample, cmap='tab10', alpha=0.7)
plt.title('True Labels')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(*scatter2.legend_elements(), title='True Labels')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('clustering_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. 다양한 거리 임계값에 따른 클러스터 수 분석
print("\n5. 거리 임계값에 따른 클러스터 수 분석...")

distances = np.arange(1, 10, 0.5)
cluster_counts = []

for dist in distances:
    clusters_at_dist = fcluster(ward_linkage, dist, criterion='distance')
    n_clusters_at_dist = len(np.unique(clusters_at_dist))
    cluster_counts.append(n_clusters_at_dist)
    print(f"Distance {dist:.1f}: {n_clusters_at_dist} clusters")

# 클러스터 수 변화 시각화
plt.figure(figsize=(10, 6))
plt.plot(distances, cluster_counts, 'bo-', linewidth=2, markersize=8)
plt.title('Number of Clusters vs Distance Threshold')
plt.xlabel('Distance Threshold')
plt.ylabel('Number of Clusters')
plt.grid(True, alpha=0.3)
plt.savefig('cluster_count_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== Dendrogram 분석 완료 ===")
print("생성된 파일들:")
print("- dendrograms_comparison.png: 4가지 linkage 방법 비교")
print("- detailed_dendrogram.png: 상세한 Ward dendrogram")
print("- clustering_comparison.png: 클러스터링 결과 vs 실제 레이블")
print("- cluster_count_analysis.png: 거리 임계값별 클러스터 수") 