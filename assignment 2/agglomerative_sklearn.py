import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from ucimlrepo import fetch_ucirepo

# scipy는 dendrogram 그리기용으로만 사용
from scipy.cluster.hierarchy import dendrogram, linkage

# 데이터 로딩 및 전처리
print("=== Agglomerative Clustering (sklearn 버전) 분석 ===")
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

# PCA 변환
pca = PCA(n_components=0.95, random_state=42)
X_std_pca = pca.fit_transform(X_scaled)  # X_std_pca로 변수명 통일

print(f"전처리 완료:")
print(f"- 원본 데이터: {X_raw.shape}")
print(f"- PCA 데이터: {X_std_pca.shape}")
print(f"- 클래스: {np.unique(y_true)}")

# 1. 다양한 linkage 방법에 대한 분석
linkage_methods = ['ward', 'complete', 'average', 'single']
colors = ['blue', 'red', 'green', 'orange']

print(f"\n1. 다양한 linkage 방법 분석...")

# 큰 figure 생성
fig = plt.figure(figsize=(20, 15))

for idx, method in enumerate(linkage_methods):
    print(f"\n--- {method.upper()} Linkage 분석 ---")
    
    # Linkage 계산 (dendrogram 그리기용)
    linkage_matrix = linkage(X_std_pca, method=method)
    
    # Dendrogram 그리기 (상단)
    plt.subplot(4, 2, idx*2 + 1)
    dendrogram(linkage_matrix, 
              truncate_mode='level', 
              p=8,
              orientation='top',
              leaf_rotation=90,
              color_threshold=0.7*max(linkage_matrix[:,2]))
    
    plt.title(f'{method.capitalize()} Linkage Dendrogram', fontsize=14)
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.grid(True, alpha=0.3)
    
    # Silhouette Score 분석 (하단)
    plt.subplot(4, 2, idx*2 + 2)
    
    # 클러스터 수별 Silhouette Score 계산
    cluster_range = range(2, 11)
    silhouette_scores = []
    
    for n_clusters in cluster_range:
        # sklearn AgglomerativeClustering 사용
        agg_model = AgglomerativeClustering(
            n_clusters=n_clusters, 
            linkage=method
        )
        clusters = agg_model.fit_predict(X_std_pca)
        
        # Silhouette Score 계산
        if len(np.unique(clusters)) > 1:  # 클러스터가 1개 이상일 때만
            sil_score = silhouette_score(X_std_pca, clusters)
            silhouette_scores.append(sil_score)
            print(f"  {n_clusters} clusters: Silhouette = {sil_score:.4f}")
        else:
            silhouette_scores.append(0)
    
    # Silhouette Score 플롯
    plt.plot(cluster_range, silhouette_scores, 'o-', 
             color=colors[idx], linewidth=2, markersize=8, 
             label=f'{method.capitalize()}')
    
    plt.title(f'{method.capitalize()} Silhouette Scores', fontsize=14)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 최적 클러스터 수 찾기
    best_n_clusters = cluster_range[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)
    print(f"  최적 클러스터 수: {best_n_clusters} (Score: {best_score:.4f})")

plt.tight_layout()
plt.savefig('agglomerative_sklearn_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Ward Linkage 상세 분석 (sklearn 버전)
print(f"\n2. Ward Linkage 상세 분석 (sklearn 버전)...")

# 2.1 상세한 Dendrogram (scipy 사용)
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
ward_linkage = linkage(X_std_pca, method='ward')
dendrogram_plot = dendrogram(ward_linkage,
                           leaf_rotation=90,
                           leaf_font_size=10,
                           color_threshold=0.7*max(ward_linkage[:,2]))

plt.title('Ward Linkage Dendrogram (Detailed)', fontsize=16)
plt.xlabel('Sample Index', fontsize=14)
plt.ylabel('Distance', fontsize=14)
plt.grid(True, alpha=0.3)

# 2.2 Silhouette Score와 Inertia 함께 분석 (sklearn 사용)
plt.subplot(1, 2, 2)

cluster_range = range(2, 16)
silhouette_scores = []
inertias = []

for n_clusters in cluster_range:
    # sklearn AgglomerativeClustering 사용
    agg_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    clusters = agg_model.fit_predict(X_std_pca)
    
    # Silhouette Score
    sil_score = silhouette_score(X_std_pca, clusters)
    silhouette_scores.append(sil_score)
    
    # 수동으로 inertia 계산 (클러스터 내 제곱합)
    inertia = 0
    for cluster_id in np.unique(clusters):
        cluster_points = X_std_pca[clusters == cluster_id]
        cluster_center = np.mean(cluster_points, axis=0)
        inertia += np.sum((cluster_points - cluster_center) ** 2)
    
    inertias.append(inertia)
    
    print(f"n_clusters={n_clusters}: Silhouette={sil_score:.4f}, Inertia={inertia:.2f}")

# 이중 y축으로 그리기
ax1 = plt.gca()
color1 = 'tab:blue'
ax1.set_xlabel('Number of Clusters', fontsize=14)
ax1.set_ylabel('Silhouette Score', color=color1, fontsize=14)
line1 = ax1.plot(cluster_range, silhouette_scores, 'o-', color=color1, 
                linewidth=2, markersize=8, label='Silhouette Score')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Inertia (Within-cluster Sum of Squares)', color=color2, fontsize=14)
line2 = ax2.plot(cluster_range, inertias, 's-', color=color2, 
                linewidth=2, markersize=8, label='Inertia')
ax2.tick_params(axis='y', labelcolor=color2)

# 범례 추가
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right')

plt.title('Ward Clustering: Silhouette Score vs Inertia (sklearn)', fontsize=16)
plt.tight_layout()
plt.savefig('ward_sklearn_detailed_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 최적 클러스터 수로 클러스터링 결과 시각화
best_n_clusters = cluster_range[np.argmax(silhouette_scores)]
best_score = max(silhouette_scores)

print(f"\n3. 최적 클러스터링 결과 시각화 (n={best_n_clusters})...")

# sklearn AgglomerativeClustering으로 최적 클러스터링
optimal_agg_model = AgglomerativeClustering(n_clusters=best_n_clusters, linkage='ward')
optimal_clusters = optimal_agg_model.fit_predict(X_std_pca)

plt.figure(figsize=(15, 5))

# 3.1 클러스터링 결과
plt.subplot(1, 3, 1)
scatter1 = plt.scatter(X_std_pca[:, 0], X_std_pca[:, 1], 
                      c=optimal_clusters, cmap='tab10', alpha=0.7)
plt.title(f'AgglomerativeClustering (n={best_n_clusters})\nSilhouette Score: {best_score:.4f}')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(*scatter1.legend_elements(), title='Clusters')
plt.grid(True, alpha=0.3)

# 3.2 실제 레이블
plt.subplot(1, 3, 2)
scatter2 = plt.scatter(X_std_pca[:, 0], X_std_pca[:, 1], 
                      c=y_true, cmap='tab10', alpha=0.7)
plt.title('True Labels')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(*scatter2.legend_elements(), title='True Labels')
plt.grid(True, alpha=0.3)

# 3.3 Silhouette Score 변화
plt.subplot(1, 3, 3)
plt.plot(cluster_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=best_n_clusters, color='red', linestyle='--', 
           label=f'Optimal: n={best_n_clusters}')
plt.title('Silhouette Score vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimal_sklearn_clustering_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 클러스터별 통계
print(f"\n4. 클러스터별 통계:")
print(f"최적 클러스터 수: {best_n_clusters}")
print(f"최고 Silhouette Score: {best_score:.4f}")

unique_clusters, cluster_counts = np.unique(optimal_clusters, return_counts=True)
print(f"\n클러스터별 샘플 수:")
for cluster_id, count in zip(unique_clusters, cluster_counts):
    print(f"  Cluster {cluster_id}: {count} samples ({count/len(optimal_clusters)*100:.1f}%)")

print(f"\n실제 클래스 분포:")
unique_true, true_counts = np.unique(y_true, return_counts=True)
for true_class, count in zip(unique_true, true_counts):
    print(f"  Class {true_class}: {count} samples ({count/len(y_true)*100:.1f}%)")

print(f"\n=== sklearn AgglomerativeClustering 분석 완료 ===")
print("생성된 파일들:")
print("- agglomerative_sklearn_analysis.png: 4가지 linkage 방법 비교 (sklearn 버전)")
print("- ward_sklearn_detailed_analysis.png: Ward 방법 상세 분석 (sklearn 버전)")
print("- optimal_sklearn_clustering_results.png: 최적 클러스터링 결과 (sklearn 버전)") 