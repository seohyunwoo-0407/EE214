import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

def vec_vis_agglocluster(x, y_clusters, n_clusters, method, title_suffix=""):
    """Agglomerative clustering 시각화 함수"""
    plt.rcParams['figure.figsize'] = [10, 8]
    
    xs = x[:,0]
    ys = x[:,1]
    
    plt.figure()
    plt.title(f"Agglomerative Clustering Results with linkage={method}{title_suffix}")
    scatter = plt.scatter(xs, ys, c=y_clusters, cmap=plt.get_cmap('tab10', n_clusters), alpha=0.7)
    legend = plt.legend(*scatter.legend_elements(), loc='upper right', title='Clusters')
    plt.xlabel('Feature 1' if 'Raw' in title_suffix else 'PC1')
    plt.ylabel('Feature 2' if 'Raw' in title_suffix else 'PC2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def compare_pca_vs_raw():
    """PCA 데이터와 Raw 데이터 시각화 비교"""
    
    # 데이터 로딩
    print("=== PCA vs Raw 데이터 시각화 비교 ===")
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
    
    print(f"데이터 정보:")
    print(f"- 원본 데이터 shape: {X_raw.shape}")
    print(f"- PCA 데이터 shape: {X_pca.shape}")
    print(f"- PC1 분산 설명률: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"- PC2 분산 설명률: {pca.explained_variance_ratio_[1]:.3f}")
    
    # 동일한 클러스터링 수행
    n_clusters = 4
    method = 'ward'
    
    # PCA 데이터로 클러스터링
    agg_pca = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    clusters_pca = agg_pca.fit_predict(X_pca)
    
    # Raw 데이터로 클러스터링 (표준화된 것 사용)
    agg_raw = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    clusters_raw = agg_raw.fit_predict(X_scaled)
    
    print(f"\n클러스터링 완료 (n_clusters={n_clusters}, method={method})")
    
    # 비교 시각화
    print("\n1. PCA 데이터 시각화 (PC1 vs PC2):")
    vec_vis_agglocluster(X_pca, clusters_pca, n_clusters, method, " (PCA Data)")
    
    print("\n2. Raw 데이터 시각화 (Feature 0 vs Feature 1):")
    vec_vis_agglocluster(X_scaled, clusters_raw, n_clusters, method, " (Raw Data)")
    
    # 상세 비교 - 4개 subplot으로 한번에 보기
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # PCA 데이터 (PC1 vs PC2)
    axes[0,0].scatter(X_pca[:,0], X_pca[:,1], c=clusters_pca, cmap='tab10', alpha=0.7)
    axes[0,0].set_title('PCA Data: PC1 vs PC2 (Clustered)')
    axes[0,0].set_xlabel(f'PC1 (Var: {pca.explained_variance_ratio_[0]:.3f})')
    axes[0,0].set_ylabel(f'PC2 (Var: {pca.explained_variance_ratio_[1]:.3f})')
    axes[0,0].grid(True, alpha=0.3)
    
    # PCA 데이터 (실제 레이블)
    axes[0,1].scatter(X_pca[:,0], X_pca[:,1], c=y_true, cmap='tab10', alpha=0.7)
    axes[0,1].set_title('PCA Data: PC1 vs PC2 (True Labels)')
    axes[0,1].set_xlabel(f'PC1 (Var: {pca.explained_variance_ratio_[0]:.3f})')
    axes[0,1].set_ylabel(f'PC2 (Var: {pca.explained_variance_ratio_[1]:.3f})')
    axes[0,1].grid(True, alpha=0.3)
    
    # Raw 데이터 (Feature 0 vs Feature 1)
    axes[1,0].scatter(X_scaled[:,0], X_scaled[:,1], c=clusters_raw, cmap='tab10', alpha=0.7)
    axes[1,0].set_title('Raw Data: Feature 0 vs Feature 1 (Clustered)')
    axes[1,0].set_xlabel('Feature 0 (Standardized)')
    axes[1,0].set_ylabel('Feature 1 (Standardized)')
    axes[1,0].grid(True, alpha=0.3)
    
    # Raw 데이터 (실제 레이블)
    axes[1,1].scatter(X_scaled[:,0], X_scaled[:,1], c=y_true, cmap='tab10', alpha=0.7)
    axes[1,1].set_title('Raw Data: Feature 0 vs Feature 1 (True Labels)')
    axes[1,1].set_xlabel('Feature 0 (Standardized)')
    axes[1,1].set_ylabel('Feature 1 (Standardized)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_vs_raw_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 다른 feature 조합들도 확인
    print("\n3. Raw 데이터의 다른 feature 조합들:")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    feature_pairs = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11)]
    
    for idx, (f1, f2) in enumerate(feature_pairs):
        row = idx // 3
        col = idx % 3
        
        if f2 < X_scaled.shape[1]:  # feature 인덱스가 유효한지 확인
            axes[row,col].scatter(X_scaled[:,f1], X_scaled[:,f2], c=clusters_raw, cmap='tab10', alpha=0.7)
            axes[row,col].set_title(f'Raw Data: Feature {f1} vs Feature {f2}')
            axes[row,col].set_xlabel(f'Feature {f1}')
            axes[row,col].set_ylabel(f'Feature {f2}')
            axes[row,col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('raw_feature_combinations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 결론
    print(f"\n=== 결론 ===")
    print(f"1. PCA 데이터 시각화:")
    print(f"   - PC1, PC2는 데이터의 주요 분산 방향")
    print(f"   - 클러스터 구조가 명확하게 보임")
    print(f"   - 총 분산의 {pca.explained_variance_ratio_[:2].sum():.1%} 설명")
    
    print(f"\n2. Raw 데이터 시각화:")
    print(f"   - 임의의 2개 특성만 사용")
    print(f"   - 클러스터 구조가 잘 보이지 않을 수 있음")
    print(f"   - Feature 조합에 따라 결과가 천차만별")
    
    print(f"\n3. 권장사항:")
    print(f"   - 고차원 데이터는 PCA 후 시각화")
    print(f"   - Raw 데이터 시각화 시 모든 feature 조합 확인 필요")

if __name__ == "__main__":
    compare_pca_vs_raw() 