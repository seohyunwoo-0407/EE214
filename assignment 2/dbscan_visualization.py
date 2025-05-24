import matplotlib.pyplot as plt
import numpy as np

def vec_vis_dbscan(x, y_clusters, eps, min_samples):
    """
    DBSCAN 클러스터링 결과를 시각화하는 함수
    
    Parameters:
    - x: 입력 데이터 (PCA 적용된 2D 데이터)
    - y_clusters: DBSCAN 클러스터 라벨 (-1은 noise)
    - eps: DBSCAN eps 파라미터
    - min_samples: DBSCAN min_samples 파라미터
    """
    plt.rcParams['figure.figsize'] = [10, 8]
    
    xs = x[:,0]
    ys = x[:,1]
    
    # 실제 클러스터 수 계산 (noise 제외)
    unique_labels = np.unique(y_clusters)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(y_clusters).count(-1)
    
    plt.figure()
    plt.title(f"DBSCAN Clustering Results\neps={eps}, min_samples={min_samples}\n"
              f"Clusters: {n_clusters}, Noise points: {n_noise}")
    
    # 클러스터 시각화
    scatter = plt.scatter(xs, ys, c=y_clusters, cmap='tab10', alpha=0.7)
    
    # 범례 생성 (noise는 별도 처리)
    legend_elements = []
    for label in unique_labels:
        if label == -1:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor='black', markersize=8, 
                                            label='Noise', alpha=0.7))
        else:
            color = plt.cm.tab10(label / 10.0)  # tab10 colormap 사용
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=8, 
                                            label=f'Cluster {label}', alpha=0.7))
    
    plt.legend(handles=legend_elements, loc='upper right', title='Clusters')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# DBSCAN 사용 예시
def dbscan_example():
    """DBSCAN 클러스터링 예시 코드"""
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler
    
    # 샘플 데이터 생성
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                          random_state=42, cluster_std=0.60)
    
    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # DBSCAN 클러스터링
    eps = 0.3
    min_samples = 10
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    y_clusters = dbscan.fit_predict(X_scaled)
    
    # 시각화
    vec_vis_dbscan(X_scaled, y_clusters, eps, min_samples)
    
    # 결과 출력
    unique_labels = np.unique(y_clusters)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(y_clusters).count(-1)
    
    print(f"DBSCAN 결과:")
    print(f"- eps: {eps}")
    print(f"- min_samples: {min_samples}")
    print(f"- 발견된 클러스터 수: {n_clusters}")
    print(f"- Noise 포인트 수: {n_noise}")
    print(f"- 클러스터 라벨: {unique_labels}")

# 더 간단한 버전 (기존 함수와 유사한 형태)
def vec_vis_dbscan_simple(x, y_clusters, eps, min_samples=None):
    """
    간단한 DBSCAN 시각화 함수 (기존 함수와 유사한 형태)
    """
    plt.rcParams['figure.figsize'] = [10, 8]
    
    xs = x[:,0]
    ys = x[:,1]
    
    # 실제 클러스터 수 계산
    n_clusters = len(np.unique(y_clusters)) - (1 if -1 in y_clusters else 0)
    n_noise = list(y_clusters).count(-1)
    
    plt.figure()
    
    # 제목 설정
    if min_samples is not None:
        title = f"DBSCAN Clustering Results (eps={eps}, min_samples={min_samples})\n"
    else:
        title = f"DBSCAN Clustering Results (eps={eps})\n"
    title += f"Clusters: {n_clusters}, Noise: {n_noise}"
    
    plt.title(title)
    scatter = plt.scatter(xs, ys, c=y_clusters, cmap='tab10', alpha=0.7)
    legend = plt.legend(*scatter.legend_elements(), loc='upper right', title='Clusters')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 예시 실행
    dbscan_example() 