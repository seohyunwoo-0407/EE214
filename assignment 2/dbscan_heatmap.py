import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

def create_dbscan_heatmap(X, eps_range, min_samples_range):
    """
    DBSCAN 파라미터 조합에 따른 클러스터 수와 Silhouette score heatmap 생성
    
    Parameters:
    - X: 입력 데이터
    - eps_range: eps 값들의 리스트
    - min_samples_range: min_samples 값들의 리스트
    
    Returns:
    - cluster_matrix: 클러스터 수 매트릭스
    - silhouette_matrix: Silhouette score 매트릭스
    """
    
    # 결과 저장용 매트릭스 초기화
    cluster_matrix = np.zeros((len(min_samples_range), len(eps_range)))
    silhouette_matrix = np.zeros((len(min_samples_range), len(eps_range)))
    
    print("DBSCAN 파라미터 조합 분석 중...")
    
    best_score = -1
    best_eps = None
    best_min_samples = None
    
    for i, min_samples in enumerate(min_samples_range):
        for j, eps in enumerate(eps_range):
            print(f"Testing eps={eps:.1f}, min_samples={min_samples}")
            
            # DBSCAN 클러스터링
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # 클러스터 수 계산 (noise 제외)
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = list(labels).count(-1)
            
            cluster_matrix[i, j] = n_clusters
            
            # Silhouette score 계산 (클러스터가 2개 이상이고 모든 점이 noise가 아닐 때만)
            if n_clusters >= 2 and n_noise < len(labels):
                try:
                    sil_score = silhouette_score(X, labels)
                    silhouette_matrix[i, j] = sil_score
                    
                    # 최적 조합 찾기
                    if sil_score > best_score:
                        best_score = sil_score
                        best_eps = eps
                        best_min_samples = min_samples
                        
                except:
                    silhouette_matrix[i, j] = np.nan
            else:
                silhouette_matrix[i, j] = np.nan
            
            print(f"  Clusters: {n_clusters}, Noise: {n_noise}, Silhouette: {silhouette_matrix[i, j]:.4f}")
    
    return cluster_matrix, silhouette_matrix, best_eps, best_min_samples, best_score

def plot_dbscan_heatmaps(cluster_matrix, silhouette_matrix, eps_range, min_samples_range, 
                        best_eps, best_min_samples, best_score):
    """
    DBSCAN 결과를 heatmap으로 시각화
    """
    
    # 데이터프레임 생성
    cluster_df = pd.DataFrame(cluster_matrix, 
                             index=min_samples_range, 
                             columns=[f"{eps:.1f}" if eps < 1 else f"{eps:.0f}" for eps in eps_range])
    
    silhouette_df = pd.DataFrame(silhouette_matrix, 
                                index=min_samples_range, 
                                columns=[f"{eps:.1f}" if eps < 1 else f"{eps:.0f}" for eps in eps_range])
    
    # Figure 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. 클러스터 수 heatmap
    sns.heatmap(cluster_df, 
                annot=True, 
                fmt='.0f', 
                cmap='Blues', 
                ax=ax1,
                cbar_kws={'label': 'Number of Clusters'})
    
    ax1.set_title('Number of Clusters (DBSCAN)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('eps', fontsize=14)
    ax1.set_ylabel('min_samples', fontsize=14)
    
    # 2. Silhouette score heatmap
    # NaN 값이 있는 경우 mask 사용
    mask = np.isnan(silhouette_matrix)
    
    sns.heatmap(silhouette_df, 
                annot=True, 
                fmt='.2f', 
                cmap='RdYlBu_r',  # 빨강-노랑-파랑 컬러맵
                ax=ax2,
                mask=mask,
                cbar_kws={'label': 'Silhouette Score'})
    
    ax2.set_title('Silhouette Score', fontsize=16, fontweight='bold')
    ax2.set_xlabel('eps', fontsize=14)
    ax2.set_ylabel('min_samples', fontsize=14)
    
    # 최적 조합 정보 추가
    fig.suptitle(f'Best eps, min_samples combination: eps={best_eps}, min_samples={best_min_samples}, silhouette={best_score:.4f}', 
                fontsize=14, y=0.02)
    
    plt.tight_layout()
    plt.savefig('dbscan_parameter_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cluster_df, silhouette_df

def main():
    """메인 함수"""
    
    # 데이터 로딩 및 전처리
    print("=== DBSCAN Parameter Heatmap Analysis ===")
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
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"데이터 전처리 완료:")
    print(f"- 원본 데이터: {X_raw.shape}")
    print(f"- PCA 데이터: {X_pca.shape}")
    
    # 파라미터 범위 설정
    eps_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    min_samples_range = list(range(2, 12))  # 2 ~ 11
    
    print(f"\n파라미터 범위:")
    print(f"- eps: {eps_range}")
    print(f"- min_samples: {min_samples_range}")
    print(f"- 총 조합 수: {len(eps_range) * len(min_samples_range)}")
    
    # DBSCAN 파라미터 조합 분석
    cluster_matrix, silhouette_matrix, best_eps, best_min_samples, best_score = create_dbscan_heatmap(
        X_pca, eps_range, min_samples_range
    )
    
    # Heatmap 시각화
    cluster_df, silhouette_df = plot_dbscan_heatmaps(
        cluster_matrix, silhouette_matrix, eps_range, min_samples_range,
        best_eps, best_min_samples, best_score
    )
    
    # 결과 요약
    print(f"\n=== 분석 결과 요약 ===")
    print(f"최적 파라미터 조합:")
    print(f"- eps: {best_eps}")
    print(f"- min_samples: {best_min_samples}")
    print(f"- Silhouette Score: {best_score:.4f}")
    
    # 최적 조합으로 최종 클러스터링 수행
    print(f"\n최적 조합으로 클러스터링 수행...")
    best_dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    best_labels = best_dbscan.fit_predict(X_pca)
    
    best_n_clusters = len(np.unique(best_labels)) - (1 if -1 in best_labels else 0)
    best_n_noise = list(best_labels).count(-1)
    
    print(f"최적 조합 결과:")
    print(f"- 클러스터 수: {best_n_clusters}")
    print(f"- Noise 포인트: {best_n_noise}")
    print(f"- Noise 비율: {best_n_noise/len(best_labels)*100:.1f}%")
    
    # CSV로 저장
    cluster_df.to_csv('dbscan_cluster_counts.csv')
    silhouette_df.to_csv('dbscan_silhouette_scores.csv')
    
    print(f"\n결과 파일 저장 완료:")
    print(f"- dbscan_parameter_heatmap.png: Heatmap 시각화")
    print(f"- dbscan_cluster_counts.csv: 클러스터 수 매트릭스")
    print(f"- dbscan_silhouette_scores.csv: Silhouette score 매트릭스")
    
    return cluster_df, silhouette_df, best_eps, best_min_samples, best_score

if __name__ == "__main__":
    cluster_df, silhouette_df, best_eps, best_min_samples, best_score = main() 