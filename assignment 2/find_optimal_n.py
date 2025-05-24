import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from true_autoencoder import TrueAutoEncoder

def find_optimal_n_pca(X, threshold=0.95):
    """PCA의 설명된 분산 비율을 사용하여 최적의 n 찾기"""
    # 모든 성분에 대해 PCA 수행
    pca = PCA()
    pca.fit(X)
    
    # 누적 설명된 분산 비율 계산
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    # 임계값을 넘는 최소 차원 수 찾기
    n_components_95 = np.argmax(cumulative_variance_ratio >= threshold) + 1
    
    # 엘보우 플롯
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             pca.explained_variance_ratio_, 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1),
             cumulative_variance_ratio, 'ro-')
    plt.axhline(y=threshold, color='g', linestyle='--',
                label=f'{threshold*100}% threshold')
    plt.axvline(x=n_components_95, color='r', linestyle='--',
                label=f'n={n_components_95}')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance Ratio')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('pca_analysis.png')
    plt.show()
    
    return n_components_95

def analyze_reconstruction_error(X, max_dim=20):
    """재구성 오차 분석을 통한 최적의 n 찾기"""
    reconstruction_errors = []
    
    for n in range(2, max_dim + 1):
        print(f"Testing dimension {n}...")
        autoencoder = TrueAutoEncoder(input_dim=X.shape[1], encoding_dim=n)
        autoencoder.fit(X, method='simultaneous')  # 더 빠른 방법 사용
        
        # 재구성 오차 계산
        reconstructed = autoencoder.reconstruct(X)
        mse = np.mean((X - reconstructed) ** 2)
        reconstruction_errors.append(mse)
    
    # 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, max_dim + 1), reconstruction_errors, 'bo-')
    plt.xlabel('Latent Dimension (n)')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title('Reconstruction Error vs Latent Dimension')
    plt.grid(True)
    plt.savefig('reconstruction_error.png')
    plt.show()
    
    # 엘보우 포인트 찾기 (간단한 방법)
    diffs = np.diff(reconstruction_errors)
    elbow_point = np.argmin(diffs) + 2  # +2는 시작 인덱스가 2이기 때문
    
    return elbow_point

def main():
    # 데이터 로딩
    print("데이터 로딩 중...")
    derm = fetch_ucirepo(name="Dermatology")
    X_raw = derm.data.features.to_numpy()
    
    # NaN 제거
    nan_mask = ~np.isnan(X_raw).any(axis=1)
    X_raw = X_raw[nan_mask]
    
    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    print(f"데이터 shape: {X_scaled.shape}")
    
    # 1. PCA 분석
    print("\n1. PCA 분석 수행 중...")
    n_pca = find_optimal_n_pca(X_scaled)
    print(f"PCA 분석 결과 추천 차원 수: {n_pca}")
    
    # 2. 재구성 오차 분석
    print("\n2. 재구성 오차 분석 수행 중...")
    n_recon = analyze_reconstruction_error(X_scaled, max_dim=20)
    print(f"재구성 오차 분석 결과 추천 차원 수: {n_recon}")
    
    # 최종 추천
    print("\n=== 최종 추천 ===")
    print(f"PCA 기반 추천 n: {n_pca}")
    print(f"재구성 오차 기반 추천 n: {n_recon}")
    print(f"입력 데이터 원본 차원: {X_scaled.shape[1]}")
    
    # 압축률 계산
    compression_ratio_pca = X_scaled.shape[1] / n_pca
    compression_ratio_recon = X_scaled.shape[1] / n_recon
    
    print(f"\nPCA 기반 압축률: {compression_ratio_pca:.2f}:1")
    print(f"재구성 오차 기반 압축률: {compression_ratio_recon:.2f}:1")

if __name__ == "__main__":
    main() 